"""Переиспользуемый цикл прогона кейсов: конкурентный fan-out + последовательный fold.

Зачем: раннеры на бэкенд-поиске (extractor_agent, дальше — sourcing_assistant,
one_line_search_query_builder) гоняют десятки I/O-bound кейсов, и всем нужно одно и
то же — пул воркеров, сбор результатов в ГЛАВНОМ потоке (чтобы безопасно копить
метрики/отчёт без локов), периодические чекпоинты и аккуратное сохранение по Ctrl+C.
Раньше это жило внутри extractor_agent; вынесено в core, чтобы новые раннеры
получали поведение даром.

Граница ответственности:
- сама «работа» (work) и «сборка» (fold) — за раннером (замыкания, доменная логика);
- любые сигналы между ними (напр. fail-fast «бэкенд лёг») раннер держит у себя как
  shared state (threading.Event и т.п.) — циклу про них знать НЕ нужно;
- цикл отвечает ТОЛЬКО за оркестрацию: параллелизм, порядок/чекпоинты, прерывание.

Это держит core свободным от доменной логики (см. контракт import-linter
qa_harness ⊥ app и границу core ↔ domain в docs/REFACTOR_PLAN.md §2): модуль
импортирует лишь stdlib.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class LoopOutcome:
    """Итог прогона. done — сколько кейсов реально свёрнуто (вызван fold)."""

    done: int
    total: int
    interrupted: bool


def run_cases(
    items: Sequence[T],
    work: Callable[[T], R],
    fold: Callable[[R], None],
    *,
    max_workers: int = 1,
    checkpoint_every: int = 0,
    on_checkpoint: Optional[Callable[[], None]] = None,
    on_interrupt: Optional[Callable[[], None]] = None,
) -> LoopOutcome:
    """Прогнать `items` через `work` (в воркерах) и `fold` (в вызывающем потоке).

    work(item) -> result   выполняется в пуле потоков (I/O-bound). ДОЛЖЕН ловить
                           ожидаемые ошибки и возвращать их как данные; исключение из
                           work всплывёт и остановит прогон — это сигнал бага, не норма.
    fold(result) -> None   вызывается строго ПОСЛЕДОВАТЕЛЬНО в вызывающем потоке по
                           мере готовности — здесь безопасно копить отчёт/метрики/
                           счётчики без локов.
    max_workers <= 1       последовательный режим: порядок `items` сохраняется и
                           детерминирован (важно для офлайна/воспроизводимости).
    max_workers >  1       пул потоков; fold вызывается в порядке ЗАВЕРШЕНИЯ задач.
    checkpoint_every N     вызвать on_checkpoint() после каждых N свёрнутых кейсов
                           (0 — промежуточных чекпоинтов нет; финальный flush — за
                           раннером после возврата).
    on_interrupt           вызывается один раз при KeyboardInterrupt (напр. печать).

    Возвращает LoopOutcome(done, total, interrupted).
    """
    total = len(items)
    done = 0
    interrupted = False

    def _maybe_checkpoint() -> None:
        if checkpoint_every and on_checkpoint and done % checkpoint_every == 0:
            on_checkpoint()

    # Последовательный режим: дешевле (без потоков) и детерминирован по порядку.
    if max_workers <= 1:
        try:
            for item in items:
                fold(work(item))
                done += 1
                _maybe_checkpoint()
        except KeyboardInterrupt:
            interrupted = True
            if on_interrupt:
                on_interrupt()
        return LoopOutcome(done=done, total=total, interrupted=interrupted)

    # Конкурентный режим: fan-out в пул, fold — в этом потоке по мере готовности.
    ex = ThreadPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(work, item) for item in items]
    try:
        for fut in as_completed(futures):
            fold(fut.result())
            done += 1
            _maybe_checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        for f in futures:
            f.cancel()
        if on_interrupt:
            on_interrupt()
    finally:
        ex.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return LoopOutcome(done=done, total=total, interrupted=interrupted)

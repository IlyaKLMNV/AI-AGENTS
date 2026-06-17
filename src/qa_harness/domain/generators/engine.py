"""Оркестратор вариативной генерации: produce → validate → retry → fallback (+ трасса).

Общий движок, который дёргают раннеры/генераторы. Инкапсулирует единый паттерн всех легаси-
генераторов (scenarios/autofill/guardrails): сгенерить кандидата, провалидировать против констрейнтов,
при провале — повторить с подсказкой (усилить/не повторяться), исчерпав попытки — детерминированный
fallback. Сам не ходит в сеть и не знает про конкретный домен: вызывающий передаёт `produce` (замыкает
контекст: историю диалога, spec, клиент), `validate` и опциональный `fallback`.

quality ≠ infra: исчерпание retry+fallback не бросает — возвращает `GenResult(source="failed")`, раннер
сам решает (обычно → errors). Usage генерации копится в `GenResult.usage` (отдельно от usage оценки).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from qa_harness.core.usage import accumulate_usage, blank_usage


@dataclass
class GenerationPolicy:
    """Политика генерации с валидацией."""

    max_retries: int = 1            # повторов СВЕРХ первой попытки (легаси: 1)
    temperature: Optional[float] = None  # справочно для трассы; применяет сам produce
    seed: Optional[int] = None      # справочно для трассы; сэмплинг конфигов — у вызывающего
    allow_fallback: bool = True


@dataclass
class Attempt:
    """Контекст одной попытки — передаётся в produce, чтобы скорректировать запрос."""

    index: int                      # 0-based номер попытки
    last_error: Optional[str]       # почему предыдущая попытка не прошла валидацию (None на первой)
    avoid: List[Any]                # ранее отвергнутые варианты — не повторять (avoid_repeating)


@dataclass
class GenResult:
    """Результат генерации + трасса для отчёта."""

    item: Any                       # принятый вариант (или None, если source="failed")
    source: str                     # "llm" | "fallback" | "failed"
    attempts: int                   # сколько раз дёрнули produce
    errors: List[str] = field(default_factory=list)   # ошибки валидации/produce по попыткам
    usage: dict = field(default_factory=blank_usage)  # накопленный usage генерации

    @property
    def ok(self) -> bool:
        return self.source != "failed"


def generate_valid(
    produce: Callable[[Attempt], Tuple[Any, Any]],
    validate: Optional[Callable[[Any], Optional[str]]] = None,
    *,
    policy: Optional[GenerationPolicy] = None,
    fallback: Optional[Callable[[], Optional[Any]]] = None,
    avoid: Optional[List[Any]] = None,
) -> GenResult:
    """Сгенерировать валидный вариант с ретраями и фолбэком.

    produce(attempt) -> (item, usage): один вызов генератора; usage может быть None.
        attempt.last_error / attempt.avoid позволяют скорректировать промпт (усилить, не повторяться).
    validate(item) -> None | str: None == ок, строка == причина отказа (уйдёт в errors и в подсказку).
    fallback() -> item | None: детерминированный запасной вариант, когда LLM не справился.
    avoid: стартовый список вариантов, которых надо избегать (напр. уже принятые в этом диалоге).
    """
    policy = policy or GenerationPolicy()
    errors: List[str] = []
    usage = blank_usage()
    avoid_list: List[Any] = list(avoid or [])
    total = max(1, policy.max_retries + 1)

    for i in range(total):
        attempt = Attempt(index=i, last_error=(errors[-1] if errors else None), avoid=avoid_list)
        try:
            item, u = produce(attempt)
        except Exception as e:  # noqa: BLE001 — сбой генерации не должен ронять прогон
            errors.append(f"produce:{type(e).__name__}:{e}")
            continue
        accumulate_usage(usage, u)
        err = validate(item) if validate is not None else None
        if err is None:
            return GenResult(item=item, source="llm", attempts=i + 1, errors=errors, usage=usage)
        errors.append(err)
        avoid_list.append(item)

    if policy.allow_fallback and fallback is not None:
        try:
            fb = fallback()
        except Exception as e:  # noqa: BLE001
            errors.append(f"fallback:{type(e).__name__}:{e}")
            fb = None
        if fb is not None:
            return GenResult(item=fb, source="fallback", attempts=total, errors=errors, usage=usage)

    return GenResult(item=None, source="failed", attempts=total, errors=errors, usage=usage)

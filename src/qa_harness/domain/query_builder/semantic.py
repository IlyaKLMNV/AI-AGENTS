"""Семантика one-line запроса по golden: обязательные термины (expect) и запреты (forbid).

Контракт (checks.py) проверяет ФОРМУ запроса; здесь — СМЫСЛ: попали ли в запрос
ключевые термины вакансии и не попали ли запрещённые. Проверка по подстроке на
нормализованной строке (lower + ё→е), чтобы не быть хрупкой к точным формулировкам и
пунктуации boolean-запроса.

expect — список ИЛИ-групп: каждая группа удовлетворена, если в запросе есть ХОТЬ ОДНА
из её форм (например, [django, flask] — достаточно любого из веб-фреймворков).
Одиночный термин можно писать строкой — он трактуется как группа из одного элемента.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple


def _norm(text: str) -> str:
    return str(text or "").lower().replace("ё", "е")


def _as_group(item: Any) -> List[str]:
    """Нормализовать элемент expect к списку форм (строка -> группа из одной формы)."""
    if isinstance(item, (list, tuple)):
        return [str(x) for x in item if str(x).strip()]
    return [str(item)] if str(item).strip() else []


def check_query_semantics(
    query: str,
    expect: Sequence[Any],
    forbid: Sequence[Any],
) -> Tuple[bool, List[str]]:
    """Вернуть (ok, diffs). diffs — машиночитаемые расхождения для триажа."""
    norm_q = _norm(query)
    diffs: List[str] = []

    for item in expect or []:
        group = _as_group(item)
        if group and not any(_norm(form) in norm_q for form in group):
            diffs.append(f"missing:{'|'.join(group)}")

    for term in forbid or []:
        t = _norm(term)
        if t and t in norm_q:
            diffs.append(f"forbidden:{term}")

    return (len(diffs) == 0), diffs

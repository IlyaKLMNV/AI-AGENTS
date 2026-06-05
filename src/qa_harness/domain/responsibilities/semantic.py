"""Семантика responsibilities_parser по golden + заземление.

Контракт (contract.py) проверяет ФОРМУ списка; здесь — СМЫСЛ:
- `check_semantics(predicted, expect, forbid)` (gate): ожидаемые термины извлечены, запрещённые — нет.
  expect — список ИЛИ-групп: группа удовлетворена, если хоть одна её форма совпала с каким-то ключевым
  словом (по нормализованной подстроке в обе стороны).
- `grounding_misses(predicted, vacancy_text)` (СИГНАЛ, не gate): какие ключевые слова не найдены в тексте
  вакансии (возможная галлюцинация). Возвращается как предупреждение в отчёт, качество не валит.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple


def _norm(s: str) -> str:
    t = (s or "").strip().lower().replace("ё", "е")
    return re.sub(r"[^a-z0-9а-я]+", "", t)


def _as_forms(item: Any) -> List[str]:
    if isinstance(item, (list, tuple)):
        return [str(x) for x in item if str(x).strip()]
    return [str(item)] if str(item).strip() else []


def _match(form: str, key: str) -> bool:
    f, k = _norm(form), _norm(key)
    return bool(f and k and (f in k or k in f))


def check_semantics(
    predicted: List[str],
    expect: Sequence[Any],
    forbid: Sequence[Any],
) -> Tuple[bool, List[str]]:
    """Вернуть (ok, diffs). expect — ИЛИ-группы обязательных терминов, forbid — запреты."""
    keys = [k for k in predicted if _norm(k)]
    diffs: List[str] = []
    for item in expect or []:
        forms = _as_forms(item)
        if forms and not any(_match(f, k) for f in forms for k in keys):
            diffs.append(f"missing:{'|'.join(forms)}")
    for term in forbid or []:
        if str(term).strip() and any(_match(term, k) for k in keys):
            diffs.append(f"forbidden:{term}")
    return (len(diffs) == 0), diffs


def grounding_misses(predicted: List[str], vacancy_text: str) -> List[str]:
    """Ключевые слова, не найденные в тексте вакансии (нормализованная подстрока). Сигнал, не gate."""
    tx = _norm(vacancy_text)
    return [k for k in predicted if _norm(k) and _norm(k) not in tx]

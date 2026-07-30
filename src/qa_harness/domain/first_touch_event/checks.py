"""Эвристики first_touch_event: приветствие, финальный вопрос про регистрацию, лишние числа, заземление.

Перенос из легаси: greeting «Имя, здравствуйте!»; финальная строка — короткий (≤14 слов) вопрос про
ссылку/регистрацию; extra_numbers разрешает только 4 («4 апреля»). facts_present_heuristic — офлайн-замена
судьи (ключевые эталонные термины в сообщении).
"""

from __future__ import annotations

import re
from typing import List

from .reference import REQUIRED_REFERENCE_TERMS

_ALLOWED_NUMBERS = {4}  # «4 апреля» — единственное допустимое число


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9а-я ]+", " ", (s or "").lower().replace("ё", "е"))).strip()


def greeting_ok(message: str, candidate_name: str) -> bool:
    name = (candidate_name or "").strip()
    expected = f"{name}, здравствуйте!" if name else "Здравствуйте!"
    return (message or "").strip().startswith(expected)


def final_question_ok(message: str) -> bool:
    lines = [ln.strip() for ln in (message or "").split("\n") if ln.strip()]
    if not lines:
        return False
    last = lines[-1]
    if not last.endswith("?"):
        return False
    if len(re.findall(r"\w+", last, flags=re.UNICODE)) > 14:
        return False
    return bool(re.search(r"(ссыл|регистрац)", last.lower()))


def extra_numbers(message: str) -> List[int]:
    found = {int(n) for n in re.findall(r"\d+", message or "")}
    return sorted(found - _ALLOWED_NUMBERS)


def facts_present_heuristic(message: str) -> List[str]:
    """Офлайн-стенд-ин судьи: какие ключевые эталонные термины НЕ найдены в сообщении (= missing)."""
    nm = _norm(message)
    return [t for t in REQUIRED_REFERENCE_TERMS if _norm(t) not in nm]

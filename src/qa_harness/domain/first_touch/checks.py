"""Эвристики first_touch: лишние числа, утечка скрытой компании, офлайн-стенд-ин судьи.

- extra_numbers: числа в сообщении, которых нет во входных фактах (защита от выдуманной зарплаты/цифр).
  Гейтим только числа длиной ≥5 (зарплата-величина), чтобы не ловить ложно годы/мелкие счётчики.
- company_name_leaked: при company_hidden оригинальное название компании НЕ должно встречаться.
- facts_present_heuristic: офлайн-замена LLM-судьи (значимые токены факта встречаются в сообщении).
"""

from __future__ import annotations

import re
from typing import Dict, List

_SALARY_MIN_LEN = 5


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9а-я ]+", " ", (s or "").lower().replace("ё", "е")).strip()


def extra_numbers(fact_values: List[str], message: str) -> List[str]:
    """Числа (≥5 цифр) в сообщении, отсутствующие во входных фактах."""
    allowed = set()
    for v in fact_values:
        allowed.update(re.findall(r"\d+", str(v)))
    return [n for n in re.findall(r"\d+", message or "") if len(n) >= _SALARY_MIN_LEN and n not in allowed]


def company_name_leaked(company_name: str, message: str) -> bool:
    """True, если название компании встречается в сообщении (для company_hidden-кейсов)."""
    name = _norm(company_name)
    return bool(name) and name in _norm(message)


def facts_present_heuristic(expected_facts: Dict[str, str], message: str) -> Dict[str, bool]:
    """Офлайн-стенд-ин LLM-судьи: факт «есть», если ≥половины значимых токенов значения в сообщении."""
    norm_msg = _norm(message)
    out: Dict[str, bool] = {}
    for k, v in (expected_facts or {}).items():
        toks = [t for t in _norm(str(v)).split() if len(t) >= 3]
        if not toks:
            out[k] = True
        else:
            hits = sum(1 for t in toks if t in norm_msg)
            out[k] = hits >= max(1, len(toks) // 2)
    return out

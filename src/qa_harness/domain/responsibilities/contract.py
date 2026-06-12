"""Контракт вывода responsibilities_parser: JSON-массив 0..5 проверяемых ТРЕБОВАНИЙ-предложений.

Новая задача промпта: вернуть 0..5 проверяемых требований ОДНИМ ПРЕДЛОЖЕНИЕМ каждое (а не keywords).
Пустой массив `[]` валиден (если обязательных требований в вакансии нет). Это «форма» (gate):
- строгий JSON-массив строк, длина 0..5, без дублей;
- каждый элемент непуст, без переносов, разумной длины (≤250);
- НЕ keyword-подобен (ловит регресс: старый промпт возвращал ["Python","Django"]);
- НЕ объединяет несколько независимых критериев в одно требование (atomicity).
Смысл (покрыты ли expected-критерии, заземление) — semantic.py. Проверка «одно предложение» — мягкая, как сигнал.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

MAX_LEN = 250

# Слова-индикаторы требования: их наличие отличает требование-предложение от голого keyword.
_REQ_INDICATORS = (
    "опыт", "знани", "работ", "разработ", "владени", "владе", "умени", "умеет", "готовнос",
    "образовани", "сертификат", "понимани", "навык", "есть", "знает", "кандидат", "лет ", "год",
)
# Технологические токены (Node.js, CI/CD, gRPC, PostgreSQL) — для проверки atomicity (multi-criteria).
_TECH_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+.#/_-]{1,}")


def parse_keywords(raw: str) -> List[str]:
    """Строгий JSON-массив строк (0..5 требований). ValueError, если это не массив строк."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty output")
    if not s.startswith("[") or not s.endswith("]"):
        raise ValueError("output is not a strict JSON array")
    obj = json.loads(s)
    if not isinstance(obj, list):
        raise ValueError("output JSON is not a list")
    out: List[str] = []
    for i, v in enumerate(obj):
        if not isinstance(v, str):
            raise ValueError(f"item[{i}] is not a string")
        out.append(v.strip())
    return out


def _norm_key(s: str) -> str:
    t = (s or "").strip().lower().replace("ё", "е")
    return re.sub(r"[^a-z0-9а-я]+", "", t)


def is_keyword_like(item: str) -> bool:
    """True, если строка похожа на старый keyword-ответ (1..3 слова без слов-индикаторов требования)."""
    t = (item or "").strip()
    if not t:
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if not (1 <= len(words) <= 3):
        return False
    low = t.lower()
    return not any(ind in low for ind in _REQ_INDICATORS)


def tech_tokens(item: str) -> List[str]:
    """Уникальные технологические токены (Node.js, CI/CD, gRPC…) — для проверки atomicity."""
    seen: set = set()
    out: List[str] = []
    for raw in _TECH_TOKEN.findall(item or ""):
        tok = raw.rstrip(".,/;:-_")
        k = tok.lower()
        if len(tok) >= 2 and k not in seen:
            seen.add(k)
            out.append(tok)
    return out


def is_multi_criteria(item: str) -> bool:
    """True, если одно требование объединяет 3+ независимых критерия (≥3 разных тех-токена)."""
    return len(tech_tokens(item)) >= 3


def validate_requirement(item: str) -> List[str]:
    """Правила одного требования-предложения: непусто, без переносов, ≤250, не keyword, не multi-criteria."""
    errors: List[str] = []
    t = (item or "").strip()
    if not t:
        return ["empty_requirement"]
    if "\n" in item or "\r" in item:
        errors.append("requirement_has_newline")
    if len(t) > MAX_LEN:
        errors.append("requirement_too_long")
    if is_keyword_like(t):
        errors.append("keyword_like_requirement")
    if is_multi_criteria(t):
        errors.append("multi_criteria_requirement")
    return errors


def find_duplicates(items: List[str]) -> List[str]:
    seen: set = set()
    dups: List[str] = []
    for item in items:
        k = _norm_key(item)
        if not k:
            continue
        if k in seen:
            dups.append(item)
        else:
            seen.add(k)
    return dups


def check_contract(predicted: List[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Вернуть (passed, issues, details). passed = длина 0..5 + форма каждого требования + нет дублей."""
    issues: List[str] = []
    item_errors: Dict[str, List[str]] = {}
    for item in predicted:
        errs = validate_requirement(item)
        if errs:
            item_errors[item or "<empty>"] = errs
    dups = find_duplicates(predicted)

    if not (0 <= len(predicted) <= 5):
        issues.append("list_len_not_0_5")
    # агрегируем коды по типам, чтобы reason_codes были информативны
    flat_codes = {c for errs in item_errors.values() for c in errs}
    for code in ("empty_requirement", "requirement_has_newline", "requirement_too_long",
                 "keyword_like_requirement", "multi_criteria_requirement"):
        if code in flat_codes:
            issues.append(code)
    if dups:
        issues.append("duplicate_requirements")

    details = {"requirements_count": len(predicted), "item_errors": item_errors, "duplicates": dups}
    return len(issues) == 0, issues, details

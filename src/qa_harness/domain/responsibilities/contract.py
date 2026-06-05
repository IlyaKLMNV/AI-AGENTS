"""Контракт вывода responsibilities_parser: строгий JSON-массив 1..5 коротких терминов.

Промпт получает текст вакансии и обязан вернуть JSON-массив СТРОК (ключевые требования/навыки),
1..5 штук, каждый термин — 1..3 слова, без чисел-одиночек, без запятых/точек с запятой, ≤60 символов,
без дублей. Это «форма» (gate). Смысл (попали ли ожидаемые термины, заземлены ли в тексте) — semantic.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


def parse_keywords(raw: str) -> List[str]:
    """Строгий JSON-массив строк. ValueError, если форма нарушена."""
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


def validate_item_format(item: str) -> List[str]:
    """Правила формата одного термина: 1..3 слова, без чисел-одиночек/запятых, ≤60 символов."""
    errors: List[str] = []
    t = (item or "").strip()
    if not t:
        return ["empty_item"]
    # числовой токен без букв (напр. "5", "2024") — не термин
    if any(re.search(r"\d", tok) and not re.search(r"[A-Za-zА-Яа-я]", tok) for tok in re.split(r"\s+", t)):
        errors.append("standalone_numeric_token")
    if "," in t or ";" in t:
        errors.append("comma_or_semicolon")
    words = [w for w in re.split(r"\s+", t) if w]
    if not (1 <= len(words) <= 3):
        errors.append(f"word_count={len(words)}(expected 1..3)")
    if len(t) > 60:
        errors.append("too_long(>60)")
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
    """Вернуть (passed, issues, details). passed = длина 1..5 + формат каждого + нет дублей."""
    issues: List[str] = []
    item_format_errors: Dict[str, List[str]] = {}
    for item in predicted:
        errs = validate_item_format(item)
        if errs:
            item_format_errors[item] = errs
    dups = find_duplicates(predicted)

    if not (1 <= len(predicted) <= 5):
        issues.append("list_len_not_1_5")
    if item_format_errors:
        issues.append("item_format_failed")
    if dups:
        issues.append("duplicate_keywords")

    details = {
        "keywords_count": len(predicted),
        "item_format_errors": item_format_errors,
        "duplicates": dups,
    }
    return len(issues) == 0, issues, details

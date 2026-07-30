"""Хард-правила формата one-line запроса + детектор утечек.

Это собственный «контракт» промпта-билдера (аналог step1-contract у extractor):
проверяется по самой строке запроса, ничего больше не нужно — то есть offline.
Семантика (покрытие ожидаемых терминов по golden) — отдельно в semantic.py.

Утечки — то, чему НЕ место в поисковом запросе по кандидатам: формат работы,
деньги/компенсация, процесс найма. Билдер обязан их вычищать из запроса, даже если
они есть в исходной вакансии.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern

LEAKAGE_PATTERNS: Dict[str, Pattern[str]] = {
    "work_format_mentioned": re.compile(
        r"\b(remote|hybrid|onsite|on-site|office)\b|удал[её]н|гибрид|офис|на месте работодателя",
        re.IGNORECASE,
    ),
    "salary_or_compensation_mentioned": re.compile(
        r"[₽$€]|зарплат|оклад|компенсац|вилка|бонус|\bgross\b|\bnet\b|\bkpi\b|\bруб(?:\.|ля|лей)?\b",
        re.IGNORECASE,
    ),
    "application_process_mentioned": re.compile(
        r"портфоли|сопровод|анкет|отклик|тестов|test task|cover letter|portfolio",
        re.IGNORECASE,
    ),
}


def build_query_checks(query: str) -> Dict[str, Any]:
    """Формальные проверки строки запроса: не пусто / одна строка / без JSON-скобок."""
    errors: List[str] = []
    stripped = (query or "").strip()
    if not stripped:
        errors.append("empty_query")
    if "\n" in stripped or "\r" in stripped:
        errors.append("query_not_single_line")
    if any(ch in stripped for ch in "{}[]"):
        errors.append("json_like_output")
    words = [w for w in stripped.split() if w]
    return {"ok": not errors, "errors": errors, "word_count": len(words), "char_count": len(stripped)}


def detect_leakage(query: str) -> List[str]:
    """Имена сработавших правил-утечек (формат работы / зарплата / процесс найма)."""
    return [name for name, pat in LEAKAGE_PATTERNS.items() if pat.search(query or "")]

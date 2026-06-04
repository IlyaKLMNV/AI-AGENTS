"""Контракт вывода sourcing_assistant: массив объектов {requirement, comment, passed} 1:1 к требованиям.

Промпт получает {requirements:[...], profile:{...}} и обязан вернуть JSON-массив той же длины,
по одному объекту на требование, В ТОМ ЖЕ ПОРЯДКЕ; каждый объект — РОВНО {requirement, comment, passed},
где requirement — точный echo требования, passed ∈ {0,1}.

Семантической оценки («реально ли кандидат подходит») здесь НЕТ: кандидаты — живые профили из
backend без разметки, поэтому проверяется только ФОРМА ответа (как и в легаси-раннере).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

ALLOWED_KEYS = {"requirement", "comment", "passed"}


def parse_sourcing_output(raw: str) -> List[Dict[str, Any]]:
    """Распарсить вывод промпта в список объектов. ValueError, если это не JSON-массив объектов."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty output")
    if not s.startswith("["):
        m = re.search(r"\[\s*(?:.|\n)*\s*\]", s)
        if not m:
            raise ValueError("output is not a JSON array")
        s = m.group(0)
    obj = json.loads(s)
    if not isinstance(obj, list):
        raise ValueError("output JSON is not a list")
    out: List[Dict[str, Any]] = []
    for i, v in enumerate(obj):
        if not isinstance(v, dict):
            raise ValueError(f"item[{i}] is not an object")
        out.append(v)
    return out


def _validate_item_shape(item: Dict[str, Any]) -> List[str]:
    """Строго: ровно {requirement, comment, passed}; passed ∈ {0,1}; строки — строки."""
    reasons: List[str] = []
    extra = sorted(set(item.keys()) - ALLOWED_KEYS)
    missing = sorted(ALLOWED_KEYS - set(item.keys()))
    if extra:
        reasons.append(f"extra_keys={extra}")
    if missing:
        reasons.append(f"missing_keys={missing}")
    if "requirement" in item and not isinstance(item["requirement"], str):
        reasons.append("requirement_not_string")
    if "comment" in item and not isinstance(item["comment"], str):
        reasons.append("comment_not_string")
    if "passed" in item and (not isinstance(item["passed"], int) or item["passed"] not in (0, 1)):
        reasons.append("passed_not_0_1")
    return reasons


def check_contract(
    requirements: List[str],
    predicted: List[Dict[str, Any]],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Вернуть (passed, issues, details). passed — вывод 1:1 к требованиям и каждый элемент по форме."""
    issues: List[str] = []
    failed_items: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {
        "requirements_count": len(requirements),
        "output_count": len(predicted),
        "missing_items_count": max(0, len(requirements) - len(predicted)),
        "extra_items_count": max(0, len(predicted) - len(requirements)),
        "failed_items_count": 0,
        "shape_fail_count": 0,
        "requirement_not_exact_count": 0,
    }

    if len(predicted) != len(requirements):
        issues.append("length_mismatch")

    n = min(len(predicted), len(requirements))
    for i in range(n):
        item = predicted[i]
        exp_req = requirements[i]
        if not isinstance(item, dict):
            checks["shape_fail_count"] += 1
            failed_items.append({"index": i, "expected_requirement": exp_req, "issues": ["item_not_object"]})
            continue

        item_issues = _validate_item_shape(item)
        if item_issues:
            checks["shape_fail_count"] += 1
        if isinstance(item.get("requirement"), str) and item["requirement"] != exp_req:
            item_issues.append("requirement_not_exact")
            checks["requirement_not_exact_count"] += 1
        if item_issues:
            failed_items.append({
                "index": i,
                "expected_requirement": exp_req,
                "actual_requirement": item.get("requirement"),
                "actual_passed": item.get("passed"),
                "actual_comment": item.get("comment"),
                "issues": item_issues,
            })

    checks["failed_items_count"] = len(failed_items)
    if checks["shape_fail_count"] > 0:
        issues.append("output_shape_failed")
    if checks["requirement_not_exact_count"] > 0:
        issues.append("requirement_not_exact")

    return len(issues) == 0, issues, {"checks": checks, "failed_items": failed_items}

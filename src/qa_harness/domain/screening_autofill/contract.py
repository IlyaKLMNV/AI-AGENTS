"""Контракт вывода screening_autofill: JSON-объект формы скрининга.

Промпт получает диалог рекрутер/кандидат и обязан вернуть JSON-объект:
{preferred_location: str, min_salary: str(цифры|""), max_salary: str(цифры|""),
 work_format: str ∈ {"", remote, office, hybrid}, additional_info: [{question: str, answer: str}]}.
Здесь — ФОРМА (наличие ключей/типы/enum/digits). Смысл (golden + анти-утечка) — semantic.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

WORK_FORMATS = ("", "remote", "office", "hybrid")
REQUIRED_KEYS = ("preferred_location", "min_salary", "max_salary", "work_format", "additional_info")


def _extract_json_object(text: str) -> Optional[str]:
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    return None


def parse_form(raw: str) -> Dict[str, Any]:
    """Распарсить вывод в dict (с фолбэком на вырезание {...}). ValueError, если не объект."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty output")
    try:
        obj = json.loads(s)
    except Exception:
        ex = _extract_json_object(s)
        if not ex:
            raise ValueError("output is not JSON")
        obj = json.loads(ex)
    if not isinstance(obj, dict):
        raise ValueError("output is not a JSON object")
    return obj


def _digits_or_empty(s: Any) -> bool:
    if s is None:
        return True
    if not isinstance(s, str):
        return False
    return s == "" or s.isdigit()


def validate_schema(obj: Any) -> List[str]:
    """Ошибки формы (пусто = форма валидна)."""
    if not isinstance(obj, dict):
        return ["output_not_object"]
    errors: List[str] = []
    for k in REQUIRED_KEYS:
        if k not in obj:
            errors.append(f"missing_key:{k}")
    if "preferred_location" in obj and not isinstance(obj.get("preferred_location"), str):
        errors.append("preferred_location_not_string")
    if "work_format" in obj:
        wf = obj.get("work_format")
        if not isinstance(wf, str):
            errors.append("work_format_not_string")
        elif wf not in WORK_FORMATS:
            errors.append("work_format_invalid_value")
    if "min_salary" in obj and not _digits_or_empty(obj.get("min_salary")):
        errors.append("min_salary_not_digits_or_empty")
    if "max_salary" in obj and not _digits_or_empty(obj.get("max_salary")):
        errors.append("max_salary_not_digits_or_empty")
    if "additional_info" in obj:
        ai = obj.get("additional_info")
        if not isinstance(ai, list):
            errors.append("additional_info_not_list")
        else:
            for i, item in enumerate(ai):
                if not isinstance(item, dict):
                    errors.append(f"additional_info[{i}]_not_object")
                elif not isinstance(item.get("question"), str) or not isinstance(item.get("answer"), str):
                    errors.append(f"additional_info[{i}]_qa_not_strings")
    return errors

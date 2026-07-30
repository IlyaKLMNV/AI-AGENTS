"""Step1: контракт-валидация extractor_json (без LLM-судьи).

Перенесено дословно из extractor_agent_runner (validate_step1_contract + ALLOWED_*).
Ловит drift: лишние поля, неверные типы/значения. Возвращает (ok, errors, warnings).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

ALLOWED_TOP_LEVEL_FIELDS = {
    "skills",
    "positions",
    "locations",
    "companies",
    "languages",
    "level",
    "experience",
    "age",
    "management_experience",
    "higher_education",
    "business_spheres",
    "keywords",
}

ALLOWED_OPERATORS = {"AND", "OR", "NOT"}
ALLOWED_LEVELS = {"junior", "middle", "senior", "lead", "head", "c-level"}


def _is_int_or_none(x: Any) -> bool:
    return x is None or isinstance(x, int)


def validate_entity_list(obj: Any, field_name: str, errors: List[str]) -> Optional[List[dict]]:
    if obj is None:
        return None
    if not isinstance(obj, list):
        errors.append(f"{field_name}_must_be_array")
        return None

    out: List[dict] = []
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            errors.append(f"{field_name}[{i}]_must_be_object")
            continue
        raw_text = it.get("raw_text")
        op = it.get("operator")
        if not isinstance(raw_text, str) or not raw_text.strip():
            errors.append(f"{field_name}[{i}].raw_text_must_be_string")
            continue
        if not isinstance(op, str) or op not in ALLOWED_OPERATORS:
            errors.append(f"{field_name}[{i}].operator_invalid")
            continue
        out.append({"raw_text": raw_text.strip(), "operator": op})
    return out


def validate_step1_contract(extractor_json: Any, user_input: str) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if extractor_json is None:
        return False, ["step1_empty_json"], []
    if not isinstance(extractor_json, dict):
        return False, ["step1_json_must_be_object"], []

    for k in extractor_json.keys():
        if k not in ALLOWED_TOP_LEVEL_FIELDS:
            errors.append(f"step1_unknown_field:{k}")

    for f in ("skills", "positions", "locations", "companies", "business_spheres", "keywords"):
        if f in extractor_json:
            validate_entity_list(extractor_json.get(f), f, errors)

    if "languages" in extractor_json:
        lang = extractor_json.get("languages")
        if not isinstance(lang, dict):
            errors.append("languages_must_be_object")
        else:
            for k in lang.keys():
                if k not in ("russian", "english"):
                    errors.append(f"languages_unknown_field:{k}")
            for k in ("russian", "english"):
                if k in lang and not isinstance(lang.get(k), bool):
                    errors.append(f"languages_{k}_must_be_boolean")
            s = user_input.lower()
            if ("англий" in s and "желател" in s) and lang.get("english") is True:
                warnings.append("languages_english_soft_requirement_marked_required")

    if "level" in extractor_json:
        lvl = extractor_json.get("level")
        if not isinstance(lvl, list):
            errors.append("level_must_be_array")
        else:
            for i, v in enumerate(lvl):
                if not isinstance(v, str):
                    errors.append(f"level[{i}]_must_be_string")
                elif v.strip().lower() not in ALLOWED_LEVELS:
                    errors.append(f"level[{i}]_invalid_value")

    for f in ("experience", "age", "management_experience"):
        if f in extractor_json:
            r = extractor_json.get(f)
            if not isinstance(r, dict):
                errors.append(f"{f}_must_be_object")
            else:
                if "from" in r and not _is_int_or_none(r.get("from")):
                    errors.append(f"{f}.from_must_be_int_or_null")
                if "to" in r and not _is_int_or_none(r.get("to")):
                    errors.append(f"{f}.to_must_be_int_or_null")

    if "higher_education" in extractor_json and not isinstance(extractor_json.get("higher_education"), bool):
        errors.append("higher_education_must_be_boolean")

    return (len(errors) == 0), errors, warnings

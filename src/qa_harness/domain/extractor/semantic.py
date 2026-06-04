"""Семантическая проверка извлечения по golden-ожиданиям (expect/forbid).

Контракт (pipeline/contract.py) проверяет ФОРМУ; здесь — СМЫСЛ: попали ли ожидаемые
термины в нужные bucket'ы и не уехали ли запрещённые (город в positions и т.п.).
Проверка по подстроке (регистронезависимо), чтобы не быть хрупкой к точным формулировкам.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

ENTITY_BUCKETS = ("positions", "skills", "locations", "companies", "keywords", "business_spheres")
RANGE_BUCKETS = ("experience", "age", "management_experience")
# слова-форматы, которым не место в locations
FORMAT_WORDS = (
    "офис", "гибрид", "гибридный", "удаленно", "удалённо", "удаленка", "удалёнка",
    "remote", "hybrid", "onsite", "on-site",
)


def _raw_texts(ej: Dict[str, Any], bucket: str) -> List[str]:
    out: List[str] = []
    vals = ej.get(bucket)
    if isinstance(vals, list):
        for e in vals:
            if isinstance(e, dict) and isinstance(e.get("raw_text"), str):
                out.append(e["raw_text"].lower())
    return out


def check_semantics(
    extractor_json: Any,
    expect: Dict[str, Any],
    forbid: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """Вернуть (ok, diffs). diffs — машиночитаемые расхождения для триажа."""
    diffs: List[str] = []
    if not isinstance(extractor_json, dict):
        return False, ["no_extractor_json"]

    expect = expect or {}
    forbid = forbid or {}

    # expect: присутствие терминов в нужных entity-bucket'ах
    for bucket in ENTITY_BUCKETS:
        texts = _raw_texts(extractor_json, bucket)
        for term in expect.get(bucket, []) or []:
            if not any(str(term).lower() in t for t in texts):
                diffs.append(f"missing:{bucket}:{term}")

    # expect.level
    levels = [str(x).lower() for x in (extractor_json.get("level") or [])]
    for lvl in expect.get("level", []) or []:
        if str(lvl).lower() not in levels:
            diffs.append(f"missing:level:{lvl}")

    # expect.languages (точное совпадение флага)
    for k, v in (expect.get("languages") or {}).items():
        if (extractor_json.get("languages") or {}).get(k) != v:
            diffs.append(f"lang:{k}!={v}")

    # expect.experience/age/management_experience
    for bucket in RANGE_BUCKETS:
        exp = expect.get(bucket)
        if isinstance(exp, dict):
            got = extractor_json.get(bucket) or {}
            for k, v in exp.items():
                if got.get(k) != v:
                    diffs.append(f"range:{bucket}.{k}!={v}")

    # forbid: запрещённые размещения
    for bucket, terms in forbid.items():
        texts = _raw_texts(extractor_json, bucket)
        for term in terms or []:
            if any(str(term).lower() in t for t in texts):
                diffs.append(f"misplaced:{bucket}:{term}")

    # всегда: слово-формат осталось в locations
    for t in _raw_texts(extractor_json, "locations"):
        if any(fw in t for fw in FORMAT_WORDS):
            diffs.append(f"format_in_locations:{t}")

    return (len(diffs) == 0), diffs

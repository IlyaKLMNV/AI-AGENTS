"""Step2: маппинг extractor_json -> payload для backend /site/searchBool.

Перенесено дословно из extractor_agent_runner (build_step3_payload + helpers).
Без «добавления смысла»: только структурное преобразование сущностей в группы/флаги.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

OFFICE_GEO_TRASH = {
    "офис", "гибрид", "гибридный", "удаленно", "удалённо", "удаленка", "удалёнка",
    "remote", "hybrid", "onsite", "on-site",
}


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def op_to_group(op: str) -> str:
    return {"AND": "all", "OR": "or", "NOT": "not"}.get(op, "all")


def entities_to_groups(entities: Optional[List[Dict[str, Any]]]) -> List[List[Any]]:
    if not entities:
        return []
    by_group: Dict[str, List[str]] = {"all": [], "or": [], "not": []}
    for e in entities:
        rt = e.get("raw_text")
        op = e.get("operator")
        if not isinstance(rt, str) or not isinstance(op, str):
            continue
        by_group[op_to_group(op)].append(rt)

    groups: List[List[Any]] = []
    for g in ("all", "or", "not"):
        vals = dedupe_keep_order([v.strip() for v in by_group[g] if v.strip()])
        if vals:
            groups.append([g, vals])
    return groups


def sanitize_geos(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        vv = v.strip()
        if vv and vv.lower() not in OFFICE_GEO_TRASH:
            out.append(vv)
    return out


def map_level_to_seniority(levels: Any) -> Optional[List[str]]:
    if not isinstance(levels, list):
        return None
    mapping = {
        "junior": "Junior", "middle": "Middle", "senior": "Senior",
        "lead": "Lead", "head": "Head", "c-level": "C-Level",
    }
    mapped = [mapping[v.strip().lower()] for v in levels if isinstance(v, str) and v.strip().lower() in mapping]
    mapped = dedupe_keep_order(mapped)
    return mapped or None


def range_obj_to_str_pair(obj: Any) -> Optional[List[str]]:
    if not isinstance(obj, dict):
        return None
    f, t = obj.get("from"), obj.get("to")
    if f is None and t is None:
        return None
    if not (f is None or isinstance(f, int)):
        return None
    if not (t is None or isinstance(t, int)):
        return None
    return [str(f) if isinstance(f, int) else "", str(t) if isinstance(t, int) else ""]


def apply_languages_to_flags(payload: Dict[str, Any], extractor_json: Dict[str, Any]) -> None:
    lang = extractor_json.get("languages")
    if not isinstance(lang, dict):
        return
    if isinstance(lang.get("russian"), bool):
        payload["onlyRussian"] = lang["russian"]
    if isinstance(lang.get("english"), bool):
        payload["onlyEnglish"] = lang["english"]


def make_base_payload(
    *,
    only_russian: bool = False,
    only_english: bool = False,
    only_with_contacts: bool = False,
    only_with_higher_education: bool = False,
    current_position_title: bool = False,
    limit: int = 20,
    offset: int = 0,
    shuffle: bool = False,
    highlight: bool = False,
) -> Dict[str, Any]:
    """Базовый 9-ключевой payload backend-поиска (флаги выборки)."""
    return {
        "onlyRussian": bool(only_russian),
        "onlyEnglish": bool(only_english),
        "onlyWithContacts": bool(only_with_contacts),
        "onlyWithHigherEducation": bool(only_with_higher_education),
        "currentPositionTitle": bool(current_position_title),
        "limit": int(limit),
        "offset": int(offset),
        "shuffle": bool(shuffle),
        "highlight": bool(highlight),
    }


# entity-bucket extractor_json -> поле payload (для проверки покрытия step2)
_ENTITY_TO_PAYLOAD = {
    "positions": "positions",
    "skills": "skills",
    "locations": "geos",
    "companies": "firms",
    "keywords": "keys",
}


def _bucket_raw_texts(extractor_json: Dict[str, Any], bucket: str) -> List[str]:
    out: List[str] = []
    vals = extractor_json.get(bucket)
    if isinstance(vals, list):
        for e in vals:
            if isinstance(e, dict) and isinstance(e.get("raw_text"), str) and e["raw_text"].strip():
                out.append(e["raw_text"].strip().lower())
    return out


def _payload_group_terms(payload: Dict[str, Any], field: str) -> set:
    terms = set()
    for g in payload.get(field) or []:
        if isinstance(g, list) and len(g) == 2 and isinstance(g[1], list):
            for v in g[1]:
                terms.add(str(v).strip().lower())
    return terms


def mapping_report(extractor_json: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Проверка покрытия step2: что потеряно/санитизировано/не замаплено.

    - dropped: термин из extractor_json не доехал до payload (тихая потеря) — это плохо;
    - sanitized: слово-формат удалено из locations->geos — это ОК (информация);
    - unmapped_fields: bucket извлечён, но build_step3_payload его не использует (например,
      business_spheres) — сигнал, что часть извлечения игнорируется.
    """
    dropped: List[str] = []
    sanitized: List[str] = []
    for ebucket, pfield in _ENTITY_TO_PAYLOAD.items():
        out_terms = _payload_group_terms(payload, pfield)
        for t in _bucket_raw_texts(extractor_json, ebucket):
            if t in out_terms:
                continue
            if ebucket == "locations" and t in OFFICE_GEO_TRASH:
                sanitized.append(t)
            else:
                dropped.append(f"{ebucket}:{t}")
    unmapped_fields: List[str] = []
    if extractor_json.get("business_spheres"):
        unmapped_fields.append("business_spheres")
    return {"dropped": dropped, "sanitized": sanitized, "unmapped_fields": unmapped_fields}


def build_step3_payload(
    extractor_json: Dict[str, Any],
    user_phrase: str,
    base_payload: Dict[str, Any],
    sanitize_office_geo: bool,
) -> Dict[str, Any]:
    payload = dict(base_payload)
    payload["user_phrase"] = user_phrase

    for src_key, dst_key in (("positions", "positions"), ("skills", "skills"), ("companies", "firms"), ("keywords", "keys")):
        val = extractor_json.get(src_key)
        if isinstance(val, list):
            groups = entities_to_groups([x for x in val if isinstance(x, dict)])
            if groups:
                payload[dst_key] = groups

    loc = extractor_json.get("locations")
    if isinstance(loc, list):
        geos_groups = entities_to_groups([x for x in loc if isinstance(x, dict)])
        if sanitize_office_geo:
            sanitized: List[List[Any]] = []
            for g, vals in geos_groups:
                if isinstance(vals, list):
                    vals2 = sanitize_geos([str(v) for v in vals])
                    if vals2:
                        sanitized.append([g, vals2])
            geos_groups = sanitized
        if geos_groups:
            payload["geos"] = geos_groups

    seniority = map_level_to_seniority(extractor_json.get("level"))
    if seniority:
        payload["seniorityLevels"] = seniority

    if extractor_json.get("higher_education") is True:
        payload["onlyWithHigherEducation"] = True

    apply_languages_to_flags(payload, extractor_json)

    exp_pair = range_obj_to_str_pair(extractor_json.get("experience"))
    if exp_pair is not None:
        payload["experience"] = exp_pair
    mgmt_pair = range_obj_to_str_pair(extractor_json.get("management_experience"))
    if mgmt_pair is not None:
        payload["managementExperience"] = mgmt_pair
    age_pair = range_obj_to_str_pair(extractor_json.get("age"))
    if age_pair is not None:
        payload["ages"] = age_pair

    return payload

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from extractor_agent_runner import (
    BackendCfg,
    PromptCfg,
    build_step3_payload,
    call_backend_search_bool,
    call_openai_step1,
    deep_get,
    ensure_dir,
    load_yaml,
    make_run_id,
    utc_now_iso,
    validate_step1_contract,
)


LEVELS = ("junior", "middle", "senior", "lead", "head", "c-level")
RUSSIAN_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "его",
    "ого",
    "ому",
    "ему",
    "ыми",
    "ими",
    "иях",
    "ях",
    "ах",
    "ия",
    "ья",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ых",
    "их",
    "ым",
    "им",
    "ов",
    "ев",
    "ей",
    "ам",
    "ям",
    "ом",
    "ем",
    "ы",
    "и",
    "а",
    "я",
    "у",
    "ю",
    "о",
    "е",
)
TOKEN_CANONICAL_MAP = {
    "developer": "developer",
    "developers": "developer",
    "develop": "developer",
    "разработчик": "developer",
    "разработч": "developer",
    "engineer": "engineer",
    "engineers": "engineer",
    "инженер": "engineer",
    "инженерн": "engineer",
    "platform": "platform",
    "platforms": "platform",
    "платформ": "platform",
    "network": "network",
    "сетев": "network",
    "сеть": "network",
    "monitoring": "monitoring",
    "monitor": "monitoring",
    "мониторинг": "monitoring",
    "логирован": "logging",
    "logging": "logging",
    "process": "process",
    "процесс": "process",
    "system": "system",
    "systems": "system",
    "систем": "system",
    "manager": "manager",
    "менеджер": "manager",
}
GENERIC_MATCH_TOKENS = {
    "and",
    "or",
    "with",
    "from",
    "for",
    "the",
    "a",
    "an",
    "и",
    "или",
    "с",
    "со",
    "в",
    "во",
    "на",
    "по",
    "из",
    "от",
    "до",
    "опыт",
    "знан",
    "пониман",
    "экспертиз",
    "работ",
    "обязан",
    "обязател",
    "необходим",
}
REAL_GEO_BLACKLIST = {
    "remote",
    "hybrid",
    "office",
    "удаленно",
    "удалённо",
    "удаленка",
    "удалёнка",
    "гибрид",
    "гибридно",
    "гибридный",
    "офис",
}
LANGUAGE_MARKERS = {
    "english": ("english", "английский"),
    "russian": ("russian", "русский"),
}
FORBIDDEN_PATTERNS = {
    "work_format_mentioned": re.compile(
        r"\b(remote|hybrid|onsite|on-site|office)\b|удал[её]н|гибрид|офис",
        re.IGNORECASE,
    ),
    "salary_or_compensation_mentioned": re.compile(
        r"[₽$€]|зарплат|оклад|компенсац|вилка|бонус|\bgross\b|\bnet\b|\bkpi\b|\bруб(?:\.|ля|лей)?\b",
        re.IGNORECASE,
    ),
    "benefits_or_marketing_mentioned": re.compile(
        r"дмс|отпуск|стомат|страхов|тимбил|корпоратив|команда|интересн|амбициозн|"
        r"обучен|курс|книг|преми|печеньк|кофе",
        re.IGNORECASE,
    ),
}
INDUSTRY_ALIASES = {
    "fintech": ["fintech", "финтех", "фин сектор", "банки", "banking"],
    "traveltech": ["traveltech", "travel", "экскур", "бронирования"],
    "media": ["media", "онлайн-кинотеатр", "стриминг", "video"],
    "robotics": ["robotics", "робототех", "складск", "логистик"],
    "manufacturing": ["manufacturing", "производств", "композит"],
    "digital content": ["digital content", "content", "контент", "видео", "reels"],
    "tech": ["tech", "it", "technology"],
    "it services": ["it", "frontend", "веб"],
    "business operations": ["operations", "операцион", "office management"],
}


@dataclass
class CdmCase:
    name: str
    path: Path
    raw_vacancy: str
    vacancy: Dict[str, Any]


@dataclass
class CaseResult:
    name: str
    cdm_file: str
    vacancy_title: str
    generated_query: str
    status: str
    passed: bool
    builder: Dict[str, Any]
    extractor: Dict[str, Any]
    semantic: Dict[str, Any]
    backend: Dict[str, Any]
    extractor_json: Optional[Dict[str, Any]] = None
    step3_payload: Optional[Dict[str, Any]] = None


def parse_steps(raw: str) -> List[int]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    out: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except Exception:
            continue
        if value in (1, 2, 3) and value not in out:
            out.append(value)
    if not out:
        out = [1, 2, 3]
    if out not in ([1], [1, 2], [1, 2, 3]):
        raise SystemExit("--steps must be one of: 1 | 1,2 | 1,2,3")
    return out


def resolve_prompt_from_cfg(
    cfg: Dict[str, Any],
    section_name: str,
    env_prefix: str,
    default_model: str,
    override_prompt_id: str,
    override_prompt_version: str,
    override_model: str,
) -> PromptCfg:
    block = cfg.get(section_name) if isinstance(cfg.get(section_name), dict) else {}
    cfg_pid = block.get("prompt_id") or block.get("promptId")
    cfg_pver = block.get("prompt_version") or block.get("promptVersion")
    env_pid = os.getenv(f"{env_prefix}_PROMPT_ID") or None
    env_pver = os.getenv(f"{env_prefix}_PROMPT_VERSION") or None
    model = (
        override_model
        or deep_get(cfg, ["model"])
        or deep_get(cfg, ["openai", "model"])
        or default_model
    )
    return PromptCfg(
        prompt_id=(override_prompt_id or "").strip() or (str(cfg_pid) if cfg_pid else None) or env_pid,
        prompt_version=(override_prompt_version or "").strip() or (str(cfg_pver) if cfg_pver else None) or env_pver,
        model=str(model),
    )


def normalize_text(text: str) -> str:
    value = str(text or "").lower().replace("ё", "е")
    value = re.sub(r"[^a-zа-я0-9+#]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def stem_token(token: str) -> str:
    value = str(token or "").strip().lower().replace("ё", "е")
    if not value:
        return ""
    if re.fullmatch(r"[a-z0-9+#]+", value):
        if value.endswith("ies") and len(value) > 5:
            return value[:-3] + "y"
        if value.endswith("ing") and len(value) > 6:
            return value[:-3]
        if value.endswith("es") and len(value) > 5:
            return value[:-2]
        if value.endswith("s") and len(value) > 4 and value not in {"js", "ts"}:
            return value[:-1]
        return value
    for suffix in RUSSIAN_SUFFIXES:
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)]
    return value


def canonicalize_token(token: str) -> str:
    stemmed = stem_token(token)
    return TOKEN_CANONICAL_MAP.get(stemmed, stemmed)


def tokenize(text: str) -> List[str]:
    out: List[str] = []
    for tok in normalize_text(text).split():
        canonical = canonicalize_token(tok)
        if canonical and canonical not in GENERIC_MATCH_TOKENS:
            out.append(canonical)
    return out


def split_csv_like(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value]
    else:
        items = [str(value).strip()]
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def extract_experience(raw_vacancy: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    text = raw_vacancy or ""
    m = re.search(r"Опыт работы:\s*более\s+(\d+)\s+лет", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), None
    m = re.search(r"Опыт работы:\s*(\d+)\s*[–-]\s*(\d+)\s*(?:лет|года)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"Опыт работы:\s*от\s*(\d+)\s+лет", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), None
    return None


def extract_languages(raw_vacancy: str) -> List[str]:
    low = normalize_text(raw_vacancy)
    out: List[str] = []
    for lang, markers in LANGUAGE_MARKERS.items():
        if any(marker in low for marker in markers):
            out.append(lang)
    return out


def extract_levels(texts: Sequence[str]) -> List[str]:
    low = " ".join(normalize_text(t) for t in texts if t)
    return [lvl for lvl in LEVELS if lvl in low]


def build_industry_evidence(industry: str, raw_vacancy: str) -> List[str]:
    out: List[str] = []
    if industry:
        out.append(industry)
        out.extend(INDUSTRY_ALIASES.get(normalize_text(industry), []))
    if raw_vacancy:
        out.append(raw_vacancy)
    return out


def is_real_geo(value: str) -> bool:
    norm = normalize_text(value)
    return bool(norm) and norm not in REAL_GEO_BLACKLIST


def make_anchor(kind: str, label: str, evidences: Sequence[str]) -> Dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "evidences": [e for e in evidences if str(e or "").strip()],
    }


def build_expected_anchors(case: CdmCase) -> Dict[str, Any]:
    vacancy = case.vacancy
    raw = case.raw_vacancy
    source_title = first_nonempty_line(raw) or str(vacancy.get("title") or "").strip()
    title_aliases = [source_title]
    current_title = str(vacancy.get("title") or "").strip()
    if current_title and current_title not in title_aliases:
        title_aliases.append(current_title)

    skills = split_csv_like(vacancy.get("vacancy_stack")) + split_csv_like(vacancy.get("vacancy_skills"))
    dedup_skills: List[str] = []
    seen = set()
    for item in skills:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            dedup_skills.append(item)
    dedup_skills = dedup_skills[:5]

    anchors: List[Dict[str, Any]] = []
    if source_title:
        anchors.append(make_anchor("position", source_title, title_aliases))

    levels = extract_levels([source_title, raw])
    for level in levels[:1]:
        anchors.append(make_anchor("level", level, [level, source_title, raw]))

    exp = extract_experience(raw)
    if exp is not None:
        frm, to = exp
        label = f"опыт {frm}-{to} лет" if frm is not None and to is not None else f"опыт от {frm} лет"
        anchors.append(make_anchor("experience", label, [raw]))

    for skill in dedup_skills:
        anchors.append(make_anchor("skill", skill, [skill, raw]))

    location = str(vacancy.get("location") or "").strip()
    if is_real_geo(location):
        anchors.append(make_anchor("location", location, [location, raw]))

    industry = str(vacancy.get("company_industry") or "").strip()
    if industry:
        anchors.append(make_anchor("industry", industry, build_industry_evidence(industry, raw)))

    for lang in extract_languages(raw):
        anchors.append(make_anchor("language", lang, [lang, raw]))

    evidence = {
        "positions": title_aliases + [raw],
        "skills": dedup_skills + [raw],
        "locations": ([location] if location else []) + [raw],
        "business_spheres": build_industry_evidence(industry, raw),
        "companies": [str(vacancy.get("company_name") or "").strip(), raw],
        "keywords": [raw],
        "level": levels + [source_title, raw],
        "languages": extract_languages(raw) + [raw],
    }
    return {"source_title": source_title, "anchors": anchors, "evidence": evidence, "experience": exp}


def entity_supported(raw_text: str, evidences: Sequence[str]) -> bool:
    needle = normalize_text(raw_text)
    if not needle:
        return True
    needle_tokens = set(tokenize(raw_text))
    for evidence in evidences:
        hay = normalize_text(evidence)
        if not hay:
            continue
        if needle in hay or hay in needle:
            return True
        hay_tokens = set(tokenize(evidence))
        if needle_tokens and hay_tokens:
            overlap = len(needle_tokens & hay_tokens) / max(1, len(needle_tokens))
            if overlap >= 0.5:
                return True
            shared = needle_tokens & hay_tokens
            if shared and len(shared) >= min(len(needle_tokens), len(hay_tokens), 2):
                return True
    return False


def iter_entity_values(extractor_json: Dict[str, Any], field: str) -> List[str]:
    raw = extractor_json.get(field)
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = str(item.get("raw_text") or "").strip()
        if value:
            out.append(value)
    return out


def ranges_overlap(
    left: Optional[Tuple[Optional[int], Optional[int]]],
    right: Optional[Tuple[Optional[int], Optional[int]]],
) -> bool:
    if left is None or right is None:
        return False
    lf, lt = left
    rf, rt = right
    lf = 0 if lf is None else lf
    rf = 0 if rf is None else rf
    lt = 10**9 if lt is None else lt
    rt = 10**9 if rt is None else rt
    return max(lf, rf) <= min(lt, rt)


def detect_forbidden_flags(query: str, vacancy: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for name, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(query or ""):
            out.append(name)
    company_name = str(vacancy.get("company_name") or "").strip()
    if company_name and normalize_text(company_name) in normalize_text(query):
        out.append("company_name_mentioned")
    return out


def evaluate_semantics(
    query: str,
    extractor_json: Dict[str, Any],
    case: CdmCase,
) -> Dict[str, Any]:
    spec = build_expected_anchors(case)
    evidence = spec["evidence"]
    unsupported_entities: List[Dict[str, str]] = []

    for field in ("positions", "skills", "locations", "companies", "business_spheres", "keywords"):
        for value in iter_entity_values(extractor_json, field):
            if not entity_supported(value, evidence.get(field, []) + [case.raw_vacancy]):
                unsupported_entities.append({"field": field, "value": value})

    unsupported_levels: List[str] = []
    for level in extractor_json.get("level") or []:
        if not isinstance(level, str):
            continue
        if not entity_supported(level, evidence.get("level", [])):
            unsupported_levels.append(level)

    lang = extractor_json.get("languages") or {}
    unsupported_languages: List[str] = []
    if isinstance(lang, dict):
        expected_langs = set(x for x in evidence.get("languages", []) if x in ("english", "russian"))
        if lang.get("english") is True and "english" not in expected_langs:
            unsupported_languages.append("english")
        if lang.get("russian") is True and "russian" not in expected_langs:
            unsupported_languages.append("russian")

    extractor_exp = extractor_json.get("experience")
    normalized_extractor_exp: Optional[Tuple[Optional[int], Optional[int]]] = None
    if isinstance(extractor_exp, dict):
        frm = extractor_exp.get("from")
        to = extractor_exp.get("to")
        normalized_extractor_exp = (
            int(frm) if isinstance(frm, int) else None,
            int(to) if isinstance(to, int) else None,
        )
    unsupported_experience = bool(
        normalized_extractor_exp and not ranges_overlap(normalized_extractor_exp, spec["experience"])
    )

    anchor_hits = 0
    anchor_details: List[Dict[str, Any]] = []
    extracted_positions = iter_entity_values(extractor_json, "positions")
    extracted_skills = iter_entity_values(extractor_json, "skills")
    extracted_locations = iter_entity_values(extractor_json, "locations")
    extracted_spheres = iter_entity_values(extractor_json, "business_spheres")
    extracted_keywords = iter_entity_values(extractor_json, "keywords")
    extracted_levels = [str(x) for x in (extractor_json.get("level") or []) if isinstance(x, str)]
    enabled_languages = [key for key, value in (lang.items() if isinstance(lang, dict) else []) if value is True]

    for anchor in spec["anchors"]:
        kind = anchor["kind"]
        hit = False
        if kind == "position":
            hit = any(entity_supported(value, anchor["evidences"]) for value in extracted_positions)
        elif kind == "skill":
            hit = any(
                entity_supported(value, anchor["evidences"])
                for value in extracted_skills + extracted_keywords
            )
        elif kind == "location":
            hit = any(entity_supported(value, anchor["evidences"]) for value in extracted_locations)
        elif kind == "industry":
            hit = any(
                entity_supported(value, anchor["evidences"])
                for value in extracted_spheres + extracted_keywords
            )
        elif kind == "level":
            hit = any(entity_supported(value, anchor["evidences"]) for value in extracted_levels)
        elif kind == "language":
            hit = anchor["label"] in enabled_languages
        elif kind == "experience":
            hit = ranges_overlap(normalized_extractor_exp, spec["experience"])
        if hit:
            anchor_hits += 1
        anchor_details.append({"kind": kind, "label": anchor["label"], "hit": hit})

    anchor_total = len(spec["anchors"])
    coverage_pct = round((anchor_hits / anchor_total) * 100.0, 2) if anchor_total else 0.0

    return {
        "expected_anchor_total": anchor_total,
        "expected_anchor_hits": anchor_hits,
        "expected_anchor_coverage_pct": coverage_pct,
        "anchor_details": anchor_details,
        "unsupported_entities": unsupported_entities,
        "unsupported_levels": unsupported_levels,
        "unsupported_languages": unsupported_languages,
        "unsupported_experience": unsupported_experience,
        "forbidden_flags": detect_forbidden_flags(query, case.vacancy),
    }


def build_query_checks(query: str) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    stripped = (query or "").strip()
    if not stripped:
        errors.append("empty_query")
    if "\n" in stripped or "\r" in stripped:
        errors.append("query_not_single_line")
    if "{" in stripped or "}" in stripped or "[" in stripped or "]" in stripped:
        errors.append("json_like_output")
    if '"' in stripped:
        errors.append("contains_quotes")
    word_count = len([x for x in stripped.split() if x])
    if word_count > 40:
        warnings.append("query_too_long")
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "word_count": word_count,
        "char_count": len(stripped),
    }


def load_cdm_cases(cdm_dir: Path, cdm_count: Optional[int]) -> List[CdmCase]:
    if not cdm_dir.exists():
        raise FileNotFoundError(f"CDM dir not found: {cdm_dir}")
    paths = sorted(cdm_dir.glob("cdm_*.json"))
    if cdm_count is not None and cdm_count > 0:
        paths = paths[:cdm_count]
    if not paths:
        raise FileNotFoundError(f"No cdm_*.json found in: {cdm_dir}")

    out: List[CdmCase] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        vacancy = data.get("vacancy") or {}
        raw_vacancy = str(vacancy.get("raw_vacancy") or "").strip()
        if not raw_vacancy:
            raise ValueError(f"{path.name} has empty vacancy.raw_vacancy")
        out.append(CdmCase(name=path.stem, path=path, raw_vacancy=raw_vacancy, vacancy=vacancy))
    return out


def dedupe_keep_order(values: Sequence[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def format_range_pair(pair: Optional[Sequence[Any]]) -> Optional[str]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    frm = str(pair[0] or "").strip()
    to = str(pair[1] or "").strip()
    if frm and to:
        return f"{frm}-{to}"
    if frm:
        return f"от {frm}"
    if to:
        return f"до {to}"
    return None


def range_obj_to_pair(obj: Any) -> Optional[List[str]]:
    if not isinstance(obj, dict):
        return None
    frm = obj.get("from")
    to = obj.get("to")
    if frm is None and to is None:
        return None
    return [str(frm) if frm is not None else "", str(to) if to is not None else ""]


def summarize_entity_list(items: Any) -> Dict[str, List[str]]:
    if not isinstance(items, list):
        return {}
    buckets: Dict[str, List[str]] = {"required": [], "optional": [], "excluded": []}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_text = str(item.get("raw_text") or "").strip()
        if not raw_text:
            continue
        op = str(item.get("operator") or "AND").upper()
        bucket = "required"
        if op == "OR":
            bucket = "optional"
        elif op == "NOT":
            bucket = "excluded"
        buckets[bucket].append(raw_text)
    return {key: dedupe_keep_order(values) for key, values in buckets.items() if values}


def summarize_grouped_payload(items: Any) -> Dict[str, List[str]]:
    if not isinstance(items, list):
        return {}
    buckets: Dict[str, List[str]] = {"required": [], "optional": [], "excluded": []}
    for item in items:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list):
            continue
        group = str(item[0] or "").strip().lower()
        values = dedupe_keep_order(item[1])
        if not values:
            continue
        bucket = "required"
        if group == "or":
            bucket = "optional"
        elif group == "not":
            bucket = "excluded"
        buckets[bucket].extend(values)
    return {key: dedupe_keep_order(values) for key, values in buckets.items() if values}


def summarize_languages(lang_obj: Any) -> Dict[str, List[str]]:
    if not isinstance(lang_obj, dict):
        return {}
    required = [key for key, value in lang_obj.items() if value is True]
    excluded = [key for key, value in lang_obj.items() if value is False]
    out: Dict[str, List[str]] = {}
    if required:
        out["required"] = dedupe_keep_order(required)
    if excluded:
        out["excluded"] = dedupe_keep_order(excluded)
    return out


def summarize_extractor_output(extractor_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(extractor_json, dict):
        return {}
    out: Dict[str, Any] = {}
    positions = summarize_entity_list(extractor_json.get("positions"))
    skills = summarize_entity_list(extractor_json.get("skills"))
    keywords = summarize_entity_list(extractor_json.get("keywords"))
    locations = summarize_entity_list(extractor_json.get("locations"))
    companies = summarize_entity_list(extractor_json.get("companies"))
    business_spheres = summarize_entity_list(extractor_json.get("business_spheres"))
    languages = summarize_languages(extractor_json.get("languages"))
    experience = format_range_pair(range_obj_to_pair(extractor_json.get("experience")))

    if positions:
        out["positions"] = positions
    if skills:
        out["skills"] = skills
    if keywords:
        out["keywords"] = keywords
    if business_spheres:
        out["business_spheres"] = business_spheres
    if locations:
        out["locations"] = locations
    if companies:
        out["companies"] = companies
    levels = [str(x) for x in (extractor_json.get("level") or []) if isinstance(x, str)]
    if levels:
        out["levels"] = dedupe_keep_order(levels)
    if experience:
        out["experience"] = experience
    if languages:
        out["languages"] = languages
    return out


def summarize_search_payload(step3_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(step3_payload, dict):
        return {}
    out: Dict[str, Any] = {}
    positions = summarize_grouped_payload(step3_payload.get("positions"))
    skills = summarize_grouped_payload(step3_payload.get("skills"))
    keywords = summarize_grouped_payload(step3_payload.get("keys"))
    locations = summarize_grouped_payload(step3_payload.get("geos"))
    companies = summarize_grouped_payload(step3_payload.get("firms"))
    experience = format_range_pair(step3_payload.get("experience"))
    levels = step3_payload.get("seniorityLevels")

    if positions:
        out["positions"] = positions
    if skills:
        out["skills"] = skills
    if keywords:
        out["keywords"] = keywords
    if locations:
        out["locations"] = locations
    if companies:
        out["companies"] = companies
    if isinstance(levels, list) and levels:
        out["levels"] = dedupe_keep_order(levels)
    if experience:
        out["experience"] = experience

    active_filters: List[str] = []
    if step3_payload.get("onlyEnglish") is True:
        active_filters.append("only_english")
    if step3_payload.get("onlyRussian") is True:
        active_filters.append("only_russian")
    if step3_payload.get("onlyWithContacts") is True:
        active_filters.append("only_with_contacts")
    if step3_payload.get("onlyWithHigherEducation") is True:
        active_filters.append("only_with_higher_education")
    if step3_payload.get("currentPositionTitle") is True:
        active_filters.append("current_position_title")
    if active_filters:
        out["filters"] = active_filters
    return out


def build_case_result_view(result: CaseResult) -> Dict[str, Any]:
    backend = result.backend or {}
    builder_errors = result.builder.get("errors") or []
    extractor_errors = result.extractor.get("errors") or []

    if result.status == "failed_builder":
        out = {
            "passed": False,
            "stage": "builder",
            "code": builder_errors[0] if builder_errors else "builder_failed",
        }
        return out
    if result.status == "failed_extractor":
        out = {
            "passed": False,
            "stage": "extractor",
            "code": extractor_errors[0] if extractor_errors else "extractor_failed",
        }
        return out
    if result.status == "failed_backend":
        out = {
            "passed": False,
            "stage": "backend",
            "code": str(backend.get("kind") or "backend_failed"),
        }
        if backend.get("status") is not None:
            out["http_status"] = backend.get("status")
        if backend.get("error_message"):
            out["message"] = backend.get("error_message")
        return out
    if result.status == "zero_results":
        out = {"passed": False, "stage": "backend", "code": "zero_results", "count": 0}
        if backend.get("status") is not None:
            out["http_status"] = backend.get("status")
        return out
    if result.status == "failed_quality":
        out = {"passed": False, "stage": "quality", "code": "quality_rules_failed"}
        if backend.get("count") is not None:
            out["count"] = backend.get("count")
        if backend.get("status") is not None:
            out["http_status"] = backend.get("status")
        return out
    out = {"passed": True, "stage": "done", "code": "passed"}
    if backend.get("count") is not None:
        out["count"] = backend.get("count")
    return out


def build_case_issues_view(result: CaseResult, min_anchor_coverage: float) -> List[str]:
    semantic = result.semantic or {}
    out: List[str] = []
    out.extend(str(x) for x in (result.builder.get("errors") or []))
    warnings = result.builder.get("warnings") or []
    out.extend(str(x) for x in warnings)
    coverage = semantic.get("expected_anchor_coverage_pct")
    forbidden_flags = semantic.get("forbidden_flags") or []
    out.extend(str(x) for x in forbidden_flags)
    out.extend(str(x) for x in (result.extractor.get("errors") or []))
    unsupported_count = 0
    unsupported_count += len(semantic.get("unsupported_entities") or [])
    unsupported_count += len(semantic.get("unsupported_levels") or [])
    unsupported_count += len(semantic.get("unsupported_languages") or [])
    unsupported_count += 1 if semantic.get("unsupported_experience") else 0
    if unsupported_count:
        out.append(f"unsupported:{unsupported_count}")
    if coverage is not None and float(coverage) < float(min_anchor_coverage):
        out.append(f"low_anchor_coverage:{coverage}")
    return dedupe_keep_order(out)


def build_case_debug_view(result: CaseResult) -> Dict[str, Any]:
    semantic = result.semantic or {}
    backend = result.backend or {}
    out: Dict[str, Any] = {}

    builder_debug: Dict[str, Any] = {}
    if result.builder.get("errors"):
        builder_debug["errors"] = result.builder.get("errors")
    if result.builder.get("warnings"):
        builder_debug["warnings"] = result.builder.get("warnings")
    if builder_debug:
        out["builder"] = builder_debug

    extractor_debug: Dict[str, Any] = {}
    if result.extractor.get("errors"):
        extractor_debug["errors"] = result.extractor.get("errors")
    if result.extractor.get("warnings"):
        extractor_debug["warnings"] = result.extractor.get("warnings")
    if extractor_debug:
        out["extractor"] = extractor_debug

    semantic_debug: Dict[str, Any] = {}
    missing_anchors = [x["label"] for x in (semantic.get("anchor_details") or []) if not x.get("hit")]
    if missing_anchors:
        semantic_debug["missing_anchors"] = missing_anchors
    if semantic.get("forbidden_flags"):
        semantic_debug["forbidden_flags"] = semantic.get("forbidden_flags")
    if semantic.get("unsupported_entities"):
        semantic_debug["unsupported_entities"] = semantic.get("unsupported_entities")
    if semantic.get("unsupported_levels"):
        semantic_debug["unsupported_levels"] = semantic.get("unsupported_levels")
    if semantic.get("unsupported_languages"):
        semantic_debug["unsupported_languages"] = semantic.get("unsupported_languages")
    if semantic.get("unsupported_experience"):
        semantic_debug["unsupported_experience"] = True
    if semantic_debug:
        out["semantic"] = semantic_debug

    backend_debug: Dict[str, Any] = {}
    if backend.get("kind") not in (None, "success", "skipped"):
        backend_debug["kind"] = backend.get("kind")
    if backend.get("status") not in (None, 200):
        backend_debug["http_status"] = backend.get("status")
    if backend.get("error_message"):
        backend_debug["message"] = backend.get("error_message")
    if backend_debug:
        out["backend"] = backend_debug

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run raw_vacancy -> one-line query -> extractor -> searchBool evaluation."
    )
    ap.add_argument("--cdm-dir", type=str, default="tests/fixtures/cdm")
    ap.add_argument("--cdm-count", type=int, default=0)
    ap.add_argument("--steps", type=str, default="1,2,3")
    ap.add_argument("--cfg", type=str, default="tests/tools/model.yaml")
    ap.add_argument("--builder-prompt-id", type=str, default="")
    ap.add_argument("--builder-prompt-version", type=str, default="")
    ap.add_argument("--extractor-prompt-id", type=str, default="")
    ap.add_argument("--extractor-prompt-version", type=str, default="")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--base-url", type=str, default=os.getenv("AI_SEARCH_BASE_URL", "").strip())
    ap.add_argument("--step3-path", type=str, default="/site/searchBool")
    ap.add_argument("--token", type=str, default=os.getenv("AI_SEARCH_AUTH_TOKEN", "").strip())
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--step3-retries", type=int, default=2)
    ap.add_argument("--token-in-body", dest="token_in_body", action="store_true", default=True)
    ap.add_argument("--token-in-header", dest="token_in_body", action="store_false")
    ap.add_argument("--only-russian", action="store_true", default=False)
    ap.add_argument("--only-english", action="store_true", default=False)
    ap.add_argument("--only-with-contacts", action="store_true", default=True)
    ap.add_argument("--only-with-higher-education", action="store_true", default=False)
    ap.add_argument("--current-position-title", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", default=False)
    ap.add_argument("--highlight", action="store_true", default=True)
    ap.add_argument("--min-anchor-coverage", type=float, default=40.0)
    ap.add_argument("--report-dir", type=str, default="tests/reports/one_line_search_query_builder")
    ap.add_argument("--report-mode", type=str, choices=["compact", "full"], default="compact")
    ap.add_argument("--report-json-indent", type=int, default=2)
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    if 3 in steps and 2 not in steps:
        raise SystemExit("Step 3 requires step 2.")

    cfg_path = Path(args.cfg)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    builder_prompt = resolve_prompt_from_cfg(
        cfg=cfg,
        section_name="one_line_search_query_builder",
        env_prefix="ONE_LINE_SEARCH_QUERY_BUILDER",
        default_model="gpt-4.1",
        override_prompt_id=args.builder_prompt_id,
        override_prompt_version=args.builder_prompt_version,
        override_model=args.model,
    )
    extractor_prompt = resolve_prompt_from_cfg(
        cfg=cfg,
        section_name="extractor_agent",
        env_prefix="EXTRACTOR_AGENT",
        default_model=builder_prompt.model,
        override_prompt_id=args.extractor_prompt_id,
        override_prompt_version=args.extractor_prompt_version,
        override_model=args.model,
    )

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is required.")
    if not builder_prompt.prompt_id:
        raise SystemExit("Missing one_line_search_query_builder.prompt_id.")
    if 2 in steps and not extractor_prompt.prompt_id:
        raise SystemExit("Missing extractor_agent.prompt_id.")
    if 3 in steps and (not args.base_url or not args.token):
        raise SystemExit("Step 3 requires both --base-url and --token (or env defaults).")

    backend_cfg = BackendCfg(
        base_url=args.base_url,
        step3_path=args.step3_path,
        token_in_body=bool(args.token_in_body),
        timeout_s=int(args.timeout_s),
        retries=int(args.step3_retries),
        sanitize_office_geo=True,
        require_search_terms=True,
        require_count=True,
    )
    base_payload = {
        "onlyRussian": bool(args.only_russian),
        "onlyEnglish": bool(args.only_english),
        "onlyWithContacts": bool(args.only_with_contacts),
        "onlyWithHigherEducation": bool(args.only_with_higher_education),
        "currentPositionTitle": bool(args.current_position_title),
        "limit": int(args.limit),
        "offset": int(args.offset),
        "shuffle": bool(args.shuffle),
        "highlight": bool(args.highlight),
    }
    cases = load_cdm_cases(Path(args.cdm_dir), args.cdm_count if args.cdm_count > 0 else None)
    run_id = make_run_id()
    print(
        f"[init] run_id={run_id} cases={len(cases)} steps={steps} "
        f"builder_prompt_id={builder_prompt.prompt_id} extractor_prompt_id={extractor_prompt.prompt_id}"
    )

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    results: List[CaseResult] = []
    builder_failures = 0
    extractor_failures = 0
    backend_failures = 0
    insufficient_search_terms = 0
    zero_results = 0
    passed_total = 0
    semantic_failures = 0
    coverage_values: List[float] = []

    for idx, case in enumerate(cases, start=1):
        print(f"[run] case {idx}/{len(cases)} file={case.path.name}")
        query = ""
        builder_info: Dict[str, Any] = {}
        extractor_info: Dict[str, Any] = {"ok": True}
        semantic_info: Dict[str, Any] = {}
        backend_info: Dict[str, Any] = {"kind": "skipped"}
        extractor_json: Optional[Dict[str, Any]] = None
        step3_payload: Optional[Dict[str, Any]] = None

        query_text, usage, err = call_openai_step1(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            prompt_cfg=builder_prompt,
            user_input=case.raw_vacancy,
            timeout_s=int(args.timeout_s),
        )
        usage_total["input_tokens"] += usage["input_tokens"]
        usage_total["output_tokens"] += usage["output_tokens"]
        usage_total["total_tokens"] += usage["total_tokens"]

        if err or not query_text:
            builder_failures += 1
            results.append(
                CaseResult(
                    name=case.name,
                    cdm_file=case.path.name,
                    vacancy_title=str(case.vacancy.get("title") or ""),
                    generated_query="",
                    status="failed_builder",
                    passed=False,
                    builder={"ok": False, "errors": [err or "builder_empty_output"]},
                    extractor=extractor_info,
                    semantic=semantic_info,
                    backend=backend_info,
                )
            )
            continue

        query = query_text.strip()
        builder_info = build_query_checks(query)
        builder_info["query"] = query
        if not builder_info["ok"]:
            builder_failures += 1
            results.append(
                CaseResult(
                    name=case.name,
                    cdm_file=case.path.name,
                    vacancy_title=str(case.vacancy.get("title") or ""),
                    generated_query=query,
                    status="failed_builder",
                    passed=False,
                    builder=builder_info,
                    extractor=extractor_info,
                    semantic=semantic_info,
                    backend=backend_info,
                )
            )
            continue

        if 2 in steps:
            extractor_text, usage2, err2 = call_openai_step1(
                api_key=os.getenv("OPENAI_API_KEY", "").strip(),
                prompt_cfg=extractor_prompt,
                user_input=query,
                timeout_s=int(args.timeout_s),
            )
            usage_total["input_tokens"] += usage2["input_tokens"]
            usage_total["output_tokens"] += usage2["output_tokens"]
            usage_total["total_tokens"] += usage2["total_tokens"]

            if err2 or not extractor_text:
                extractor_failures += 1
                results.append(
                    CaseResult(
                        name=case.name,
                        cdm_file=case.path.name,
                        vacancy_title=str(case.vacancy.get("title") or ""),
                        generated_query=query,
                        status="failed_extractor",
                        passed=False,
                        builder=builder_info,
                        extractor={"ok": False, "errors": [err2 or "extractor_empty_output"]},
                        semantic=semantic_info,
                        backend=backend_info,
                    )
                )
                continue

            try:
                parsed = json.loads(extractor_text)
            except Exception as exc:
                extractor_failures += 1
                results.append(
                    CaseResult(
                        name=case.name,
                        cdm_file=case.path.name,
                        vacancy_title=str(case.vacancy.get("title") or ""),
                        generated_query=query,
                        status="failed_extractor",
                        passed=False,
                        builder=builder_info,
                        extractor={"ok": False, "errors": [f"extractor_invalid_json:{exc}"]},
                        semantic=semantic_info,
                        backend=backend_info,
                    )
                )
                continue

            if not isinstance(parsed, dict):
                extractor_failures += 1
                results.append(
                    CaseResult(
                        name=case.name,
                        cdm_file=case.path.name,
                        vacancy_title=str(case.vacancy.get("title") or ""),
                        generated_query=query,
                        status="failed_extractor",
                        passed=False,
                        builder=builder_info,
                        extractor={"ok": False, "errors": ["extractor_json_must_be_object"]},
                        semantic=semantic_info,
                        backend=backend_info,
                    )
                )
                continue

            extractor_json = parsed
            contract_ok, contract_errors, contract_warnings = validate_step1_contract(extractor_json, query)
            has_anchor = bool(
                iter_entity_values(extractor_json, "positions")
                or iter_entity_values(extractor_json, "skills")
                or iter_entity_values(extractor_json, "keywords")
            )
            extractor_info = {
                "ok": contract_ok and has_anchor,
                "contract_ok": contract_ok,
                "has_anchor": has_anchor,
            }
            if contract_errors:
                extractor_info["errors"] = contract_errors
            if contract_warnings:
                extractor_info["warnings"] = contract_warnings

            if not extractor_info["ok"]:
                extractor_failures += 1
                results.append(
                    CaseResult(
                        name=case.name,
                        cdm_file=case.path.name,
                        vacancy_title=str(case.vacancy.get("title") or ""),
                        generated_query=query,
                        status="failed_extractor",
                        passed=False,
                        builder=builder_info,
                        extractor=extractor_info,
                        semantic=semantic_info,
                        backend=backend_info,
                        extractor_json=extractor_json,
                    )
                )
                continue

            semantic_info = evaluate_semantics(query, extractor_json, case)
            coverage_values.append(float(semantic_info["expected_anchor_coverage_pct"]))
            if (
                semantic_info["unsupported_entities"]
                or semantic_info["unsupported_levels"]
                or semantic_info["unsupported_languages"]
                or semantic_info["unsupported_experience"]
                or semantic_info["forbidden_flags"]
                or float(semantic_info.get("expected_anchor_coverage_pct") or 0.0) < float(args.min_anchor_coverage)
            ):
                semantic_failures += 1

        if 3 in steps:
            if extractor_json is None:
                extractor_failures += 1
                results.append(
                    CaseResult(
                        name=case.name,
                        cdm_file=case.path.name,
                        vacancy_title=str(case.vacancy.get("title") or ""),
                        generated_query=query,
                        status="failed_extractor",
                        passed=False,
                        builder=builder_info,
                        extractor={"ok": False, "errors": ["missing_extractor_json_for_step3"]},
                        semantic=semantic_info,
                        backend=backend_info,
                    )
                )
                continue

            step3_payload = build_step3_payload(
                extractor_json=extractor_json,
                user_phrase=query,
                base_payload=base_payload,
                sanitize_office_geo=True,
            )
            kind, status_code, attempts, count, err_msg, response_json = call_backend_search_bool(
                backend=backend_cfg,
                token=args.token,
                payload=step3_payload,
            )
            backend_info = {"kind": kind, "status": status_code, "attempts": attempts}
            if err_msg:
                backend_info["error_message"] = err_msg
            if response_json is not None:
                backend_info["response"] = response_json
            if count is not None:
                backend_info["count"] = count

            if kind == "insufficient_search_terms":
                insufficient_search_terms += 1
            elif kind != "success":
                backend_failures += 1
            elif count == 0:
                zero_results += 1

        backend_success = backend_info.get("kind") == "success"
        count_positive = isinstance(backend_info.get("count"), int) and backend_info["count"] > 0
        retrieval_ok = builder_info["ok"] and extractor_info.get("ok", True) and ((3 not in steps) or (backend_success and count_positive))
        semantic_ok = True
        if 2 in steps:
            semantic_ok = (
                not semantic_info.get("unsupported_entities")
                and not semantic_info.get("unsupported_levels")
                and not semantic_info.get("unsupported_languages")
                and not semantic_info.get("unsupported_experience")
                and not semantic_info.get("forbidden_flags")
                and float(semantic_info.get("expected_anchor_coverage_pct") or 0.0) >= float(args.min_anchor_coverage)
            )
        passed = retrieval_ok and semantic_ok

        if passed:
            passed_total += 1

        status = (
            "passed"
            if passed
            else "failed_quality"
            if retrieval_ok
            else "zero_results"
            if backend_success
            else "failed_backend"
        )
        results.append(
            CaseResult(
                name=case.name,
                cdm_file=case.path.name,
                vacancy_title=str(case.vacancy.get("title") or ""),
                generated_query=query,
                status=status,
                passed=passed,
                builder=builder_info,
                extractor=extractor_info,
                semantic=semantic_info,
                backend=backend_info,
                extractor_json=extractor_json,
                step3_payload=step3_payload,
            )
        )

    total = len(results)
    avg_anchor_coverage_pct = (
        round(sum(coverage_values) / len(coverage_values), 2) if coverage_values else 0.0
    )
    status_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    report: Dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "report_mode": args.report_mode,
            "steps": steps,
            "cases": len(cases),
            "prompts": {
                "builder": {
                    "prompt_id": builder_prompt.prompt_id,
                    "prompt_version": builder_prompt.prompt_version,
                },
                "extractor": {
                    "prompt_id": extractor_prompt.prompt_id,
                    "prompt_version": extractor_prompt.prompt_version,
                },
            },
        },
        "summary": {
            "total": total,
            "passed": passed_total,
            "failed": total - passed_total,
            "by_status": status_counts,
            "avg_anchor_coverage_pct": avg_anchor_coverage_pct,
        },
        "cases": [],
    }

    if args.report_mode == "full":
        report["debug"] = {
            "config": {
                "cdm_dir": str(Path(args.cdm_dir)),
                "min_anchor_coverage": float(args.min_anchor_coverage),
                "builder_model": builder_prompt.model,
                "extractor_model": extractor_prompt.model,
                "backend": {
                    "base_url": backend_cfg.base_url,
                    "timeout_s": backend_cfg.timeout_s,
                    "retries": backend_cfg.retries,
                },
                "search_flags": [
                    name
                    for name, enabled in (
                        ("only_russian", base_payload.get("onlyRussian")),
                        ("only_english", base_payload.get("onlyEnglish")),
                        ("only_with_contacts", base_payload.get("onlyWithContacts")),
                        ("only_with_higher_education", base_payload.get("onlyWithHigherEducation")),
                        ("current_position_title", base_payload.get("currentPositionTitle")),
                    )
                    if enabled
                ],
            },
            "token_usage_total": usage_total,
        }

    for result in results:
        issues = build_case_issues_view(result, float(args.min_anchor_coverage))
        case_item: Dict[str, Any] = {
            "name": result.name,
            "title": result.vacancy_title,
            "query": result.generated_query,
            "result": build_case_result_view(result),
            "search_summary": summarize_search_payload(result.step3_payload)
            or summarize_extractor_output(result.extractor_json),
        }
        if issues:
            case_item["issues"] = issues
        coverage = result.semantic.get("expected_anchor_coverage_pct") if isinstance(result.semantic, dict) else None
        if coverage is not None and (not result.passed or float(coverage) < 100.0):
            case_item["anchor_coverage_pct"] = coverage
        if args.report_mode == "full" and not result.passed:
            debug_view = build_case_debug_view(result)
            if debug_view:
                case_item["debug"] = debug_view
        report["cases"].append(case_item)

    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)
    out_path = report_dir / f"one_line_search_query_builder_report_{run_id}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=int(args.report_json_indent)),
        encoding="utf-8",
    )

    print(
        f"[summary] total={total} passed={passed_total} "
        f"builder_failures={builder_failures} extractor_failures={extractor_failures} "
        f"backend_failures={backend_failures} insufficient={insufficient_search_terms} "
        f"zero_results={zero_results} semantic_failures={semantic_failures} "
        f"avg_anchor_coverage={avg_anchor_coverage_pct:.2f}"
    )
    print(f"[done] report saved: {out_path}")

    return 0 if passed_total == total else 2


if __name__ == "__main__":
    raise SystemExit(main())

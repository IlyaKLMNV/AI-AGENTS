#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extractor_agent_runner.py

Pipeline:
  step1: LLM extracts structured JSON (extractor_json)
  step2: build backend /site/searchBool payload (no "adding meaning", only mapping)
  step3: call backend and collect count

Key mappings:
  step1.companies   -> step3.firms
  step1.keywords    -> step3.keys
  step1.level       -> step3.seniorityLevels
  step1.locations   -> step3.geos
  step1.languages   -> step3.onlyRussian / step3.onlyEnglish (both true allowed)
  step1.experience  -> step3.experience ["from","to"] (strings; null -> "")
  step1.age         -> step3.ages ["from","to"] (strings; null -> "")
  step1.management_experience -> step3.managementExperience ["from","to"] (strings; null -> "")
  step1.higher_education -> step3.onlyWithHigherEducation

Backend expects groups for boolean fields:
  field: [ ["all"|"or"|"not", [values...]], ... ]

Important:
  - Anything that requires dictionaries / ID mapping (firmCategories IDs, contacts codes like "lnlink",
    additionalSkills split, etc.) is NOT implemented here.
  - Suite cases are built-in sanity/regression prompts that ALWAYS include at least one anchor:
    positions OR skills OR keywords.

Defaults (so you don't have to pass them every run):
  - --base-url defaults to env AI_SEARCH_BASE_URL
  - --token defaults to env AI_SEARCH_AUTH_TOKEN
  - --step3-path defaults to /site/searchBool
  - --report-mode defaults to compact
  - --report-json-indent defaults to 2

Env:
  OPENAI_API_KEY required for step1
  AI_SEARCH_BASE_URL, AI_SEARCH_AUTH_TOKEN recommended for step3
  Optional: OPENAI_BASE_URL (defaults to https://api.openai.com/v1)

This runner is intentionally strict about the step1 contract to catch prompt drift.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ----------------------------
# Built-in suite (regression) cases
# MUST include at least one anchor: positions OR skills OR keywords
# ----------------------------

SUITE_CASES: List[Dict[str, str]] = [
    # basic anchors
    {"name": "suite_0001", "input": "Python разработчик"},
    {"name": "suite_0002", "input": "React developer"},
    {"name": "suite_0003", "input": "ключевые слова: Иван Петров"},
    # operators and separators
    {"name": "suite_0004", "input": "React + TypeScript + Next.js"},
    {"name": "suite_0005", "input": "Django или Flask"},
    {"name": "suite_0006", "input": "Python developer, но не Django"},
    {"name": "suite_0007", "input": "SRE / DevOps"},
    # plus tokens should not break
    {"name": "suite_0008", "input": "C++ developer"},
    {"name": "suite_0009", "input": "опыт с g++ и clang"},
    # languages nuance (still anchored by a skill)
    {"name": "suite_0010", "input": "React, русский обязателен, английский желателен"},
    {"name": "suite_0011", "input": "React, английский B2, русский не нужен"},
    # geo sanitize (anchored by position)
    {"name": "suite_0012", "input": "Backend разработчик Москва офис"},
    # experience formats (anchored by position)
    {"name": "suite_0013", "input": "Python разработчик опыт от 5 лет"},
    {"name": "suite_0014", "input": "DevOps инженер опыт 3-5 лет"},
    # keywords (email/phone) - explicit keyword anchor
    {"name": "suite_0015", "input": "контакт: ivan.petrov@gmail.com, +7 999 123-45-67"},
    {"name": "suite_0016", "input": "Платов Анатолий backend"},
    # business sphere mention (still anchored by position; sphere will be ignored without dictionary)
    {"name": "suite_0017", "input": "DevOps из финтеха"},
    # company mention (anchored by position; firms-only may be insufficient in backend)
    {"name": "suite_0018", "input": "SRE, последнее место работы: Яндекс"},
]


# ----------------------------
# Helpers: time, json, io
# ----------------------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def make_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()

def safe_json_loads(s: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------
# Config loading
# ----------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, but YAML config was requested.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data

def load_yaml_any(path: Path) -> Any:
    if not path.exists():
        return None
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, but YAML config was requested.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_get(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------
# Models
# ----------------------------

@dataclass
class PromptCfg:
    prompt_id: Optional[str]
    prompt_version: Optional[str]
    model: str

@dataclass
class BackendCfg:
    base_url: str
    step3_path: str
    token_in_body: bool
    timeout_s: int
    retries: int
    sanitize_office_geo: bool
    require_search_terms: bool
    require_count: bool

@dataclass
class MixCfg:
    ratios: Dict[str, int]
    seed: int
    total_limit: Optional[int]

@dataclass
class RunnerCfg:
    cases_dir: str
    counts: Dict[str, int]
    mix: MixCfg
    steps: List[int]
    prompt: PromptCfg
    backend: BackendCfg


# ----------------------------
# Case loading and mixing
# ----------------------------

@dataclass
class Case:
    name: str
    input: str
    source: str  # real | suite | syn

def guess_source(name: str) -> str:
    if name.startswith("suite_"):
        return "suite"
    if name.startswith("syn_"):
        return "syn"
    return "real"

def _extract_input_from_mapping(obj: Dict[str, Any]) -> Optional[str]:
    raw_input = (
        obj.get("input")
        or obj.get("query")
        or obj.get("text")
        or obj.get("user_phrase")
    )
    if isinstance(raw_input, str):
        raw_input = raw_input.strip()
    return raw_input if isinstance(raw_input, str) and raw_input else None

def _build_case_from_mapping(obj: Dict[str, Any], stem: str, idx: Optional[int] = None) -> Optional[Case]:
    raw_input = _extract_input_from_mapping(obj)
    if not raw_input:
        return None

    name_raw = obj.get("name")
    if isinstance(name_raw, str) and name_raw.strip():
        name = name_raw.strip()
    elif idx is not None:
        name = f"{stem}_{idx:04d}"
    else:
        name = stem

    source = guess_source(name)
    return Case(name=name, input=raw_input, source=source)

def _cases_from_data(data: Any, stem: str) -> List[Case]:
    out: List[Case] = []

    if isinstance(data, str):
        s = data.strip()
        if s:
            out.append(Case(name=stem, input=s, source=guess_source(stem)))
        return out

    if isinstance(data, dict):
        nested = data.get("cases")
        if isinstance(nested, list):
            data = nested
        else:
            c = _build_case_from_mapping(data, stem=stem)
            if c is not None:
                out.append(c)
            return out

    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, str):
                s = item.strip()
                if s:
                    name = f"{stem}_{i:04d}"
                    out.append(Case(name=name, input=s, source=guess_source(name)))
                continue
            if isinstance(item, dict):
                c = _build_case_from_mapping(item, stem=stem, idx=i)
                if c is not None:
                    out.append(c)
        return out

    return out

def load_cases_from_dir(cases_dir: Path) -> List[Case]:
    cases: List[Case] = []
    if not cases_dir.exists():
        raise FileNotFoundError(f"cases_dir not found: {cases_dir}")

    files = sorted([p for p in cases_dir.rglob("*") if p.is_file() and p.name[0] != "."])
    for p in files:
        stem = p.stem
        suffix = p.suffix.lower()

        if suffix in (".json",):
            obj, err = safe_json_loads(read_text_file(p))
            if err:
                data: Any = read_text_file(p)
            else:
                data = obj
            cases.extend(_cases_from_data(data, stem))
        elif suffix in (".yml", ".yaml"):
            data = load_yaml_any(p)
            cases.extend(_cases_from_data(data, stem))
        else:
            raw_input = read_text_file(p)
            if not raw_input:
                continue
            name = stem
            cases.append(Case(name=name, input=str(raw_input), source=guess_source(name)))

    uniq: Dict[str, Case] = {}
    for c in cases:
        uniq[c.name] = c
    return list(uniq.values())

def build_suite_cases(limit: int, seed: int) -> List[Case]:
    # deterministic shuffle so suite doesn't become "first N always"
    rnd = random.Random(seed)
    items = [dict(x) for x in SUITE_CASES]
    rnd.shuffle(items)
    if limit > 0:
        items = items[:limit]
    out: List[Case] = []
    for it in items:
        name = it["name"]
        inp = it["input"]
        out.append(Case(name=name, input=inp, source="suite"))
    return out

def build_synthetic_cases(base_cases: List[Case], n: int, seed: int) -> List[Case]:
    rnd = random.Random(seed)
    if not base_cases or n <= 0:
        return []

    out: List[Case] = []
    for i in range(1, n + 1):
        c = rnd.choice(base_cases)
        s = c.input.strip()

        tokens = s.split()
        if len(tokens) > 3:
            drop_p = rnd.uniform(0.15, 0.35)
            kept = [t for t in tokens if rnd.random() > drop_p]
            if len(kept) >= 2:
                tokens = kept

        if len(tokens) > 6 and rnd.random() < 0.35:
            cut = rnd.randint(2, max(3, len(tokens) // 2))
            tokens = tokens[:cut]

        if tokens and rnd.random() < 0.25:
            j = rnd.randrange(len(tokens))
            t = tokens[j]
            if len(t) >= 5:
                k = rnd.randint(1, len(t) - 2)
                t2 = t[:k] + t[k+1] + t[k] + t[k+2:]
                tokens[j] = t2

        syn_text = " ".join(tokens).strip()
        out.append(Case(name=f"syn_{i:04d}", input=syn_text, source="syn"))

    return out

def parse_mix_ratios(s: str) -> Dict[str, int]:
    ratios: Dict[str, int] = {"real": 1, "suite": 0, "syn": 0}
    s = (s or "").strip()
    if not s:
        return ratios
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in ("real", "suite", "syn"):
            continue
        try:
            ratios[k] = max(0, int(v))
        except Exception:
            pass
    return ratios

def sample_cases(cases: List[Case], n: int, rnd: random.Random) -> List[Case]:
    if n <= 0 or not cases:
        return []
    if n <= len(cases):
        return rnd.sample(cases, n)
    out: List[Case] = []
    i = 0
    while len(out) < n:
        out.append(cases[i % len(cases)])
        i += 1
    rnd.shuffle(out)
    return out

def mix_cases(
    real_cases: List[Case],
    suite_cases: List[Case],
    syn_cases: List[Case],
    per_unit_count: int,
    ratios: Dict[str, int],
    seed: int,
    total_limit: Optional[int],
) -> Tuple[List[Case], Dict[str, int]]:
    rnd = random.Random(seed)

    want_real = per_unit_count * int(ratios.get("real", 0))
    want_suite = per_unit_count * int(ratios.get("suite", 0))
    want_syn = per_unit_count * int(ratios.get("syn", 0))

    picked: List[Case] = []
    picked += sample_cases(real_cases, want_real, rnd)
    picked += sample_cases(suite_cases, want_suite, rnd)
    picked += sample_cases(syn_cases, want_syn, rnd)
    rnd.shuffle(picked)

    if total_limit is not None and total_limit > 0:
        picked = picked[:total_limit]

    factual = {"real": 0, "suite": 0, "syn": 0}
    for c in picked:
        factual[c.source] = factual.get(c.source, 0) + 1

    counts = {
        "real": factual.get("real", 0),
        "suite": factual.get("suite", 0),
        "syn": factual.get("syn", 0),
        "total": len(picked),
    }
    return picked, counts


# ----------------------------
# Step1 contract validation
# ----------------------------

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

OFFICE_GEO_TRASH = {
    "офис", "гибрид", "гибридный", "удаленно", "удалённо", "удаленка", "удалёнка",
    "remote", "hybrid", "onsite", "on-site",
}

def _is_int_or_none(x: Any) -> bool:
    return x is None or isinstance(x, int)

def validate_entity_list(
    obj: Any,
    field_name: str,
    errors: List[str],
) -> Optional[List[Dict[str, Any]]]:
    if obj is None:
        return None
    if not isinstance(obj, list):
        errors.append(f"{field_name}_must_be_array")
        return None

    out: List[Dict[str, Any]] = []
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
                else:
                    vv = v.strip().lower()
                    if vv not in ALLOWED_LEVELS:
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

    ok = len(errors) == 0
    return ok, errors, warnings


# ----------------------------
# Step2: map extractor_json -> backend payload
# ----------------------------

def op_to_group(op: str) -> str:
    if op == "AND":
        return "all"
    if op == "OR":
        return "or"
    if op == "NOT":
        return "not"
    return "all"

def entities_to_groups(entities: Optional[List[Dict[str, Any]]]) -> List[List[Any]]:
    if not entities:
        return []
    by_group: Dict[str, List[str]] = {"all": [], "or": [], "not": []}
    for e in entities:
        rt = e.get("raw_text")
        op = e.get("operator")
        if not isinstance(rt, str) or not isinstance(op, str):
            continue
        g = op_to_group(op)
        by_group[g].append(rt)

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
        if not vv:
            continue
        if vv.lower() in OFFICE_GEO_TRASH:
            continue
        out.append(vv)
    return out

def map_level_to_seniority(levels: Any) -> Optional[List[str]]:
    if not isinstance(levels, list):
        return None
    mapped: List[str] = []
    for v in levels:
        if not isinstance(v, str):
            continue
        vv = v.strip().lower()
        if vv == "junior":
            mapped.append("Junior")
        elif vv == "middle":
            mapped.append("Middle")
        elif vv == "senior":
            mapped.append("Senior")
        elif vv == "lead":
            mapped.append("Lead")
        elif vv == "head":
            mapped.append("Head")
        elif vv == "c-level":
            mapped.append("C-Level")
    mapped = dedupe_keep_order(mapped)
    return mapped or None

def range_obj_to_str_pair(obj: Any) -> Optional[List[str]]:
    if not isinstance(obj, dict):
        return None
    f = obj.get("from")
    t = obj.get("to")
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
    ru = lang.get("russian")
    en = lang.get("english")
    if isinstance(ru, bool):
        payload["onlyRussian"] = ru
    if isinstance(en, bool):
        payload["onlyEnglish"] = en

def build_step3_payload(
    extractor_json: Dict[str, Any],
    user_phrase: str,
    base_payload: Dict[str, Any],
    sanitize_office_geo: bool,
) -> Dict[str, Any]:
    payload = dict(base_payload)
    payload["user_phrase"] = user_phrase

    pos = extractor_json.get("positions")
    if isinstance(pos, list):
        groups = entities_to_groups([x for x in pos if isinstance(x, dict)])
        if groups:
            payload["positions"] = groups

    skills = extractor_json.get("skills")
    if isinstance(skills, list):
        groups = entities_to_groups([x for x in skills if isinstance(x, dict)])
        if groups:
            payload["skills"] = groups

    loc = extractor_json.get("locations")
    if isinstance(loc, list):
        geos_groups = entities_to_groups([x for x in loc if isinstance(x, dict)])
        if sanitize_office_geo:
            sanitized: List[List[Any]] = []
            for g, vals in geos_groups:
                if not isinstance(vals, list):
                    continue
                vals2 = sanitize_geos([str(v) for v in vals])
                if vals2:
                    sanitized.append([g, vals2])
            geos_groups = sanitized
        if geos_groups:
            payload["geos"] = geos_groups

    comp = extractor_json.get("companies")
    if isinstance(comp, list):
        firms_groups = entities_to_groups([x for x in comp if isinstance(x, dict)])
        if firms_groups:
            payload["firms"] = firms_groups

    kw = extractor_json.get("keywords")
    if isinstance(kw, list):
        keys_groups = entities_to_groups([x for x in kw if isinstance(x, dict)])
        if keys_groups:
            payload["keys"] = keys_groups

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


# ----------------------------
# Step1: OpenAI Responses API call (HTTP)
# ----------------------------

def extract_response_text(resp: Dict[str, Any]) -> str:
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"]

    out = resp.get("output")
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        chunks.append(c["text"])
        if chunks:
            return "\n".join(chunks).strip()

    return ""

def call_openai_step1(
    api_key: str,
    prompt_cfg: PromptCfg,
    user_input: str,
    timeout_s: int,
) -> Tuple[Optional[str], Dict[str, int], Optional[str]]:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if not prompt_cfg.prompt_id:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "step1_prompt_id_missing"

    payload: Dict[str, Any] = {
        "model": prompt_cfg.model,
        "input": user_input,
        "prompt": {"id": prompt_cfg.prompt_id},
    }
    if prompt_cfg.prompt_version:
        payload["prompt"]["version"] = str(prompt_cfg.prompt_version)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except Exception as e:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, str(e)

    if r.status_code >= 400:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, f"openai_http_{r.status_code}:{r.text[:500]}"

    try:
        data = r.json()
    except Exception as e:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, f"openai_bad_json:{e}"

    usage_raw = data.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("input_tokens") or 0),
        "output_tokens": int(usage_raw.get("output_tokens") or 0),
        "total_tokens": int(usage_raw.get("total_tokens") or 0),
    }

    text = extract_response_text(data)
    if not text:
        return None, usage, "openai_empty_output_text"
    return text.strip(), usage, None


# ----------------------------
# Step3: backend call
# ----------------------------

INSUFFICIENT_MSG_RE = re.compile(r"Positions or skills or keys must be set", re.IGNORECASE)

def classify_step3_error(status: int, body_text: str) -> str:
    if status == 400 and INSUFFICIENT_MSG_RE.search(body_text or ""):
        return "insufficient_search_terms"
    return "http_error"

def call_backend_search_bool(
    backend: BackendCfg,
    token: str,
    payload: Dict[str, Any],
) -> Tuple[str, int, int, Optional[int], Optional[str], Optional[Dict[str, Any]]]:
    url = backend.base_url.rstrip("/") + backend.step3_path
    attempts = 0
    last_err: Optional[str] = None
    last_json: Optional[Dict[str, Any]] = None

    req_payload = dict(payload)
    headers = {"Content-Type": "application/json"}
    if backend.token_in_body:
        req_payload["token"] = token
    else:
        headers["X-Auth-Token"] = token

    for _ in range(max(1, backend.retries + 1)):
        attempts += 1
        try:
            r = requests.post(url, headers=headers, json=req_payload, timeout=backend.timeout_s)
        except Exception as e:
            last_err = str(e)
            continue

        status = r.status_code
        text = r.text or ""
        if status >= 400:
            kind = classify_step3_error(status, text)
            if kind == "insufficient_search_terms":
                return "insufficient_search_terms", status, attempts, None, None, None
            last_err = text[:800]
            continue

        try:
            data = r.json()
            last_json = data if isinstance(data, dict) else None
        except Exception as e:
            last_err = f"backend_bad_json:{e}"
            return "bad_json", status, attempts, None, last_err, None

        if backend.require_count and (not isinstance(last_json, dict) or "count" not in last_json):
            return "missing_count", status, attempts, None, "backend_missing_count", last_json

        count = None
        if isinstance(last_json, dict):
            c = last_json.get("count")
            if isinstance(c, int):
                count = c

        return "success", status, attempts, count, None, last_json

    return "http_error", 0, attempts, None, last_err, last_json


# ----------------------------
# Reporting structs
# ----------------------------

@dataclass
class CaseResult:
    name: str
    input: str
    source: str
    status: str
    ok: bool
    step1: Dict[str, Any]
    step3: Dict[str, Any]
    extractor_json: Optional[Dict[str, Any]] = None
    step3_payload: Optional[Dict[str, Any]] = None

def compute_pass_rate(ok_count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((ok_count / total) * 100.0, 2)


# ----------------------------
# Main
# ----------------------------

def parse_steps(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [1, 2, 3]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
            if v in (1, 2, 3) and v not in out:
                out.append(v)
        except Exception:
            pass
    return out or [1, 2, 3]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run extractor-agent step1/2/3 tests and produce a compact report.")
    ap.add_argument("--cases-dir", required=True, help="Path to fixtures directory (tests/fixtures/extractor_agent)")
    ap.add_argument("--cases-count", type=int, default=20, help="Per-unit count for mixing (see --mix-ratios).")
    ap.add_argument("--mix-ratios", type=str, default="real=1,suite=1,syn=1", help="real=1,suite=1,syn=1")
    ap.add_argument("--mix-seed", type=int, default=42, help="Seed for mixing/shuffling")
    ap.add_argument("--mix-total-limit", type=int, default=0, help="If >0, cap total mixed cases to this number")

    ap.add_argument("--include-suite", action="store_true", default=True, help="Include built-in suite cases (default: on)")
    ap.add_argument("--no-include-suite", action="store_true", default=False, help="Disable built-in suite cases")
    ap.add_argument("--suite-count", type=int, default=20, help="How many built-in suite cases to use (default: 20)")

    ap.add_argument("--synthetic-count", type=int, default=0, help="How many synthetic cases to generate (syn_*)")
    ap.add_argument("--synthetic-seed", type=int, default=1234)

    ap.add_argument("--steps", type=str, default="1,2,3", help="Comma-separated steps: 1,2,3")

    ap.add_argument("--cfg", type=str, default="tests/tools/model.yaml", help="Path to model/prompt yaml config")
    ap.add_argument("--prompt-id", type=str, default="", help="Override prompt_id")
    ap.add_argument("--prompt-version", type=str, default="", help="Override prompt_version (optional)")
    ap.add_argument("--model", type=str, default="", help="Override model name")

    # Defaults from env so you don't type them each run:
    ap.add_argument("--base-url", type=str, default=os.getenv("AI_SEARCH_BASE_URL", "").strip(),
                    help='Backend base url (default: env AI_SEARCH_BASE_URL)')
    ap.add_argument("--step3-path", type=str, default="/site/searchBool", help='Backend step3 path (default: /site/searchBool)')
    ap.add_argument("--token", type=str, default=os.getenv("AI_SEARCH_AUTH_TOKEN", "").strip(),
                    help='Backend auth token (default: env AI_SEARCH_AUTH_TOKEN)')

    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--step3-retries", type=int, default=2)

    ap.add_argument("--token-in-body", dest="token_in_body", action="store_true", default=True)
    ap.add_argument("--token-in-header", dest="token_in_body", action="store_false")

    ap.add_argument("--sanitize-office-geo", action="store_true", default=True)
    ap.add_argument("--no-sanitize-office-geo", action="store_true", default=False)

    ap.add_argument("--require-search-terms", action="store_true", default=True)
    ap.add_argument("--no-require-search-terms", action="store_true", default=False)

    ap.add_argument("--require-count", action="store_true", default=True)
    ap.add_argument("--no-require-count", action="store_true", default=False)

    ap.add_argument("--only-russian", action="store_true", default=False)
    ap.add_argument("--only-english", action="store_true", default=False)
    ap.add_argument("--only-with-contacts", action="store_true", default=True)
    ap.add_argument("--only-with-higher-education", action="store_true", default=False)
    ap.add_argument("--current-position-title", action="store_true", default=True)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", default=False)
    ap.add_argument("--highlight", action="store_true", default=True)

    ap.add_argument("--report-dir", type=str, default="tests/reports/extractor_agent_full")
    ap.add_argument("--report-mode", type=str, default="compact", choices=["compact", "full"])
    ap.add_argument("--report-json-indent", type=int, default=2)

    args = ap.parse_args()

    # validate required runtime params (step3 defaults from env, but still must exist if step3 runs)
    if 3 in parse_steps(args.steps):
        if not args.base_url:
            raise SystemExit("Step3 enabled but base_url is missing. Set AI_SEARCH_BASE_URL or pass --base-url.")
        if not args.token:
            raise SystemExit("Step3 enabled but token is missing. Set AI_SEARCH_AUTH_TOKEN or pass --token.")

    steps = parse_steps(args.steps)
    ratios = parse_mix_ratios(args.mix_ratios)
    total_limit = args.mix_total_limit if args.mix_total_limit and args.mix_total_limit > 0 else None

    sanitize_office_geo = args.sanitize_office_geo and (not args.no_sanitize_office_geo)
    require_search_terms = args.require_search_terms and (not args.no_require_search_terms)
    require_count = args.require_count and (not args.no_require_count)

    cfg_path = Path(args.cfg)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}

    def _resolve_prompt_from_cfg(cfg_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        block = {}
        if isinstance(cfg_obj.get("extractor_agent"), dict):
            block = cfg_obj.get("extractor_agent") or {}
        elif isinstance(cfg_obj.get("extractor"), dict):
            block = cfg_obj.get("extractor") or {}

        pid = None
        pver = None
        if block:
            pid = block.get("prompt_id") or block.get("promptId")
            pver = block.get("prompt_version") or block.get("promptVersion")

        if not pid:
            pid = deep_get(cfg_obj, ["prompt_id"]) or deep_get(cfg_obj, ["prompt", "prompt_id"]) or deep_get(cfg_obj, ["prompt", "id"])
        if not pver:
            pver = deep_get(cfg_obj, ["prompt_version"]) or deep_get(cfg_obj, ["prompt", "prompt_version"]) or deep_get(cfg_obj, ["prompt", "version"])

        return (str(pid) if pid else None, str(pver) if pver else None)

    cfg_pid, cfg_pver = _resolve_prompt_from_cfg(cfg)

    env_pid = os.getenv("EXTRACTOR_AGENT_PROMPT_ID") or None
    env_pver = os.getenv("EXTRACTOR_AGENT_PROMPT_VERSION") or None

    prompt_id = (args.prompt_id or "").strip() or cfg_pid or env_pid
    prompt_version = (args.prompt_version or "").strip() or cfg_pver or env_pver

    model = args.model or deep_get(cfg, ["model"]) or deep_get(cfg, ["openai", "model"]) or "gpt-4.1"

    if 1 in steps:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise SystemExit("Step1 enabled but OPENAI_API_KEY is missing in env.")
        if not prompt_id:
            raise SystemExit(
                "Step1 enabled but prompt_id missing. Provide --prompt-id or set it in tests/tools/model.yaml "
                "(extractor_agent.prompt_id) or via EXTRACTOR_AGENT_PROMPT_ID."
            )

    prompt_cfg = PromptCfg(
        prompt_id=prompt_id,
        prompt_version=str(prompt_version) if prompt_version else None,
        model=str(model),
    )

    backend_cfg = BackendCfg(
        base_url=args.base_url,
        step3_path=args.step3_path,
        token_in_body=bool(args.token_in_body),
        timeout_s=int(args.timeout_s),
        retries=int(args.step3_retries),
        sanitize_office_geo=sanitize_office_geo,
        require_search_terms=require_search_terms,
        require_count=require_count,
    )

    run_id = make_run_id()

    print(f"[init] loaded cfg: {cfg_path.resolve() if cfg_path.exists() else str(cfg_path)}")

    all_cases = load_cases_from_dir(Path(args.cases_dir))
    real_cases = [c for c in all_cases if c.source == "real"]

    use_suite = bool(args.include_suite) and (not bool(args.no_include_suite))
    suite_cases = build_suite_cases(limit=int(args.suite_count), seed=int(args.mix_seed)) if use_suite else []

    syn_base = real_cases + suite_cases
    syn_cases = build_synthetic_cases(syn_base, args.synthetic_count, args.synthetic_seed)

    mixed, counts = mix_cases(
        real_cases=real_cases,
        suite_cases=suite_cases,
        syn_cases=syn_cases,
        per_unit_count=int(args.cases_count),
        ratios=ratios,
        seed=int(args.mix_seed),
        total_limit=total_limit,
    )

    runner_cfg = RunnerCfg(
        cases_dir=args.cases_dir,
        counts=counts,
        mix=MixCfg(ratios=ratios, seed=int(args.mix_seed), total_limit=total_limit),
        steps=steps,
        prompt=prompt_cfg,
        backend=backend_cfg,
    )

    step3_url = backend_cfg.base_url.rstrip("/") + backend_cfg.step3_path

    print(
        "[init] "
        f"run_id={run_id} "
        f"cases_dir={args.cases_dir} "
        f"counts={counts} "
        f"steps={steps} "
        f"prompt_id={prompt_cfg.prompt_id} prompt_version={prompt_cfg.prompt_version} model={prompt_cfg.model} "
        f"base_url={backend_cfg.base_url} step3_url={step3_url} "
        f"token_in_body={backend_cfg.token_in_body} timeout_s={backend_cfg.timeout_s} retries={backend_cfg.retries} "
        f"sanitize_office_geo={backend_cfg.sanitize_office_geo} require_search_terms={backend_cfg.require_search_terms} require_count={backend_cfg.require_count} "
        f"suite_enabled={use_suite} suite_count={len(suite_cases)}"
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

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    results: List[CaseResult] = []

    step1_errors_total = 0
    step3_http_errors = 0
    insufficient_count = 0
    passed = 0
    failed_step1 = 0
    failed_step3_or_step2 = 0
    zero_results = 0

    mismatches: List[Dict[str, Any]] = []
    insufficient_cases: List[Dict[str, str]] = []

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    for idx, case in enumerate(mixed, start=1):
        print(f"[run] case {idx}/{len(mixed)} name={case.name} input={case.input!r}")

        extractor_json: Optional[Dict[str, Any]] = None
        step3_payload: Optional[Dict[str, Any]] = None

        step1_info: Dict[str, Any] = {"ok": True}
        step3_info: Dict[str, Any] = {"kind": "skipped"}

        step1_ok = True
        step1_errors: List[str] = []
        step1_warnings: List[str] = []

        if 1 in steps:
            text, usage, err = call_openai_step1(
                api_key=openai_key,
                prompt_cfg=prompt_cfg,
                user_input=case.input,
                timeout_s=int(args.timeout_s),
            )
            usage_total["input_tokens"] += usage["input_tokens"]
            usage_total["output_tokens"] += usage["output_tokens"]
            usage_total["total_tokens"] += usage["total_tokens"]

            if err or not text:
                step1_ok = False
                step1_errors = [err or "openai_no_text"]
                step1_info = {"ok": False, "errors": step1_errors}
            else:
                obj, jerr = safe_json_loads(text)
                if jerr or obj is None:
                    step1_ok = False
                    step1_errors = ["step1_invalid_json"]
                    step1_info = {"ok": False, "errors": step1_errors}
                    if isinstance(obj, dict):
                        extractor_json = obj
                else:
                    if isinstance(obj, dict):
                        extractor_json = obj
                        step1_ok, step1_errors, step1_warnings = validate_step1_contract(extractor_json, case.input)
                        step1_info = {"ok": step1_ok}
                        if step1_errors:
                            step1_info["errors"] = step1_errors
                        if step1_warnings:
                            step1_info["warnings"] = step1_warnings
                    else:
                        step1_ok = False
                        step1_errors = ["step1_json_must_be_object"]
                        step1_info = {"ok": False, "errors": step1_errors}

        if not step1_ok:
            step1_errors_total += len(step1_errors)

        if (1 in steps) and (not step1_ok):
            status = "failed_step1"
            ok = False
            failed_step1 += 1

            results.append(CaseResult(
                name=case.name,
                input=case.input,
                source=case.source,
                status=status,
                ok=ok,
                step1=step1_info,
                step3={"kind": "skipped_due_to_failed_step1"},
                extractor_json=extractor_json,
                step3_payload=step3_payload,
            ))

            mismatches.append({
                "name": case.name,
                "input": case.input,
                "status": status,
                "ok": ok,
                "step1": step1_info,
                "step3": {"kind": "skipped_due_to_failed_step1"},
                "extractor_json": extractor_json,
            })
            continue

        if 2 in steps:
            if extractor_json is None:
                extractor_json = {}
            step3_payload = build_step3_payload(
                extractor_json=extractor_json,
                user_phrase=case.input,
                base_payload=base_payload,
                sanitize_office_geo=sanitize_office_geo,
            )

        if 3 in steps:
            if step3_payload is None:
                step3_payload = dict(base_payload)
                step3_payload["user_phrase"] = case.input

            kind, status_code, attempts, count, err_msg, _resp_json = call_backend_search_bool(
                backend=backend_cfg,
                token=args.token,
                payload=step3_payload,
            )
            step3_info = {"kind": kind, "status": status_code, "attempts": attempts}
            if kind == "success":
                step3_info["count"] = count
                if count == 0:
                    zero_results += 1
            elif kind == "insufficient_search_terms":
                insufficient_count += 1
                insufficient_cases.append({"name": case.name, "input": case.input})
            else:
                step3_http_errors += 1
                if err_msg:
                    step3_info["error_message"] = err_msg

        if step3_info.get("kind") == "insufficient_search_terms":
            status = "insufficient_search_terms"
            ok = True
        elif step3_info.get("kind") == "success" or (3 not in steps):
            status = "passed"
            ok = True
            passed += 1
        else:
            status = "failed_step3_or_step2"
            ok = False
            failed_step3_or_step2 += 1

        results.append(CaseResult(
            name=case.name,
            input=case.input,
            source=case.source,
            status=status,
            ok=ok,
            step1=step1_info,
            step3=step3_info,
            extractor_json=extractor_json,
            step3_payload=step3_payload,
        ))

        if status == "failed_step3_or_step2":
            mm = {
                "name": case.name,
                "input": case.input,
                "status": status,
                "ok": ok,
                "step1": step1_info,
                "step3": step3_info,
                "extractor_json": extractor_json,
            }
            if args.report_mode == "full":
                mm["step3_payload"] = step3_payload
            mismatches.append(mm)

    total = len(results)
    ok_total = passed + insufficient_count
    pass_rate = compute_pass_rate(ok_total, total)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "config": {
            "cases_dir": runner_cfg.cases_dir,
            "counts": runner_cfg.counts,
            "mix": {
                "ratios": runner_cfg.mix.ratios,
                "seed": runner_cfg.mix.seed,
                "total_limit": runner_cfg.mix.total_limit,
            },
            "steps": runner_cfg.steps,
            "prompt": {
                "prompt_id": runner_cfg.prompt.prompt_id,
                "prompt_version": runner_cfg.prompt.prompt_version,
                "model": runner_cfg.prompt.model,
            },
            "backend": {
                "base_url": runner_cfg.backend.base_url,
                "step3_path": runner_cfg.backend.step3_path,
                "token_in_body": runner_cfg.backend.token_in_body,
                "timeout_s": runner_cfg.backend.timeout_s,
                "retries": runner_cfg.backend.retries,
                "sanitize_office_geo": runner_cfg.backend.sanitize_office_geo,
                "require_search_terms": runner_cfg.backend.require_search_terms,
                "require_count": runner_cfg.backend.require_count,
            },
            "suite": {
                "enabled": use_suite,
                "suite_count": len(suite_cases),
            },
        },
        "token_usage_total": usage_total,
        "summary": {
            "results_total": total,
            "passed": passed,
            "insufficient_search_terms": insufficient_count,
            "failed_step1": failed_step1,
            "failed_step3_or_step2": failed_step3_or_step2,
            "pass_rate": pass_rate,
            "pass_rate_strict": round((passed / total) * 100.0, 2) if total else 0.0,
            "step1_errors_total": step1_errors_total,
            "step3_http_errors": step3_http_errors,
            "zero_results": zero_results,
            "mismatches_count": len(mismatches),
        },
        "cases": [],
        "mismatches": mismatches,
        "insufficient_cases": insufficient_cases,
    }

    for r in results:
        report["cases"].append({
            "name": r.name,
            "source": r.source,
            "input": r.input,
            "status": r.status,
            "ok": r.ok,
            "step1": r.step1,
            "step3": r.step3,
            "extractor_json": r.extractor_json or {},
            "step3_payload": r.step3_payload or {},
        })

    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)
    out_path = report_dir / f"extractor_agent_full_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=int(args.report_json_indent)), encoding="utf-8")

    print(
        f"[summary] total={total} passed={passed} insufficient={insufficient_count} "
        f"failed_step1={failed_step1} failed_step3_or_step2={failed_step3_or_step2} "
        f"pass_rate={pass_rate:.2f} pass_rate_strict={report['summary']['pass_rate_strict']:.2f} "
        f"step1_errors_total={step1_errors_total} step3_http_errors={step3_http_errors} zero_results={zero_results}"
    )
    print(f"[done] report saved: {out_path}")

    return 0 if len(mismatches) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
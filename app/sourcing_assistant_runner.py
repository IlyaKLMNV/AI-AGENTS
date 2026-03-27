from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pathlib
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

try:
    from app.extractor_agent_runner import BackendCfg, call_backend_search_bool
except Exception:
    from extractor_agent_runner import BackendCfg, call_backend_search_bool  # type: ignore

# Repo root: if this file is in app/, parents[1] is repo root.
ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
REPORTS_DIR = ROOT / "tests" / "reports" / "sourcing_assistant"

REPORT_VERBOSITY_VALUES = ("compact", "standard", "full")
REQUIREMENTS_SOURCE_VALUES = ("cdm_key_requirements", "stack_skills", "responsibilities_parser")
SAMPLE_MODE_VALUES = ("first", "random")
NON_GEO_LOCATIONS = {"remote", "hybrid", "office", "onsite", "on-site", "удаленно", "удалённо", "офис"}


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cdm_files(cdm_dir: pathlib.Path, cdm_count: Optional[int]) -> List[pathlib.Path]:
    if not cdm_dir.exists():
        raise FileNotFoundError(f"CDM dir not found: {cdm_dir}")

    paths = [pathlib.Path(p) for p in sorted(glob.glob(str(cdm_dir / "cdm_*.json")))]
    if not paths:
        raise FileNotFoundError(f"No cdm_*.json found in: {cdm_dir}")

    if cdm_count is not None:
        if cdm_count <= 0:
            raise ValueError("--cdm-count must be > 0")
        paths = paths[:cdm_count]

    return paths


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        it = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input_token_count") or 0
        ot = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output_token_count") or 0
        tt = usage.get("total_tokens") or usage.get("token_count")
    else:
        it = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        ot = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        tt = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)

    if tt is None:
        tt = (it or 0) + (ot or 0)

    return int(it or 0), int(ot or 0), int(tt or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    it, ot, tt = _extract_usage_numbers(usage)
    bucket["input_tokens"] += it
    bucket["output_tokens"] += ot
    bucket["total_tokens"] += tt


def _resolve_prompt_from_cfg(cfg: Dict[str, Any], block_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # tests/tools/model.yaml:
    # sourcing_assistant:
    #   prompt_id: pmpt_...
    #   prompt_version: ...
    #   seed: 1234
    block = cfg.get(block_name) or {}
    pid = block.get("prompt_id")
    pver = block.get("prompt_version")
    seed = block.get("seed")
    return (str(pid) if pid else None, str(pver) if pver else None, int(seed) if seed is not None else None)


def _resolve_requirements_source_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    block = cfg.get("sourcing_assistant") or {}
    v = block.get("requirements_source")
    return str(v) if v else None


def _split_list_like(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\n;|]+", str(s))
    out: List[str] = []
    for p in parts:
        t = re.sub(r"\s+", " ", (p or "").strip())
        if t:
            out.append(t)
    return out


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = x.strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _norm_key(s: str) -> str:
    t = (s or "").strip().lower()
    t = t.replace("ё", "е")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip()
    # keep latin+cyrillic+digits, drop punctuation/spaces
    t = re.sub(r"[^a-z0-9а-я]+", "", t, flags=re.IGNORECASE)
    return t


def _contains_norm(needle: str, hay: str) -> bool:
    if not needle or not hay:
        return False

    # quick path
    if needle.strip().lower() in hay.lower():
        return True

    n = _norm_key(needle)
    h = _norm_key(hay)
    if not n or not h:
        return False
    return n in h


def _parse_json_array_strict(raw: str) -> List[Any]:
    if not raw:
        raise ValueError("empty output")

    s = raw.strip()
    if not s.startswith("["):
        m = re.search(r"\[\s*(?:.|\n)*\s*\]", s)
        if not m:
            raise ValueError(f"output is not a JSON array: {raw!r}")
        s = m.group(0)

    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"invalid JSON: {e!r}; raw={raw!r}") from e

    if not isinstance(obj, list):
        raise ValueError(f"output JSON is not a list: {raw!r}")
    return obj


def _parse_json_array_of_strings(raw: str) -> List[str]:
    arr = _parse_json_array_strict(raw)
    out: List[str] = []
    for i, v in enumerate(arr):
        if not isinstance(v, str):
            raise ValueError(f"requirements item[{i}] is not a string: {raw!r}")
        t = re.sub(r"\s+", " ", v.strip())
        if t:
            out.append(t)
    return out


def _parse_json_array_of_objects(raw: str) -> List[Dict[str, Any]]:
    arr = _parse_json_array_strict(raw)
    out: List[Dict[str, Any]] = []
    for i, v in enumerate(arr):
        if not isinstance(v, dict):
            raise ValueError(f"output item[{i}] is not an object: {raw!r}")
        out.append(v)
    return out


class VacancyTextBuilder:
    """
    Детерминированно собирает текст вакансии из CDM.
    (Можно использовать для responsibilities_parser, если выберете requirements_source=responsibilities_parser)
    """

    def build(self, vacancy: Dict[str, Any]) -> str:
        title = vacancy.get("title") or "Вакансия"
        company = vacancy.get("company_name") or "Компания"
        industry = vacancy.get("company_industry") or ""
        location = vacancy.get("location") or ""
        work_format = vacancy.get("work_format") or ""
        company_desc = vacancy.get("company_description") or ""
        responsibilities = vacancy.get("responsibilities") or ""
        stack = vacancy.get("vacancy_stack") or ""
        skills = vacancy.get("vacancy_skills") or ""
        questions = vacancy.get("questions") or ""

        meta_lines = [f"Должность: {title}", f"Компания: {company}"]
        if industry:
            meta_lines.append(f"Индустрия: {industry}")
        if location:
            meta_lines.append(f"Локация: {location}")
        if work_format:
            meta_lines.append(f"Формат: {work_format}")

        blocks: List[str] = []
        blocks.append("\n".join(meta_lines))

        if company_desc:
            blocks.append("О компании:\n" + company_desc)

        if responsibilities:
            blocks.append("Обязанности:\n" + responsibilities)

        # Явно вставляем стек/скиллы (чтобы terms точно присутствовали в тексте)
        req_lines: List[str] = []
        stack_items = _split_list_like(stack)
        skills_items = _split_list_like(skills)
        if stack_items:
            req_lines.append("Технологии/инструменты:")
            req_lines.extend(f"- {x}" for x in stack_items)
        if skills_items:
            req_lines.append("Ключевые требования:")
            req_lines.extend(f"- {x}" for x in skills_items)
        if req_lines:
            blocks.append("Требования:\n" + "\n".join(req_lines))

        if questions:
            blocks.append("Вопросы:\n" + questions)

        return "\n\n".join([b for b in blocks if b]).strip()


class ResponsibilitiesParserRunner:
    def __init__(self, prompt_id: str, prompt_version: Optional[str]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def extract(self, vacancy_text: str) -> Tuple[List[str], str]:
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=(vacancy_text or "").strip(),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
        items = _parse_json_array_of_strings(raw)

        # responsibilities_parser гарантирует 1..5, но на всякий случай чистим
        items = _dedupe_preserve_order(items)[:5]
        return items, raw


class SourcingAssistantRunner:
    def __init__(self, prompt_id: str, prompt_version: Optional[str]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def run(self, requirements: List[str], profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        payload = {"requirements": requirements, "profile": profile}
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=json.dumps(payload, ensure_ascii=False),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
        items = _parse_json_array_of_objects(raw)
        return items, raw


def _build_profile_from_cdm_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    CDM candidate -> "profile" в формате, близком к вашему поисковому JSON:
    - about (str)
    - skills[]: {skill: str, ...}
    - positions[]: {name, pos, description, rangeStr, dates, current, positions_norm, company_norm{categories}}
    """
    cand_name = candidate.get("candidate_name") or candidate.get("name") or ""
    recruiter = candidate.get("recruiter_name") or ""
    about = f"{cand_name}".strip()
    if recruiter:
        about = (about + f". Recruiter: {recruiter}").strip(". ").strip()

    # skills
    skills_in = candidate.get("candidate_skills") or []
    skills_out: List[Dict[str, Any]] = []
    for s in skills_in:
        if not isinstance(s, dict):
            continue
        sk = s.get("skill")
        if isinstance(sk, str) and sk.strip():
            skills_out.append({"skill": re.sub(r"\s+", " ", sk.strip())})

    # positions
    jobs_in = candidate.get("candidate_job_list") or []
    positions_out: List[Dict[str, Any]] = []
    for j in jobs_in:
        if not isinstance(j, dict):
            continue
        title = j.get("title") or ""
        company = j.get("company") or ""
        company_norm = j.get("company_norm") or {}
        if not isinstance(company_norm, dict):
            company_norm = {}

        positions_out.append(
            {
                "name": re.sub(r"\s+", " ", str(company).strip()),
                "pos": re.sub(r"\s+", " ", str(title).strip()),
                "description": "",          # в CDM нет — оставляем пустым
                "rangeStr": "",             # в CDM нет — оставляем пустым
                "dates": [],                # в CDM нет — оставляем пустым
                "current": False,           # неизвестно
                "positions_norm": [],
                "company_norm": {
                    "categories": company_norm.get("categories") or []
                },
            }
        )

    return {
        "about": about,
        "skills": skills_out,
        "positions": positions_out,
    }


def _requirements_from_cdm_key_requirements(vacancy: Dict[str, Any]) -> List[str]:
    raw = vacancy.get("key_requirements")
    items: List[str] = []
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, str):
                t = re.sub(r"\s+", " ", value.strip())
                if t:
                    items.append(t)
    else:
        items = _split_list_like(str(raw) if raw is not None else None)
    return _dedupe_preserve_order(items)[:5]


def _requirements_from_stack_skills(vacancy: Dict[str, Any]) -> List[str]:
    stack = _split_list_like(vacancy.get("vacancy_stack") or "")
    skills = _split_list_like(vacancy.get("vacancy_skills") or "")
    merged = _dedupe_preserve_order(stack + skills)
    return merged[:5]


def _is_real_geo_value(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    return bool(text) and text not in NON_GEO_LOCATIONS


def _build_backend_search_payload(
    vacancy: Dict[str, Any],
    requirements: List[str],
    candidate_pool_size: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "limit": int(candidate_pool_size),
        "offset": 0,
        "onlyWithContacts": True,
        "currentPositionTitle": True,
        "shuffle": False,
        "highlight": True,
    }

    title = re.sub(r"\s+", " ", str(vacancy.get("title") or "").strip())
    if title:
        payload["positions"] = [["all", [title]]]

    reqs = _dedupe_preserve_order([re.sub(r"\s+", " ", str(x).strip()) for x in requirements if str(x).strip()])
    if reqs:
        payload["keys"] = [["or", reqs]]

    location = vacancy.get("location")
    if _is_real_geo_value(location):
        payload["geos"] = [["or", [re.sub(r"\s+", " ", str(location).strip())]]]

    return payload


def _sample_backend_profiles(
    profiles: List[Dict[str, Any]],
    sample_size: int,
    sample_mode: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    clean = [p for p in profiles if isinstance(p, dict)]
    if sample_size <= 0 or not clean:
        return []

    take_n = min(sample_size, len(clean))
    if sample_mode == "first":
        return clean[:take_n]
    if take_n == len(clean):
        return clean
    indices = sorted(rng.sample(range(len(clean)), k=take_n))
    return [clean[i] for i in indices]


def _build_profile_from_backend_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "name": candidate.get("name"),
        "about": candidate.get("about") or "",
        "geo": candidate.get("geo") or "",
        "geos": candidate.get("geos") or [],
        "skills": candidate.get("skills") or [],
        "positions": candidate.get("positions") or [],
        "positions_array": candidate.get("positions_array") or [],
        "positions_array_current": candidate.get("positions_array_current") or [],
        "pastPositions": candidate.get("pastPositions") or [],
        "experience": candidate.get("experience") or [],
        "seniority": candidate.get("seniority") or [],
        "educations": candidate.get("educations") or [],
        "is_english": bool(candidate.get("is_english")),
        "is_russian": bool(candidate.get("is_russian")),
        "has_higher_education": bool(candidate.get("has_higher_education")),
    }


def _profile_search_texts(profile: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        t = re.sub(r"\s+", " ", value.strip())
        if t:
            texts.append(t)

    add(profile.get("about"))
    add(profile.get("name"))
    add(profile.get("geo"))

    for field in ("positions_array", "positions_array_current", "pastPositions"):
        for item in (profile.get(field) or []):
            add(item)

    for skill in (profile.get("skills") or []):
        if isinstance(skill, dict):
            add(skill.get("skill"))
            category = skill.get("category") or {}
            if isinstance(category, dict):
                add(category.get("title"))
        else:
            add(skill)

    for pos in (profile.get("positions") or []):
        if not isinstance(pos, dict):
            continue
        for key in ("name", "pos", "description", "rangeStr", "type"):
            add(pos.get(key))

        for value in (pos.get("positions_norm") or []):
            if isinstance(value, dict):
                for key in ("title", "name", "raw_text"):
                    add(value.get(key))
            else:
                add(value)

        company_norm = pos.get("company_norm") or {}
        if isinstance(company_norm, dict):
            add(company_norm.get("title"))
            add(company_norm.get("site"))
            for category in (company_norm.get("categories") or []):
                if isinstance(category, dict):
                    add(category.get("title"))
                else:
                    add(category)

        pos_skills = pos.get("skills")
        if isinstance(pos_skills, list):
            for item in pos_skills:
                if isinstance(item, dict):
                    add(item.get("skill"))
                else:
                    add(item)

    for geo in (profile.get("geos") or []):
        if isinstance(geo, dict):
            for key in ("title", "level", "render_type"):
                add(geo.get(key))
        else:
            add(geo)

    for exp in (profile.get("experience") or []):
        if isinstance(exp, dict):
            add(exp.get("type"))
            add(exp.get("text"))

    for seniority in (profile.get("seniority") or []):
        if not isinstance(seniority, dict):
            continue
        add(seniority.get("level"))
        add(seniority.get("reason"))
        category = seniority.get("category") or {}
        if isinstance(category, dict):
            add(category.get("title"))

    for edu in (profile.get("educations") or []):
        if not isinstance(edu, dict):
            continue
        add(edu.get("institutionName"))
        add(edu.get("specialization"))
        university_norm = edu.get("university_norm") or {}
        if isinstance(university_norm, dict):
            add(university_norm.get("title"))

    return _dedupe_preserve_order(texts)


def _expected_passed_for_requirement(req: str, profile: Dict[str, Any]) -> int:
    """
    Детерминированная "истина" для теста:
    passed=1 если requirement встречается (с нормализацией) в about/skills/positions/categories.
    """
    if not req:
        return 0

    req_low = req.strip().lower()

    if ("english" in req_low or "англий" in req_low) and bool(profile.get("is_english")):
        return 1
    if ("russian" in req_low or "русск" in req_low) and bool(profile.get("is_russian")):
        return 1
    if (("higher education" in req_low) or ("высш" in req_low and "образ" in req_low)) and bool(profile.get("has_higher_education")):
        return 1

    for text in _profile_search_texts(profile):
        if _contains_norm(req, text):
            return 1

    return 0


def _validate_output_item_shape(item: Dict[str, Any]) -> List[str]:
    """
    Строго: только requirement/comment/passed. Никаких лишних полей.
    """
    reasons: List[str] = []
    allowed = {"requirement", "comment", "passed"}
    extra = sorted(set(item.keys()) - allowed)
    missing = sorted(allowed - set(item.keys()))
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

    c = item.get("comment") if isinstance(item.get("comment"), str) else ""
    if "\n" in c or "\r" in c:
        reasons.append("comment_has_newlines")
    if len(c) > 220:
        reasons.append("comment_too_long>220")

    return reasons


def _run_case_strict(requirements: List[str], expected_passed: List[int], predicted: List[Dict[str, Any]]) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    per_item: List[Dict[str, Any]] = []

    if len(predicted) != len(requirements):
        reasons.append(f"len_mismatch predicted={len(predicted)} requirements={len(requirements)}")

    n = min(len(predicted), len(requirements))
    for i in range(n):
        item = predicted[i]
        exp_req = requirements[i]
        exp_pass = expected_passed[i]
        item_reasons: List[str] = []

        if not isinstance(item, dict):
            item_reasons.append("item_not_object")
            per_item.append({"index": i, "ok": False, "reasons": item_reasons})
            continue

        item_reasons.extend(_validate_output_item_shape(item))

        # requirement must be EXACT
        if isinstance(item.get("requirement"), str) and item["requirement"] != exp_req:
            item_reasons.append("requirement_not_exact")

        # passed must match deterministic expected
        if item.get("passed") != exp_pass:
            item_reasons.append("passed_mismatch")

        # if passed=0 -> must include phrase
        if exp_pass == 0:
            c = (item.get("comment") or "")
            if not isinstance(c, str) or "в резюме не указано" not in c.lower():
                item_reasons.append('missing_phrase_required("в резюме не указано")')

        ok = len(item_reasons) == 0
        per_item.append(
            {
                "index": i,
                "ok": ok,
                "reasons": item_reasons,
                "requirement": item.get("requirement"),
                "passed": item.get("passed"),
            }
        )

    if any(not x["ok"] for x in per_item):
        reasons.append("one_or_more_items_failed")

    return len(reasons) == 0, reasons, {"per_item": per_item}


def _run_case_contract(requirements: List[str], expected_passed: List[int], predicted: List[Dict[str, Any]]) -> Tuple[bool, List[str], Dict[str, Any]]:
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
        "passed_mismatch_count": 0,
        "comment_contract_fail_count": 0,
    }

    if len(predicted) != len(requirements):
        issues.append("length_mismatch")

    n = min(len(predicted), len(requirements))
    for i in range(n):
        item = predicted[i]
        exp_req = requirements[i]
        exp_pass = expected_passed[i]

        if not isinstance(item, dict):
            checks["shape_fail_count"] += 1
            failed_items.append(
                {
                    "index": i,
                    "expected_requirement": exp_req,
                    "actual_requirement": None,
                    "expected_passed": exp_pass,
                    "actual_passed": None,
                    "issues": ["item_not_object"],
                }
            )
            continue

        item_issues = _validate_output_item_shape(item)
        if item_issues:
            checks["shape_fail_count"] += 1

        if isinstance(item.get("requirement"), str) and item["requirement"] != exp_req:
            item_issues.append("requirement_not_exact")
            checks["requirement_not_exact_count"] += 1

        if item.get("passed") != exp_pass:
            item_issues.append("passed_mismatch")
            checks["passed_mismatch_count"] += 1

        if exp_pass == 0:
            comment = item.get("comment")
            if not isinstance(comment, str) or "в резюме не указано" not in comment.lower():
                item_issues.append('missing_phrase_required("в резюме не указано")')
                checks["comment_contract_fail_count"] += 1

        if item_issues:
            failed_items.append(
                {
                    "index": i,
                    "expected_requirement": exp_req,
                    "actual_requirement": item.get("requirement"),
                    "expected_passed": exp_pass,
                    "actual_passed": item.get("passed"),
                    "issues": item_issues,
                }
            )

    checks["failed_items_count"] = len(failed_items)

    if checks["shape_fail_count"] > 0:
        issues.append("output_shape_failed")
    if checks["requirement_not_exact_count"] > 0:
        issues.append("requirement_not_exact")
    if checks["passed_mismatch_count"] > 0:
        issues.append("passed_mismatch")
    if checks["comment_contract_fail_count"] > 0:
        issues.append("comment_contract_failed")

    return len(issues) == 0, issues, {"checks": checks, "failed_items": failed_items}


def run_sourcing_assistant_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    cases_count: Optional[int],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    seed: Optional[int],
    requirements_source: str,
    report_verbosity: str,
    base_url: str,
    token: str,
    step3_path: str,
    timeout_s: int,
    step3_retries: int,
    token_in_body: bool,
    candidate_pool_size: int,
    candidate_sample_size: int,
    sample_mode: str,
    quiet: bool,
) -> pathlib.Path:
    ensure_dirs()

    if report_verbosity not in REPORT_VERBOSITY_VALUES:
        raise ValueError(f"--report-verbosity must be one of: {', '.join(REPORT_VERBOSITY_VALUES)}")
    if requirements_source not in REQUIREMENTS_SOURCE_VALUES:
        raise ValueError(f"--requirements-source must be one of: {', '.join(REQUIREMENTS_SOURCE_VALUES)}")
    if sample_mode not in SAMPLE_MODE_VALUES:
        raise ValueError(f"--sample-mode must be one of: {', '.join(SAMPLE_MODE_VALUES)}")
    if candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be > 0")
    if candidate_sample_size <= 0:
        raise ValueError("--candidate-sample-size must be > 0")
    if not base_url:
        raise EnvironmentError("AI_SEARCH_BASE_URL is required (or pass --base-url)")
    if not token:
        raise EnvironmentError("AI_SEARCH_AUTH_TOKEN is required (or pass --token)")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    cfg: Dict[str, Any] = {}
    if CFG_PATH.is_file():
        cfg = load_yaml(CFG_PATH) or {}
        _log(quiet, f"[init] loaded cfg: {CFG_PATH}")
    else:
        _log(quiet, f"[init] cfg not found: {CFG_PATH} (ok, will use env/cli)")

    # sourcing_assistant prompt
    cfg_pid, cfg_pver, cfg_seed = _resolve_prompt_from_cfg(cfg, "sourcing_assistant")

    env_pid = os.environ.get("SOURCING_ASSISTANT_PROMPT_ID")
    env_pver = os.environ.get("SOURCING_ASSISTANT_PROMPT_VERSION")

    final_pid = prompt_id or cfg_pid or env_pid
    final_pver = prompt_version or cfg_pver or env_pver
    if not final_pid:
        raise EnvironmentError(
            "No sourcing_assistant prompt_id found. Provide --prompt-id, or set SOURCING_ASSISTANT_PROMPT_ID, "
            "or add tests/tools/model.yaml -> sourcing_assistant.prompt_id"
        )

    final_seed = seed if seed is not None else cfg_seed
    rng = random.Random(final_seed)

    # requirements_source from cfg if not provided by cli (cli already has a default)
    cfg_req_source = _resolve_requirements_source_from_cfg(cfg)
    if cfg_req_source in REQUIREMENTS_SOURCE_VALUES and requirements_source == "cdm_key_requirements":
        # allow cfg override only if user didn't explicitly pass another in cli
        requirements_source = cfg_req_source

    # responsibilities_parser prompt (optional)
    resp_parser: Optional[ResponsibilitiesParserRunner] = None
    builder = VacancyTextBuilder()
    resp_prompt_id: Optional[str] = None
    resp_prompt_version: Optional[str] = None

    if requirements_source == "responsibilities_parser":
        rpid, rpver, _ = _resolve_prompt_from_cfg(cfg, "responsibilities_parser")
        resp_prompt_id = rpid or os.environ.get("RESPONSIBILITIES_PARSER_PROMPT_ID")
        resp_prompt_version = rpver or os.environ.get("RESPONSIBILITIES_PARSER_PROMPT_VERSION")
        if not resp_prompt_id:
            raise EnvironmentError(
                "requirements_source=responsibilities_parser требует prompt_id. "
                "Добавьте tests/tools/model.yaml -> responsibilities_parser.prompt_id "
                "или установите RESPONSIBILITIES_PARSER_PROMPT_ID"
            )
        resp_parser = ResponsibilitiesParserRunner(prompt_id=resp_prompt_id, prompt_version=resp_prompt_version)

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)
    if cases_count is not None:
        if cases_count <= 0:
            raise ValueError("--cases-count must be > 0")
        sample_count = min(cases_count, len(cdm_paths))
        cdm_paths = rng.sample(cdm_paths, k=sample_count)

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
        f"cdm_dir={cdm_dir} "
        f"cases_count={len(cdm_paths)} "
        f"seed={final_seed} "
        f"requirements_source={requirements_source} "
        f"candidate_pool_size={candidate_pool_size} "
        f"candidate_sample_size={candidate_sample_size} "
        f"sample_mode={sample_mode} "
        f"report_verbosity={report_verbosity}",
    )
    _log(quiet, f"[init] sourcing_assistant prompt_id={final_pid} prompt_version={final_pver}")
    _log(quiet, f"[init] backend base_url={base_url} step3_path={step3_path} token_in_body={token_in_body}")
    if resp_parser:
        _log(quiet, f"[init] responsibilities_parser prompt_id={resp_prompt_id} prompt_version={resp_prompt_version}")

    sa_runner = SourcingAssistantRunner(prompt_id=final_pid, prompt_version=final_pver)
    backend_cfg = BackendCfg(
        base_url=base_url,
        step3_path=step3_path,
        token_in_body=bool(token_in_body),
        timeout_s=int(timeout_s),
        retries=int(step3_retries),
        sanitize_office_geo=True,
        require_search_terms=True,
        require_count=True,
    )

    token_usage_total = _blank_usage()
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total_candidates_evaluated = 0
    total_candidates_passed = 0
    backend_total_found_sum = 0

    for cdm_path in cdm_paths:
        cdm = load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}
        v_title = vacancy.get("title")
        v_company = vacancy.get("company_name")

        # 1) requirements
        raw_req_output = None
        vacancy_text = None
        if requirements_source == "responsibilities_parser" and resp_parser is not None:
            vacancy_text = builder.build(vacancy)
            try:
                requirements, raw_req_output = resp_parser.extract(vacancy_text)
                _accumulate_usage(token_usage_total, resp_parser.last_usage)
            except Exception:
                requirements = _requirements_from_cdm_key_requirements(vacancy) or _requirements_from_stack_skills(vacancy)
        elif requirements_source == "cdm_key_requirements":
            requirements = _requirements_from_cdm_key_requirements(vacancy) or _requirements_from_stack_skills(vacancy)
        else:
            requirements = _requirements_from_stack_skills(vacancy)

        # Ensure we have 1..5 requirements (like parser contract)
        requirements = _dedupe_preserve_order([r.strip() for r in requirements if isinstance(r, str) and r.strip()])[:5]
        if not requirements:
            # If nothing to test — skip
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "error": "no_requirements_generated",
                }
            )
            continue

        # 2) backend search for real candidates
        search_payload = _build_backend_search_payload(vacancy=vacancy, requirements=requirements, candidate_pool_size=candidate_pool_size)
        kind, status_code, attempts, found_count, backend_error, backend_response = call_backend_search_bool(
            backend=backend_cfg,
            token=token,
            payload=search_payload,
        )
        if kind != "success":
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "error": backend_error or kind,
                    "backend_kind": kind,
                    "http_status": status_code,
                }
            )
            _log(
                quiet,
                f"[err] cdm={cdm_path.name} title={v_title} company={v_company} "
                f"backend_kind={kind} status={status_code} error={backend_error}",
            )
            continue

        backend_profiles = []
        if isinstance(backend_response, dict) and isinstance(backend_response.get("profiles"), list):
            backend_profiles = [p for p in backend_response.get("profiles") or [] if isinstance(p, dict)]

        if not backend_profiles:
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "error": "no_profiles_returned",
                    "http_status": status_code,
                    "backend_count": found_count,
                }
            )
            _log(quiet, f"[err] cdm={cdm_path.name} title={v_title} company={v_company} error=no_profiles_returned")
            continue

        sampled_profiles = _sample_backend_profiles(
            profiles=backend_profiles,
            sample_size=candidate_sample_size,
            sample_mode=sample_mode,
            rng=rng,
        )
        if not sampled_profiles:
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "error": "no_profiles_sampled",
                    "backend_count": found_count,
                }
            )
            continue

        backend_total_found_sum += int(found_count or 0)

        candidate_results: List[Dict[str, Any]] = []
        case_issue_counter: Counter[str] = Counter()
        case_checks: Dict[str, int] = {
            "profiles_found_count": int(found_count or 0),
            "profiles_returned_count": len(backend_profiles),
            "profiles_sampled_count": len(sampled_profiles),
            "candidate_eval_failed_count": 0,
            "execution_error_count": 0,
            "shape_fail_count": 0,
            "requirement_not_exact_count": 0,
            "passed_mismatch_count": 0,
            "comment_contract_fail_count": 0,
            "failed_items_count": 0,
        }
        case_passed = True

        for candidate in sampled_profiles:
            profile = _build_profile_from_backend_candidate(candidate)
            expected_passed = [_expected_passed_for_requirement(req, profile) for req in requirements]

            predicted: List[Dict[str, Any]] = []
            raw_sa_output = ""
            raw_error: Optional[str] = None
            try:
                predicted, raw_sa_output = sa_runner.run(requirements=requirements, profile=profile)
                _accumulate_usage(token_usage_total, sa_runner.last_usage)
            except Exception as e:
                raw_error = repr(e)

            total_candidates_evaluated += 1
            candidate_id = candidate.get("id")
            candidate_name = candidate.get("name")

            if raw_error is not None:
                case_passed = False
                case_issue_counter["execution_error"] += 1
                case_checks["execution_error_count"] += 1
                case_checks["candidate_eval_failed_count"] += 1
                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "passed": False,
                        "issues": ["execution_error"],
                        "error": raw_error,
                    }
                )
                continue

            candidate_passed, candidate_issues, candidate_details = _run_case_contract(
                requirements=requirements,
                expected_passed=expected_passed,
                predicted=predicted,
            )

            if candidate_passed:
                total_candidates_passed += 1
            else:
                case_passed = False
                case_checks["candidate_eval_failed_count"] += 1

            for issue in candidate_issues:
                case_issue_counter[issue] += 1

            candidate_checks = candidate_details["checks"]
            case_checks["shape_fail_count"] += int(candidate_checks.get("shape_fail_count", 0))
            case_checks["requirement_not_exact_count"] += int(candidate_checks.get("requirement_not_exact_count", 0))
            case_checks["passed_mismatch_count"] += int(candidate_checks.get("passed_mismatch_count", 0))
            case_checks["comment_contract_fail_count"] += int(candidate_checks.get("comment_contract_fail_count", 0))
            case_checks["failed_items_count"] += int(candidate_checks.get("failed_items_count", 0))

            candidate_rec: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "passed": candidate_passed,
                "issues": candidate_issues,
                "checks": candidate_checks,
            }

            if report_verbosity in {"standard", "full"} and (not candidate_passed or report_verbosity == "full"):
                candidate_rec["failed_items"] = candidate_details["failed_items"]

            if report_verbosity == "full":
                candidate_rec["expected_passed"] = expected_passed
                candidate_rec["predicted"] = predicted
                candidate_rec["raw_sourcing_assistant_output"] = raw_sa_output
                candidate_rec["profile_used"] = profile

            candidate_results.append(candidate_rec)

        candidates_passed = sum(1 for item in candidate_results if item.get("passed"))

        case_rec: Dict[str, Any] = {
            "cdm_file": str(cdm_path),
            "vacancy_title": v_title,
            "vacancy_company": v_company,
            "passed": case_passed,
            "issues": sorted(case_issue_counter.keys()),
            "checks": case_checks,
            "backend": {
                "count": int(found_count or 0),
                "profiles_returned": len(backend_profiles),
                "profiles_sampled": len(sampled_profiles),
                "http_status": status_code,
                "attempts": attempts,
            },
            "candidates_evaluated": len(candidate_results),
            "candidates_passed": candidates_passed,
        }

        if report_verbosity in {"standard", "full"}:
            case_rec["requirements"] = requirements
            case_rec["candidate_results"] = candidate_results

        if report_verbosity == "full":
            case_rec["backend_request_payload"] = search_payload
            if raw_req_output is not None:
                case_rec["raw_requirements_parser_output"] = raw_req_output
            if vacancy_text is not None:
                case_rec["vacancy_text_used_for_requirements"] = vacancy_text

        cases.append({k: v for k, v in case_rec.items() if v is not None})

        _log(
            quiet,
            "[case] "
            f"cdm={cdm_path.name} "
            f"title={v_title} "
            f"reqs={len(requirements)} "
            f"backend_found={int(found_count or 0)} "
            f"sampled={len(sampled_profiles)} "
            f"passed={case_passed} "
            f"issues={sorted(case_issue_counter.keys())}",
        )

    total = len(cases)
    passed_n = sum(1 for c in cases if c.get("passed"))
    failed_n = total - passed_n
    pass_rate = round((passed_n / total * 100.0), 2) if total else 0.0
    candidate_failures = total_candidates_evaluated - total_candidates_passed
    candidate_pass_rate = round((total_candidates_passed / total_candidates_evaluated * 100.0), 2) if total_candidates_evaluated else 0.0

    issue_counter = Counter(
        reason for c in cases for reason in (c.get("issues") or [])
    )
    item_issue_totals = {
        "output_shape_failed": sum(int((c.get("checks") or {}).get("shape_fail_count", 0)) for c in cases),
        "requirement_not_exact": sum(int((c.get("checks") or {}).get("requirement_not_exact_count", 0)) for c in cases),
        "passed_mismatch": sum(int((c.get("checks") or {}).get("passed_mismatch_count", 0)) for c in cases),
        "comment_contract_failed": sum(int((c.get("checks") or {}).get("comment_contract_fail_count", 0)) for c in cases),
        "failed_items_total": sum(int((c.get("checks") or {}).get("failed_items_count", 0)) for c in cases),
        "execution_error": sum(int((c.get("checks") or {}).get("execution_error_count", 0)) for c in cases),
    }

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_dir": str(cdm_dir),
        "cdm_count": cdm_count,
        "cases_count": len(cdm_paths),
        "cases_count_effective": total,
        "seed": final_seed,
        "requirements_source": requirements_source,
        "report_verbosity": report_verbosity,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "backend": {
            "base_url": base_url,
            "step3_path": step3_path,
            "timeout_s": int(timeout_s),
            "step3_retries": int(step3_retries),
            "token_in_body": bool(token_in_body),
            "candidate_pool_size": int(candidate_pool_size),
            "candidate_sample_size": int(candidate_sample_size),
            "sample_mode": sample_mode,
        },
        "token_usage_total": token_usage_total,
        "summary": {
            "total_cases": total,
            "passed": passed_n,
            "failed": failed_n,
            "pass_rate_pct": pass_rate,
            "total_candidates_evaluated": total_candidates_evaluated,
            "candidate_eval_passed": total_candidates_passed,
            "candidate_eval_failed": candidate_failures,
            "candidate_eval_pass_rate_pct": candidate_pass_rate,
            "backend_total_found_sum": backend_total_found_sum,
            "errors_count": len(errors),
            "by_issue": dict(issue_counter),
            "item_issue_totals": item_issue_totals,
        },
        "cases": cases,
        "errors": errors,
    }

    out_path = REPORTS_DIR / f"sourcing_assistant_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"total_cases={total} "
        f"passed={passed_n} "
        f"pass_rate={pass_rate:.2f}% "
        f"candidates={total_candidates_evaluated} "
        f"candidate_pass_rate={candidate_pass_rate:.2f}% "
        f"errors={len(errors)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sourcing_assistant over real backend candidates fetched for CDM vacancies with strict deterministic evaluation."
    )
    parser.add_argument(
        "--cdm-dir",
        type=str,
        default=str(DEFAULT_CDM_DIR),
        help=f"CDM fixtures dir (default: {DEFAULT_CDM_DIR})",
    )
    parser.add_argument("--cdm-count", type=int, default=None, help="Use first N CDM fixtures (sorted).")
    parser.add_argument("--cases-count", type=int, default=None, help="Sample N CDMs from selected set.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides cfg seed).")
    parser.add_argument("--prompt-id", type=str, default=None, help="Override sourcing_assistant prompt id.")
    parser.add_argument("--prompt-version", type=str, default=None, help="Override sourcing_assistant prompt version.")
    parser.add_argument(
        "--requirements-source",
        type=str,
        choices=list(REQUIREMENTS_SOURCE_VALUES),
        default="cdm_key_requirements",
        help="How to build requirements: cdm_key_requirements (default), stack_skills (deterministic) or responsibilities_parser (LLM).",
    )
    parser.add_argument(
        "--report-verbosity",
        type=str,
        choices=list(REPORT_VERBOSITY_VALUES),
        default="compact",
        help="Report detail level: compact, standard, full",
    )
    parser.add_argument("--base-url", type=str, default=os.getenv("AI_SEARCH_BASE_URL", "").strip(), help="Backend base url.")
    parser.add_argument("--step3-path", type=str, default="/site/searchBool", help="Backend search path.")
    parser.add_argument("--token", type=str, default=os.getenv("AI_SEARCH_AUTH_TOKEN", "").strip(), help="Backend auth token.")
    parser.add_argument("--timeout-s", type=int, default=30, help="Backend timeout in seconds.")
    parser.add_argument("--step3-retries", type=int, default=2, help="Backend retries.")
    parser.add_argument("--token-in-body", dest="token_in_body", action="store_true", default=True)
    parser.add_argument("--token-in-header", dest="token_in_body", action="store_false")
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="How many backend profiles to fetch per vacancy.")
    parser.add_argument("--candidate-sample-size", type=int, default=10, help="How many fetched profiles to evaluate.")
    parser.add_argument(
        "--sample-mode",
        type=str,
        choices=list(SAMPLE_MODE_VALUES),
        default="first",
        help="How to choose evaluated candidates from fetched backend profiles.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable console output.")

    args = parser.parse_args()

    run_sourcing_assistant_dataset(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        cases_count=args.cases_count,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        seed=args.seed,
        requirements_source=str(args.requirements_source),
        report_verbosity=str(args.report_verbosity),
        base_url=str(args.base_url),
        token=str(args.token),
        step3_path=str(args.step3_path),
        timeout_s=int(args.timeout_s),
        step3_retries=int(args.step3_retries),
        token_in_body=bool(args.token_in_body),
        candidate_pool_size=int(args.candidate_pool_size),
        candidate_sample_size=int(args.candidate_sample_size),
        sample_mode=str(args.sample_mode),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()

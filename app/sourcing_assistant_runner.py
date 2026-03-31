from __future__ import annotations

import argparse
import datetime
import glob
import html
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
    from app.extractor_agent_runner import BackendCfg, build_step3_payload, call_backend_search_bool
except Exception:
    from extractor_agent_runner import BackendCfg, build_step3_payload, call_backend_search_bool  # type: ignore

# Repo root: if this file is in app/, parents[1] is repo root.
ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
REPORTS_DIR = ROOT / "tests" / "reports" / "sourcing_assistant"

REPORT_VERBOSITY_VALUES = ("compact", "standard", "full")
REQUIREMENTS_SOURCE_VALUES = ("cdm_key_requirements", "stack_skills", "responsibilities_parser")
SAMPLE_MODE_VALUES = ("first", "random")


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


def _path_for_report(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _strip_html(value: Any) -> str:
    text = str(value or "")
    text = html.unescape(text)
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"(?i)</p\s*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_search_text(value: Any) -> str:
    return _strip_html(value).lower().replace("ё", "е")


def _norm_key(s: str) -> str:
    t = (s or "").strip().lower()
    t = t.replace("ё", "е")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip()
    # keep latin+cyrillic+digits, drop punctuation/spaces
    t = re.sub(r"[^a-z0-9а-я]+", "", t, flags=re.IGNORECASE)
    return t


def _norm_phrase(s: Any) -> str:
    t = _normalize_search_text(s)
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[^a-z0-9а-я]+", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _norm_compact(s: Any) -> str:
    return _norm_phrase(s).replace(" ", "")


def _norm_keep_punct(s: Any) -> str:
    t = _normalize_search_text(s)
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s*([^\w\s]+)\s*", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def _needs_compact_match(s: Any) -> bool:
    raw = _strip_html(s)
    phrase = _norm_phrase(s)
    compact = _norm_compact(s)
    if len(compact) < 2:
        return False
    return (" " in phrase) or bool(re.search(r"[^0-9A-Za-zА-Яа-яЁё]+", raw))


def _contains_norm(needle: str, hay: str) -> bool:
    if not needle or not hay:
        return False

    raw_needle = _strip_html(needle)
    n_compact = _norm_compact(needle)
    if bool(re.search(r"[^0-9A-Za-zА-Яа-яЁё]+", raw_needle)) and len(n_compact) < 2:
        n_keep = _norm_keep_punct(needle)
        h_keep = _norm_keep_punct(hay)
        return bool(n_keep) and n_keep in h_keep

    n_phrase = _norm_phrase(needle)
    h_phrase = _norm_phrase(hay)
    if n_phrase and h_phrase and f" {n_phrase} " in f" {h_phrase} ":
        return True

    if _needs_compact_match(needle):
        n = n_compact
        h = _norm_compact(hay)
        if n and h:
            return n in h

    # conservative fallback for pure alnum single-token requirements
    n = _norm_key(needle)
    h = _norm_key(hay)
    if not n or not h or len(n) < 2 or _needs_compact_match(needle):
        return False
    return re.search(rf"(?<![a-z0-9а-я]){re.escape(n_phrase)}(?![a-z0-9а-я])", f" {h_phrase} ", flags=re.IGNORECASE) is not None


_SLASH_SINGLE_TERM_REQUIREMENTS = {
    "ci/cd",
    "nlp/llm",
    "tts/stt",
}

_DIRECTED_REQUIREMENT_VARIANTS = {
    "html5": ["HTML5", "HTML"],
    "css3": ["CSS3", "CSS"],
    "работа с ии": ["работа с ИИ", "AI"],
}


def _normalize_requirement_token(req: Any) -> str:
    text = _normalize_search_text(req)
    text = re.sub(r"\s*/\s*", "/", text)
    return re.sub(r"\s+", " ", text).strip()


def _requirement_search_variants(req: str) -> List[str]:
    raw = re.sub(r"\s+", " ", _strip_html(req)).strip()
    if not raw:
        return []

    normalized = _normalize_requirement_token(raw)
    if normalized in _DIRECTED_REQUIREMENT_VARIANTS:
        return _DIRECTED_REQUIREMENT_VARIANTS[normalized]

    if "/" in raw and normalized not in _SLASH_SINGLE_TERM_REQUIREMENTS:
        parts = [part.strip() for part in re.split(r"\s*/\s*", raw) if part.strip()]
        if parts:
            return _dedupe_preserve_order(parts)

    return [raw]


def _extract_raw_vacancy_title(vacancy: Dict[str, Any]) -> str:
    raw_vacancy = str(vacancy.get("raw_vacancy") or "").strip()
    if not raw_vacancy:
        return ""
    for line in raw_vacancy.splitlines():
        title = re.sub(r"\s+", " ", line.strip())
        if title:
            return title
    return ""


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
        self.last_output_text: str = ""

    def run(self, requirements: List[str], profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        payload = {"requirements": requirements, "profile": profile}
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=json.dumps(payload, ensure_ascii=False),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
        self.last_output_text = raw
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


def _build_backend_search_payload(
    vacancy: Dict[str, Any],
    candidate_pool_size: int,
) -> Dict[str, Any]:
    base_payload: Dict[str, Any] = {
        "limit": int(candidate_pool_size),
        "offset": 0,
        "onlyWithContacts": True,
        "currentPositionTitle": True,
        "shuffle": False,
        "highlight": True,
    }

    title = re.sub(r"\s+", " ", str(vacancy.get("title") or "").strip())
    extractor_entities = vacancy.get("extractor_entities")
    if not title or not isinstance(extractor_entities, dict):
        return dict(base_payload)

    return build_step3_payload(
        extractor_json=extractor_entities,
        user_phrase=title,
        base_payload=base_payload,
        sanitize_office_geo=True,
    )


def _search_backend_candidates(
    vacancy: Dict[str, Any],
    candidate_pool_size: int,
    backend_cfg: BackendCfg,
    token: str,
) -> Dict[str, Any]:
    title = re.sub(r"\s+", " ", str(vacancy.get("title") or "").strip())
    extractor_entities = vacancy.get("extractor_entities")
    if not title:
        return {
            "kind": "no_title_in_cdm",
            "status_code": 0,
            "attempts_total": 0,
            "found_count": 0,
            "backend_error": "no_title_in_cdm",
            "backend_response": None,
            "backend_profiles": [],
            "search_strategy": None,
            "search_payload": {},
        }
    if not isinstance(extractor_entities, dict):
        return {
            "kind": "no_search_entities_in_cdm",
            "status_code": 0,
            "attempts_total": 0,
            "found_count": 0,
            "backend_error": "no_search_entities_in_cdm",
            "backend_response": None,
            "backend_profiles": [],
            "search_strategy": None,
            "search_payload": {},
        }

    payload = _build_backend_search_payload(
        vacancy=vacancy,
        candidate_pool_size=candidate_pool_size,
    )
    if not any(key in payload for key in ("positions", "skills", "keys")):
        return {
            "kind": "no_search_entities_in_cdm",
            "status_code": 0,
            "attempts_total": 0,
            "found_count": 0,
            "backend_error": "no_search_entities_in_cdm",
            "backend_response": None,
            "backend_profiles": [],
            "search_strategy": None,
            "search_payload": payload,
        }

    kind, status_code, attempts, found_count, backend_error, backend_response = call_backend_search_bool(
        backend=backend_cfg,
        token=token,
        payload=payload,
    )

    profiles: List[Dict[str, Any]] = []
    if isinstance(backend_response, dict) and isinstance(backend_response.get("profiles"), list):
        profiles = [p for p in (backend_response.get("profiles") or []) if isinstance(p, dict)]

    return {
        "kind": kind,
        "status_code": int(status_code or 0),
        "attempts_total": int(attempts or 0),
        "found_count": int(found_count or 0),
        "backend_error": backend_error,
        "backend_response": backend_response,
        "backend_profiles": profiles,
        "search_strategy": "cdm_extractor_entities",
        "search_payload": payload,
    }


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
    skills_out: List[Dict[str, Any]] = []
    for skill in (candidate.get("skills") or []):
        if not isinstance(skill, dict):
            continue
        skill_name = skill.get("skill")
        if isinstance(skill_name, str) and skill_name.strip():
            skills_out.append({"skill": re.sub(r"\s+", " ", skill_name.strip())})

    positions_out: List[Dict[str, Any]] = []
    for pos in (candidate.get("positions") or []):
        if not isinstance(pos, dict):
            continue
        company_norm = pos.get("company_norm") or {}
        categories = []
        if isinstance(company_norm, dict):
            for category in (company_norm.get("categories") or []):
                if isinstance(category, dict):
                    title = category.get("title")
                    if isinstance(title, str) and title.strip():
                        categories.append({"title": re.sub(r"\s+", " ", title.strip())})
                elif isinstance(category, str) and category.strip():
                    categories.append({"title": re.sub(r"\s+", " ", category.strip())})

        positions_norm = []
        for value in (pos.get("positions_norm") or []):
            if isinstance(value, dict):
                item: Dict[str, Any] = {}
                for key in ("title", "name", "raw_text"):
                    v = value.get(key)
                    if isinstance(v, str) and v.strip():
                        item[key] = re.sub(r"\s+", " ", v.strip())
                if item:
                    positions_norm.append(item)
            elif isinstance(value, str) and value.strip():
                positions_norm.append(re.sub(r"\s+", " ", value.strip()))

        positions_out.append(
            {
                "name": re.sub(r"\s+", " ", str(pos.get("name") or "").strip()),
                "pos": re.sub(r"\s+", " ", str(pos.get("pos") or "").strip()),
                "description": re.sub(r"\s+", " ", str(pos.get("description") or "").strip()),
                "rangeStr": re.sub(r"\s+", " ", str(pos.get("rangeStr") or "").strip()),
                "dates": pos.get("dates") or [],
                "current": bool(pos.get("current")),
                "positions_norm": positions_norm,
                "company_norm": {"categories": categories},
            }
        )

    return {
        "about": re.sub(r"\s+", " ", str(candidate.get("about") or "").strip()),
        "skills": skills_out,
        "positions": positions_out,
    }


def _flatten_profile_for_report(profile: Dict[str, Any]) -> Dict[str, Any]:
    skills_csv = ", ".join(
        re.sub(r"\s+", " ", str(skill.get("skill") or "").strip())
        for skill in (profile.get("skills") or [])
        if isinstance(skill, dict) and str(skill.get("skill") or "").strip()
    )

    positions_flat: List[str] = []
    for pos in (profile.get("positions") or []):
        if not isinstance(pos, dict):
            continue

        company_name = re.sub(r"\s+", " ", str(pos.get("name") or "").strip())
        position_name = re.sub(r"\s+", " ", str(pos.get("pos") or "").strip())
        range_str = re.sub(r"\s+", " ", str(pos.get("rangeStr") or "").strip())
        positions_norm = ", ".join(
            re.sub(r"\s+", " ", str(value).strip())
            for value in (pos.get("positions_norm") or [])
            if isinstance(value, str) and str(value).strip()
        )
        categories = ", ".join(
            re.sub(r"\s+", " ", str(cat.get("title") or "").strip())
            for cat in ((pos.get("company_norm") or {}).get("categories") or [])
            if isinstance(cat, dict) and str(cat.get("title") or "").strip()
        )
        dates = ", ".join(
            re.sub(r"\s+", " ", str(value).strip())
            for value in (pos.get("dates") or [])
            if str(value).strip()
        )

        parts = [
            f"company={company_name}",
            f"pos={position_name}",
        ]
        if dates:
            parts.append(f"dates={dates}")
        if range_str:
            parts.append(f"range={range_str}")
        if positions_norm:
            parts.append(f"positions_norm={positions_norm}")
        if categories:
            parts.append(f"categories={categories}")
        if str(pos.get("description") or "").strip():
            parts.append(f"description={_strip_html(pos.get('description'))}")
        positions_flat.append(" | ".join(part for part in parts if part and not part.endswith("=")))

    return {
        "about": _strip_html(profile.get("about")),
        "skills": skills_csv,
        "positions": positions_flat,
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

    for skill in (profile.get("skills") or []):
        if isinstance(skill, dict):
            add(skill.get("skill"))

    for pos in (profile.get("positions") or []):
        if not isinstance(pos, dict):
            continue
        for key in ("name", "pos", "description", "rangeStr"):
            add(pos.get(key))

        for value in (pos.get("dates") or []):
            add(value)

        for value in (pos.get("positions_norm") or []):
            if isinstance(value, dict):
                for key in ("title", "name", "raw_text"):
                    add(value.get(key))
            else:
                add(value)

        company_norm = pos.get("company_norm") or {}
        if isinstance(company_norm, dict):
            for category in (company_norm.get("categories") or []):
                if isinstance(category, dict):
                    add(category.get("title"))
                else:
                    add(category)

    return _dedupe_preserve_order(texts)


def _expected_passed_for_requirement(req: str, profile: Dict[str, Any]) -> int:
    """
    Детерминированная "истина" для теста по контракту prompt-а:
    требование ищется только в profile.about, skills[].skill и разрешённых полях positions
    с нормализацией регистра/пробелов/пунктуации.

    Важно:
    - для большинства требований с "/" judge следует текущему prompt-контракту и
      трактует их как OR между частями;
    - "CI/CD", "NLP/LLM", "TTS/STT" остаются едиными терминами;
    - для нескольких продуктово-согласованных кейсов поддерживаются направленные
      варианты поиска, например "HTML5" -> "HTML".
    """
    if not req:
        return 0

    variants = _requirement_search_variants(req)
    if not variants:
        return 0

    for text in _profile_search_texts(profile):
        for variant in variants:
            if _contains_norm(variant, text):
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
    }

    if len(predicted) != len(requirements):
        issues.append("length_mismatch")

    n = min(len(predicted), len(requirements))
    for i in range(n):
        item = predicted[i]
        exp_req = requirements[i]
        exp_pass = expected_passed[i]
        actual_comment = item.get("comment") if isinstance(item, dict) else None

        if not isinstance(item, dict):
            checks["shape_fail_count"] += 1
            failed_items.append(
                {
                    "index": i,
                    "expected_requirement": exp_req,
                    "actual_requirement": None,
                    "expected_passed": exp_pass,
                    "actual_passed": None,
                    "actual_comment": None,
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

        if item_issues:
            failed_items.append(
                {
                    "index": i,
                    "expected_requirement": exp_req,
                    "actual_requirement": item.get("requirement"),
                    "expected_passed": exp_pass,
                    "actual_passed": item.get("passed"),
                    "actual_comment": actual_comment,
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

    return len(issues) == 0, issues, {"checks": checks, "failed_items": failed_items}


def _compact_failed_items(failed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in failed_items:
        compact: Dict[str, Any] = {
            "requirement": item.get("expected_requirement"),
            "expected_passed": item.get("expected_passed"),
            "actual_passed": item.get("actual_passed"),
            "issues": item.get("issues") or [],
        }
        actual_requirement = item.get("actual_requirement")
        if actual_requirement is not None and actual_requirement != item.get("expected_requirement"):
            compact["actual_requirement"] = actual_requirement
        actual_comment = item.get("actual_comment")
        if isinstance(actual_comment, str):
            compact["actual_comment"] = actual_comment
        out.append(compact)
    return out


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
    execution_errors: List[Dict[str, Any]] = []
    total_candidates_evaluated = 0
    total_candidates_passed = 0

    for cdm_path in cdm_paths:
        cdm = load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}
        v_title = vacancy.get("title")
        v_company = vacancy.get("company_name") or vacancy.get("companyName")
        raw_vacancy_title = _extract_raw_vacancy_title(vacancy) or None

        # 1) requirements
        if requirements_source == "responsibilities_parser" and resp_parser is not None:
            vacancy_text = builder.build(vacancy)
            try:
                requirements, _ = resp_parser.extract(vacancy_text)
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
            execution_errors.append(
                {
                    "cdm_file": _path_for_report(cdm_path),
                    "structured_vacancy_title": v_title,
                    "structured_vacancy_company": v_company,
                    "raw_vacancy_title": raw_vacancy_title,
                    "error": "no_requirements_generated",
                }
            )
            continue
        # 2) backend search for real candidates using extractor entities stored in CDM
        search_result = _search_backend_candidates(
            vacancy=vacancy,
            candidate_pool_size=candidate_pool_size,
            backend_cfg=backend_cfg,
            token=token,
        )
        kind = str(search_result.get("kind") or "")
        status_code = int(search_result.get("status_code") or 0)
        found_count = int(search_result.get("found_count") or 0)
        backend_error = search_result.get("backend_error")
        backend_profiles = [p for p in (search_result.get("backend_profiles") or []) if isinstance(p, dict)]
        search_strategy = search_result.get("search_strategy")
        search_payload = search_result.get("search_payload")

        if kind in {"no_title_in_cdm", "no_search_entities_in_cdm"}:
            case_rec: Dict[str, Any] = {
                "cdm_file": _path_for_report(cdm_path),
                "structured_vacancy_title": v_title,
                "structured_vacancy_company": v_company,
                "raw_vacancy_title": raw_vacancy_title,
                "key_requirements": requirements,
                "passed": False,
                "search_status": kind,
                "contract_issues": [],
                "backend_found_count": 0,
                "backend_profiles_evaluated": 0,
                "backend_candidates_passed": 0,
                "search_strategy": None,
            }
            if report_verbosity in {"standard", "full"}:
                case_rec["candidate_results"] = []
            if report_verbosity == "full":
                case_rec["backend_search_payload_used"] = search_payload
            cases.append(case_rec)
            _log(quiet, f"[case] cdm={cdm_path.name} title={v_title} search_status={kind}")
            continue

        if kind != "success":
            execution_errors.append(
                {
                    "cdm_file": _path_for_report(cdm_path),
                    "structured_vacancy_title": v_title,
                    "structured_vacancy_company": v_company,
                    "raw_vacancy_title": raw_vacancy_title,
                    "error": backend_error or kind,
                    "backend_kind": kind,
                    "http_status": status_code,
                    "backend_found_count": found_count,
                    "search_strategy": search_strategy,
                    "backend_search_payload_used": search_payload,
                }
            )
            case_rec = {
                "cdm_file": _path_for_report(cdm_path),
                "structured_vacancy_title": v_title,
                "structured_vacancy_company": v_company,
                "raw_vacancy_title": raw_vacancy_title,
                "key_requirements": requirements,
                "passed": False,
                "search_status": "backend_error",
                "contract_issues": [],
                "backend_found_count": int(found_count or 0),
                "backend_profiles_evaluated": 0,
                "backend_candidates_passed": 0,
                "search_strategy": search_strategy,
                "backend_error": backend_error or kind,
                "http_status": status_code,
            }
            if report_verbosity in {"standard", "full"}:
                case_rec["candidate_results"] = []
            if report_verbosity == "full":
                case_rec["backend_search_payload_used"] = search_payload
            cases.append(case_rec)
            _log(
                quiet,
                f"[err] cdm={cdm_path.name} title={v_title} company={v_company} "
                f"backend_kind={kind} status={status_code} error={backend_error}",
            )
            continue

        if found_count <= 0 and not backend_profiles:
            case_rec = {
                "cdm_file": _path_for_report(cdm_path),
                "structured_vacancy_title": v_title,
                "structured_vacancy_company": v_company,
                "raw_vacancy_title": raw_vacancy_title,
                "key_requirements": requirements,
                "passed": False,
                "search_status": "no_candidates_found",
                "contract_issues": [],
                "backend_found_count": int(found_count or 0),
                "backend_profiles_evaluated": 0,
                "backend_candidates_passed": 0,
                "search_strategy": search_strategy,
                "http_status": status_code,
            }
            if report_verbosity in {"standard", "full"}:
                case_rec["candidate_results"] = []
            if report_verbosity == "full":
                case_rec["backend_search_payload_used"] = search_payload
            cases.append(case_rec)
            _log(
                quiet,
                f"[case] cdm={cdm_path.name} title={v_title} backend_found=0 search_status=no_candidates_found",
            )
            continue

        if not backend_profiles:
            execution_errors.append(
                {
                    "cdm_file": _path_for_report(cdm_path),
                    "structured_vacancy_title": v_title,
                    "structured_vacancy_company": v_company,
                    "raw_vacancy_title": raw_vacancy_title,
                    "error": "profiles_missing_in_backend_response",
                    "http_status": status_code,
                    "backend_found_count": found_count,
                    "search_strategy": search_strategy,
                    "backend_search_payload_used": search_payload,
                }
            )
            case_rec = {
                "cdm_file": _path_for_report(cdm_path),
                "structured_vacancy_title": v_title,
                "structured_vacancy_company": v_company,
                "raw_vacancy_title": raw_vacancy_title,
                "key_requirements": requirements,
                "passed": False,
                "search_status": "backend_error",
                "contract_issues": [],
                "backend_found_count": int(found_count or 0),
                "backend_profiles_evaluated": 0,
                "backend_candidates_passed": 0,
                "search_strategy": search_strategy,
                "backend_error": "profiles_missing_in_backend_response",
                "http_status": status_code,
            }
            if report_verbosity in {"standard", "full"}:
                case_rec["candidate_results"] = []
            if report_verbosity == "full":
                case_rec["backend_search_payload_used"] = search_payload
            cases.append(case_rec)
            _log(
                quiet,
                f"[err] cdm={cdm_path.name} title={v_title} status={status_code} error=profiles_missing_in_backend_response",
            )
            continue

        sampled_profiles = _sample_backend_profiles(
            profiles=backend_profiles,
            sample_size=candidate_sample_size,
            sample_mode=sample_mode,
            rng=rng,
        )
        if not sampled_profiles:
            execution_errors.append(
                {
                    "cdm_file": _path_for_report(cdm_path),
                    "structured_vacancy_title": v_title,
                    "structured_vacancy_company": v_company,
                    "raw_vacancy_title": raw_vacancy_title,
                    "error": "no_profiles_sampled",
                    "backend_found_count": found_count,
                    "search_strategy": search_strategy,
                    "backend_search_payload_used": search_payload,
                }
            )
            case_rec = {
                "cdm_file": _path_for_report(cdm_path),
                "structured_vacancy_title": v_title,
                "structured_vacancy_company": v_company,
                "raw_vacancy_title": raw_vacancy_title,
                "key_requirements": requirements,
                "passed": False,
                "search_status": "sampling_error",
                "contract_issues": [],
                "backend_found_count": int(found_count or 0),
                "backend_profiles_evaluated": 0,
                "backend_candidates_passed": 0,
                "search_strategy": search_strategy,
            }
            if report_verbosity in {"standard", "full"}:
                case_rec["candidate_results"] = []
            if report_verbosity == "full":
                case_rec["backend_search_payload_used"] = search_payload
            cases.append(case_rec)
            continue

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
            "failed_items_count": 0,
        }
        case_passed = True

        for candidate in sampled_profiles:
            profile = _build_profile_from_backend_candidate(candidate)
            profile_flat = _flatten_profile_for_report(profile)
            expected_passed = [_expected_passed_for_requirement(req, profile) for req in requirements]

            predicted: List[Dict[str, Any]] = []
            raw_output = ""
            raw_error: Optional[str] = None
            try:
                predicted, raw_output = sa_runner.run(requirements=requirements, profile=profile)
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
                raw_output = sa_runner.last_output_text or ""
                execution_errors.append(
                    {
                        "cdm_file": _path_for_report(cdm_path),
                        "structured_vacancy_title": v_title,
                        "structured_vacancy_company": v_company,
                        "raw_vacancy_title": raw_vacancy_title,
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "error": "sourcing_assistant_execution_error",
                        "details": raw_error,
                        "prompt_output_raw": raw_output,
                    }
                )
                candidate_rec = {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "passed": False,
                    "contract_issues": ["execution_error"],
                }
                if report_verbosity == "full":
                    candidate_rec["prompt_input_flat"] = profile_flat
                    candidate_rec["prompt_output_raw"] = raw_output
                candidate_results.append(candidate_rec)
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
            case_checks["failed_items_count"] += int(candidate_checks.get("failed_items_count", 0))

            candidate_rec: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "passed": candidate_passed,
                "contract_issues": candidate_issues,
            }

            if (not candidate_passed) or report_verbosity in {"standard", "full"}:
                failed_items = candidate_details["failed_items"]
                if failed_items:
                    candidate_rec["failed_requirements"] = _compact_failed_items(failed_items)

            if report_verbosity == "full":
                candidate_rec["prompt_input_flat"] = profile_flat
                candidate_rec["prompt_output"] = predicted

            candidate_results.append(candidate_rec)

        candidates_passed = sum(1 for item in candidate_results if item.get("passed"))

        case_rec: Dict[str, Any] = {
            "cdm_file": _path_for_report(cdm_path),
            "structured_vacancy_title": v_title,
            "structured_vacancy_company": v_company,
            "raw_vacancy_title": raw_vacancy_title,
            "key_requirements": requirements,
            "passed": case_passed,
            "search_status": "evaluated",
            "contract_issues": sorted(case_issue_counter.keys()),
            "backend_found_count": int(found_count or 0),
            "backend_profiles_evaluated": len(candidate_results),
            "backend_candidates_passed": candidates_passed,
            "search_strategy": search_strategy,
        }

        if report_verbosity in {"standard", "full"}:
            case_rec["candidate_results"] = candidate_results

        if report_verbosity == "full":
            case_rec["backend_search_payload_used"] = search_payload

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
            f"issues={sorted(case_issue_counter.keys())} "
            f"search_strategy={search_strategy}",
        )

    total = len(cases)
    search_status_counter = Counter(str(c.get("search_status") or "unknown") for c in cases)
    evaluated_cases = [c for c in cases if c.get("search_status") == "evaluated"]
    evaluated_total = len(evaluated_cases)
    passed_n = sum(1 for c in evaluated_cases if c.get("passed"))
    failed_n = evaluated_total - passed_n
    pass_rate = round((passed_n / evaluated_total * 100.0), 2) if evaluated_total else 0.0
    candidate_failures = total_candidates_evaluated - total_candidates_passed
    candidate_pass_rate = round((total_candidates_passed / total_candidates_evaluated * 100.0), 2) if total_candidates_evaluated else 0.0

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_dir": _path_for_report(cdm_dir),
        "cdm_count": cdm_count,
        "cases_count": len(cdm_paths),
        "cases_count_requested": cases_count,
        "seed": final_seed,
        "report_verbosity": report_verbosity,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "backend": {
            "step3_path": step3_path,
            "requirements_source": requirements_source,
            "candidate_pool_size": int(candidate_pool_size),
            "candidate_sample_size": int(candidate_sample_size),
            "sample_mode": sample_mode,
        },
        "token_usage_total": token_usage_total,
        "summary": {
            "total_cases": total,
            "evaluated_cases": evaluated_total,
            "cases_without_candidates": int(search_status_counter.get("no_candidates_found", 0)),
            "cases_with_backend_error": int(search_status_counter.get("backend_error", 0)),
            "cases_without_title": int(search_status_counter.get("no_title_in_cdm", 0)),
            "passed": passed_n,
            "failed": failed_n,
            "pass_rate_pct": pass_rate,
            "total_candidates_evaluated": total_candidates_evaluated,
            "candidate_eval_passed": total_candidates_passed,
            "candidate_eval_failed": candidate_failures,
            "candidate_eval_pass_rate_pct": candidate_pass_rate,
            "execution_errors_count": len(execution_errors),
        },
        "cases": cases,
        "execution_errors": execution_errors,
    }

    out_path = REPORTS_DIR / f"sourcing_assistant_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"total_cases={total} "
        f"evaluated_cases={evaluated_total} "
        f"passed={passed_n} "
        f"pass_rate={pass_rate:.2f}% "
        f"candidates={total_candidates_evaluated} "
        f"candidate_pass_rate={candidate_pass_rate:.2f}% "
        f"errors={len(execution_errors)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sourcing_assistant over real backend candidates fetched from backend payloads built from CDM extractor entities."
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
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="How many backend profiles to fetch per vacancy from backend search.")
    parser.add_argument("--candidate-sample-size", type=int, default=10, help="How many fetched profiles to evaluate with sourcing_assistant.")
    parser.add_argument(
        "--sample-mode",
        type=str,
        choices=list(SAMPLE_MODE_VALUES),
        default="first",
        help="How to choose evaluated candidates from fetched backend profiles returned by backend search.",
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

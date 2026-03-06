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

# Repo root: if this file is in app/, parents[1] is repo root.
ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
REPORTS_DIR = ROOT / "tests" / "reports" / "sourcing_assistant"

REPORT_VERBOSITY_VALUES = ("compact", "standard", "full")
REQUIREMENTS_SOURCE_VALUES = ("stack_skills", "responsibilities_parser")


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


def _requirements_from_stack_skills(vacancy: Dict[str, Any]) -> List[str]:
    stack = _split_list_like(vacancy.get("vacancy_stack") or "")
    skills = _split_list_like(vacancy.get("vacancy_skills") or "")
    merged = _dedupe_preserve_order(stack + skills)
    return merged[:5]


def _expected_passed_for_requirement(req: str, profile: Dict[str, Any]) -> int:
    """
    Детерминированная "истина" для теста:
    passed=1 если requirement встречается (с нормализацией) в about/skills/positions/categories.
    """
    if not req:
        return 0

    about = profile.get("about") or ""
    if isinstance(about, str) and _contains_norm(req, about):
        return 1

    for s in (profile.get("skills") or []):
        if isinstance(s, dict):
            sk = s.get("skill") or ""
            if isinstance(sk, str) and _contains_norm(req, sk):
                return 1

    for p in (profile.get("positions") or []):
        if not isinstance(p, dict):
            continue
        for k in ("pos", "description", "name"):
            v = p.get(k) or ""
            if isinstance(v, str) and _contains_norm(req, v):
                return 1

        # categories titles
        cn = p.get("company_norm") or {}
        if isinstance(cn, dict):
            cats = cn.get("categories") or []
            if isinstance(cats, list):
                for c in cats:
                    if isinstance(c, dict):
                        t = c.get("title") or ""
                        if isinstance(t, str) and _contains_norm(req, t):
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


def run_sourcing_assistant_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    cases_count: Optional[int],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    seed: Optional[int],
    requirements_source: str,
    report_verbosity: str,
    quiet: bool,
) -> pathlib.Path:
    ensure_dirs()

    if report_verbosity not in REPORT_VERBOSITY_VALUES:
        raise ValueError(f"--report-verbosity must be one of: {', '.join(REPORT_VERBOSITY_VALUES)}")
    if requirements_source not in REQUIREMENTS_SOURCE_VALUES:
        raise ValueError(f"--requirements-source must be one of: {', '.join(REQUIREMENTS_SOURCE_VALUES)}")

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
    if cfg_req_source in REQUIREMENTS_SOURCE_VALUES and requirements_source == "stack_skills":
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
        f"report_verbosity={report_verbosity}",
    )
    _log(quiet, f"[init] sourcing_assistant prompt_id={final_pid} prompt_version={final_pver}")
    if resp_parser:
        _log(quiet, f"[init] responsibilities_parser prompt_id={resp_prompt_id} prompt_version={resp_prompt_version}")

    sa_runner = SourcingAssistantRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for cdm_path in cdm_paths:
        cdm = load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}
        candidate = cdm.get("candidate") or {}

        v_title = vacancy.get("title")
        v_company = vacancy.get("company_name")
        c_name = candidate.get("candidate_name")

        # 1) requirements
        raw_req_output = None
        vacancy_text = None
        if requirements_source == "responsibilities_parser" and resp_parser is not None:
            vacancy_text = builder.build(vacancy)
            try:
                requirements, raw_req_output = resp_parser.extract(vacancy_text)
                _accumulate_usage(token_usage_total, resp_parser.last_usage)
            except Exception:
                # fallback to deterministic
                requirements = _requirements_from_stack_skills(vacancy)
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
                    "candidate_name": c_name,
                    "error": "no_requirements_generated",
                }
            )
            continue

        # 2) profile for sourcing_assistant
        profile = _build_profile_from_cdm_candidate(candidate)

        # 3) expected passed (deterministic truth)
        expected_passed = [_expected_passed_for_requirement(req, profile) for req in requirements]

        # 4) run sourcing_assistant
        predicted: List[Dict[str, Any]] = []
        raw_sa_output = ""
        raw_error: Optional[str] = None
        try:
            predicted, raw_sa_output = sa_runner.run(requirements=requirements, profile=profile)
            _accumulate_usage(token_usage_total, sa_runner.last_usage)
        except Exception as e:
            raw_error = repr(e)

        if raw_error is not None:
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "candidate_name": c_name,
                    "error": raw_error,
                }
            )
            _log(quiet, f"[err] cdm={cdm_path.name} title={v_title} company={v_company} error={raw_error}")
            continue

        strict_passed, strict_fail_reasons, strict_details = _run_case_strict(
            requirements=requirements,
            expected_passed=expected_passed,
            predicted=predicted,
        )

        case_rec: Dict[str, Any] = {
            "cdm_file": str(cdm_path),
            "vacancy_title": v_title,
            "vacancy_company": v_company,
            "candidate_name": c_name,
            "requirements": requirements if report_verbosity in {"standard", "full"} else None,
            "expected_passed": expected_passed if report_verbosity in {"standard", "full"} else None,
            "passed": strict_passed,
            "strict_fail_reasons": strict_fail_reasons,
        }

        if report_verbosity in {"standard", "full"}:
            case_rec["predicted"] = predicted
            case_rec["strict_details"] = strict_details
            case_rec["raw_sourcing_assistant_output"] = raw_sa_output
            if raw_req_output is not None:
                case_rec["raw_requirements_parser_output"] = raw_req_output

        if report_verbosity == "full":
            case_rec["profile_used"] = profile
            if vacancy_text is not None:
                case_rec["vacancy_text_used_for_requirements"] = vacancy_text

        cases.append({k: v for k, v in case_rec.items() if v is not None})

        _log(
            quiet,
            "[case] "
            f"cdm={cdm_path.name} "
            f"title={v_title} "
            f"candidate={c_name} "
            f"reqs={len(requirements)} "
            f"passed={strict_passed} "
            f"reasons={strict_fail_reasons}",
        )

    total = len(cases)
    passed_n = sum(1 for c in cases if c.get("passed"))
    pass_rate = round((passed_n / total * 100.0), 2) if total else 0.0

    fail_reasons_counter = Counter(
        reason for c in cases for reason in (c.get("strict_fail_reasons") or [])
    )

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
        "token_usage_total": token_usage_total,
        "summary": {
            "total_cases": total,
            "passed": passed_n,
            "pass_rate_pct": pass_rate,
            "errors_count": len(errors),
            "fail_reasons": dict(fail_reasons_counter),
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
        f"errors={len(errors)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sourcing_assistant over CDM fixtures with strict deterministic evaluation (no precision/recall)."
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
        default="stack_skills",
        help="How to build requirements: stack_skills (deterministic) or responsibilities_parser (LLM).",
    )
    parser.add_argument(
        "--report-verbosity",
        type=str,
        choices=list(REPORT_VERBOSITY_VALUES),
        default="compact",
        help="Report detail level: compact, standard, full",
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
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()
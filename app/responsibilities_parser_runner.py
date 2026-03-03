# vacancy_keywords_runner.py
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
REPORTS_DIR = ROOT / "tests" / "reports" / "responsibilities_parser"

DEFAULT_VACANCY_GEN_MODEL = "gpt-4.1-mini"
VACANCY_GEN_MAX_RETRIES = 1
DEFAULT_PASS_SCORE_THRESHOLD = 70.0
REPORT_VERBOSITY_VALUES = ("compact", "standard", "full")
SCORE_WEIGHTS = {
    "precision": 0.45,
    "recall": 0.35,
    "grounding": 0.20,
}


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


def _resolve_prompt_from_cfg(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # Optional in tests/tools/model.yaml:
    # responsibilities_parser:
    #   prompt_id: pmpt_...
    #   prompt_version: ...
    #   seed: 1234
    block = cfg.get("responsibilities_parser") or {}
    pid = block.get("prompt_id")
    pver = block.get("prompt_version")
    seed = block.get("seed")
    return (str(pid) if pid else None, str(pver) if pver else None, int(seed) if seed is not None else None)


def _resolve_vacancy_gen_model_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    # Optional in tests/tools/model.yaml:
    # responsibilities_parser:
    #   vacancy_gen_model: gpt-4.1-mini
    block = cfg.get("responsibilities_parser") or {}
    m = block.get("vacancy_gen_model")
    return str(m) if m else None


def _split_list_like(s: Optional[str]) -> List[str]:
    """
    Splits a comma/semicolon/newline-separated skills string into a clean list.
    Keeps original casing in returned values.
    """
    if not s:
        return []
    parts = re.split(r"[,\n;|]+", str(s))
    out: List[str] = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        t = re.sub(r"\s+", " ", t)
        out.append(t)
    return out


def _norm_key(s: str) -> str:
    """
    Normalization for matching. Makes CI/CD and cicd comparable, ignores punctuation.
    """
    t = (s or "").strip().lower()
    t = t.replace("ё", "е")
    # unify some common variants
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip()
    # remove all non-alnum (keep cyrillic/latin digits too, but digits handled separately)
    t = re.sub(r"[^a-z0-9а-я]+", "", t, flags=re.IGNORECASE)
    return t


def _contains_in_text(item: str, text: str) -> bool:
    """
    Checks if item appears in text (case-insensitive) with light normalization.
    """
    if not item or not text:
        return False
    it = item.strip().lower()
    tx = text.lower()
    if it in tx:
        return True

    # Try also normalized matching without punctuation/spaces
    it2 = _norm_key(item)
    tx2 = _norm_key(text)
    if it2 and it2 in tx2:
        return True
    return False


def _parse_json_array_strict(raw: str) -> List[str]:
    """
    Expects a JSON array of strings. Raises ValueError if invalid.
    """
    if not raw:
        raise ValueError("empty output")

    # Some models may wrap JSON in text. Try to extract the first [...] block.
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

    out: List[str] = []
    for i, v in enumerate(obj):
        if not isinstance(v, str):
            raise ValueError(f"item[{i}] is not a string: {raw!r}")
        out.append(v.strip())
    return out


def _validate_item_format(item: str) -> Tuple[bool, List[str]]:
    """
    Enforces prompt format rules:
    - 1-3 words (spaces)
    - no digits
    - no commas/semicolons
    - looks like a term (no trailing punctuation)
    """
    errors: List[str] = []
    t = (item or "").strip()
    if not t:
        return False, ["empty item"]

    if re.search(r"\d", t):
        errors.append("contains digits")

    if "," in t or ";" in t:
        errors.append("contains comma/semicolon")

    # Word count by spaces (TTS/STT is one word, Function Calling is two)
    words = [w for w in re.split(r"\s+", t) if w]
    if not (1 <= len(words) <= 3):
        errors.append(f"word_count={len(words)} (expected 1..3)")

    # no full sentences
    if len(t) > 60:
        errors.append("too long (>60 chars)")

    # discourage verbs by simple heuristic (not strict)
    # (kept soft: we do not fail on it, only warn)
    # You can promote to a hard error if needed:
    # if re.search(r"\b(делать|уметь|понимать|знать|настраивать|разрабатывать)\b", t.lower()):
    #     errors.append("looks like a phrase with a verb")

    return len(errors) == 0, errors


def _match_to_expected(pred: str, expected: List[str]) -> Optional[str]:
    """
    Returns matched expected skill if found, else None.
    Matching is done via normalized key equality or substring containment.
    """
    if not pred:
        return None
    pk = _norm_key(pred)
    if not pk:
        return None

    expected_keys = {e: _norm_key(e) for e in expected}
    for e, ek in expected_keys.items():
        if pk == ek and ek:
            return e

    # fallback: sometimes pred is a shorter part of a longer expected item
    for e, ek in expected_keys.items():
        if pk and ek and (pk in ek or ek in pk):
            return e

    return None


class VacancyTextBuilder:
    """
    Deterministic generator of a "detailed vacancy text" from structured CDM vacancy fields.
    This keeps the dataset stable and ensures required terms appear explicitly.
    """

    def build(self, vacancy: Dict[str, Any]) -> str:
        title = vacancy.get("title") or "Вакансия"
        company = vacancy.get("company_name") or "Компания"
        industry = vacancy.get("company_industry") or ""
        location = vacancy.get("location") or ""
        work_format = vacancy.get("work_format") or ""
        salary_from = vacancy.get("salary_range_from")
        salary_to = vacancy.get("salary_range_to")
        company_desc = vacancy.get("company_description") or vacancy.get("firm_description") or ""
        responsibilities = vacancy.get("responsibilities") or ""
        stack = vacancy.get("vacancy_stack") or vacancy.get("stack") or ""
        skills = vacancy.get("vacancy_skills") or ""
        questions = vacancy.get("questions") or ""

        salary_line = ""
        if salary_from is not None or salary_to is not None:
            salary_line = f"Зарплата: {salary_from or ''} - {salary_to or ''}".strip(" -")

        resp_items = _split_list_like(responsibilities)
        resp_block = ""
        if resp_items:
            resp_block = "Задачи:\n" + "\n".join(f"- {x}" for x in resp_items)

        # Make skills explicit in a "requirements" section.
        stack_items = _split_list_like(stack)
        skills_items = _split_list_like(skills)

        req_lines: List[str] = []
        if stack_items:
            req_lines.append("Обязательные технологии/инструменты:")
            req_lines.extend(f"- {x}" for x in stack_items)
        if skills_items:
            req_lines.append("Ключевые навыки:")
            req_lines.extend(f"- {x}" for x in skills_items)

        req_block = ""
        if req_lines:
            req_block = "Требования:\n" + "\n".join(req_lines)

        meta_lines = [
            f"Должность: {title}",
            f"Компания: {company}",
        ]
        if industry:
            meta_lines.append(f"Индустрия: {industry}")
        if location:
            meta_lines.append(f"Локация: {location}")
        if work_format:
            meta_lines.append(f"Формат: {work_format}")
        if salary_line:
            meta_lines.append(salary_line)

        about = ""
        if company_desc:
            about = f"О компании:\n{company_desc}"

        q_block = ""
        if questions:
            q_block = f"Вопросы на скрининг:\n{questions}".strip()

        blocks = [("\n".join(meta_lines)).strip(), about.strip(), resp_block.strip(), req_block.strip(), q_block.strip()]
        text = "\n\n".join([b for b in blocks if b]).strip()
        return text


class VacancyTextSynthesizerLLM:
    """
    Optional: expands structured vacancy fields into a more realistic job posting text using an LLM.
    Important: to keep evaluation meaningful, we force the model to include stack/skills terms verbatim.
    """

    def __init__(self, model: str, seed: Optional[int]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.seed = seed
        self.last_usage: Any = None

    def synthesize(self, vacancy: Dict[str, Any]) -> str:
        title = vacancy.get("title")
        company = vacancy.get("company_name")
        company_desc = vacancy.get("company_description") or vacancy.get("firm_description")
        industry = vacancy.get("company_industry")
        location = vacancy.get("location")
        work_format = vacancy.get("work_format")
        salary_from = vacancy.get("salary_range_from")
        salary_to = vacancy.get("salary_range_to")
        responsibilities = vacancy.get("responsibilities")
        stack_items = _split_list_like(vacancy.get("vacancy_stack") or vacancy.get("stack") or "")
        skills_items = _split_list_like(vacancy.get("vacancy_skills") or "")
        questions = vacancy.get("questions") or ""

        ctx = {
            "title": title,
            "company_name": company,
            "company_description": company_desc,
            "industry": industry,
            "location": location,
            "work_format": work_format,
            "salary_range_from": salary_from,
            "salary_range_to": salary_to,
            "responsibilities": responsibilities,
            "stack_terms_verbatim": stack_items,
            "skills_terms_verbatim": skills_items,
            "screening_questions": questions,
        }

        instruction = (
            "Сгенерируй подробный текст вакансии на русском языке.\n"
            "Важно: термины из списков stack_terms_verbatim и skills_terms_verbatim должны быть вставлены ВЕРБАТИМ "
            "(точно как в списках) в раздел 'Требования' (списком).\n"
            "Не выдумывай новые технологии, навыки и требования.\n"
            "Структура:\n"
            "1) Кратко о компании\n"
            "2) Чем предстоит заниматься\n"
            "3) Требования (обязательно отдельным разделом со списком)\n"
            "4) Вопросы на скрининг (если есть)\n"
            "Верни только текст вакансии, без JSON и без markdown.\n"
        )

        resp = self.client.responses.create(
            model=self.model,
            input=instruction + "\n\nCONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False),
            temperature=0.0,
        )
        self.last_usage = getattr(resp, "usage", None)
        text = (getattr(resp, "output_text", "") or "").strip()
        if not text:
            raise ValueError("vacancy generator returned empty text")
        return text


class VacancyKeywordsRunner:
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
        items = _parse_json_array_strict(raw)
        return items, raw


def _case_metrics(
    predicted: List[str],
    vacancy_text: str,
    expected_stack: List[str],
    expected_skills: List[str],
) -> Dict[str, Any]:
    expected_all = list(dict.fromkeys(expected_stack + expected_skills))  # stable unique
    matched_stack: List[str] = []
    matched_skills: List[str] = []
    matched_any: List[str] = []

    format_ok = True
    format_errors: Dict[str, List[str]] = {}
    not_in_text: List[str] = []

    # Validate predicted list size
    list_ok = True
    list_errors: List[str] = []
    if not (1 <= len(predicted) <= 5):
        list_ok = False
        list_errors.append(f"len={len(predicted)} (expected 1..5)")

    # Per-item checks
    for it in predicted:
        ok, errs = _validate_item_format(it)
        if not ok:
            format_ok = False
            format_errors[it] = errs

        if not _contains_in_text(it, vacancy_text):
            not_in_text.append(it)

        m_stack = _match_to_expected(it, expected_stack)
        if m_stack:
            matched_stack.append(m_stack)

        m_skills = _match_to_expected(it, expected_skills)
        if m_skills:
            matched_skills.append(m_skills)

        m_any = _match_to_expected(it, expected_all)
        if m_any:
            matched_any.append(m_any)

    # Unique matches
    matched_stack_u = list(dict.fromkeys(matched_stack))
    matched_any_u = list(dict.fromkeys(matched_any))
    in_text_ok = len(not_in_text) == 0

    precision = 0.0
    if predicted:
        precision = round(len(matched_any_u) / len(predicted) * 100.0, 2)

    expected_total_count = len(expected_all)
    recall = 0.0
    if expected_total_count:
        recall = len(matched_any_u) / expected_total_count

    not_in_text_count = len(not_in_text)
    grounding = 1.0 - (not_in_text_count / max(1, len(predicted)))
    score_norm = (
        SCORE_WEIGHTS["precision"] * (precision / 100.0)
        + SCORE_WEIGHTS["recall"] * recall
        + SCORE_WEIGHTS["grounding"] * grounding
    )
    final_score = round(score_norm * 100.0, 2)

    if final_score >= 85.0:
        quality_level = "strong_pass"
    elif final_score >= 70.0:
        quality_level = "pass"
    elif final_score >= 55.0:
        quality_level = "borderline"
    else:
        quality_level = "fail"

    return {
        "list_ok": list_ok,
        "list_errors": list_errors,
        "format_ok": format_ok,
        "format_errors": format_errors,
        "in_text_ok": in_text_ok,
        "not_in_text": not_in_text,
        "not_in_text_count": not_in_text_count,
        "expected_stack": expected_stack,
        "expected_skills": expected_skills,
        "expected_total": expected_all,
        "expected_total_count": expected_total_count,
        "matched_stack": matched_stack_u,
        "matched_total": matched_any_u,
        "stack_matches_count": len(matched_stack_u),
        "total_matches_count": len(matched_any_u),
        "precision_pct": precision,
        "recall_pct": round(recall * 100.0, 2),
        "grounding_pct": round(grounding * 100.0, 2),
        "final_score_pct": final_score,
        "quality_level": quality_level,
    }


def _strict_fail_reasons(
    metrics: Dict[str, Any],
    min_total_matches: int,
    min_stack_matches: int,
    require_all_in_text: bool,
) -> List[str]:
    reasons: List[str] = []
    if not metrics["list_ok"]:
        reasons.append("list_constraints_failed")
    if not metrics["format_ok"]:
        reasons.append("item_format_failed")
    if require_all_in_text and not metrics["in_text_ok"]:
        reasons.append("predicted_not_found_in_text")
    if metrics["total_matches_count"] < min_total_matches:
        reasons.append(f"total_matches<{min_total_matches}")
    if metrics["stack_matches_count"] < min_stack_matches:
        reasons.append(f"stack_matches<{min_stack_matches}")
    return reasons


def _project_metrics_for_report(metrics: Dict[str, Any], report_verbosity: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "list_ok": metrics["list_ok"],
        "format_ok": metrics["format_ok"],
        "in_text_ok": metrics["in_text_ok"],
        "not_in_text": metrics["not_in_text"],
        "not_in_text_count": metrics["not_in_text_count"],
        "stack_matches_count": metrics["stack_matches_count"],
        "total_matches_count": metrics["total_matches_count"],
        "precision_pct": metrics["precision_pct"],
        "recall_pct": metrics["recall_pct"],
        "grounding_pct": metrics["grounding_pct"],
        "final_score_pct": metrics["final_score_pct"],
        "quality_level": metrics["quality_level"],
    }

    if not metrics["list_ok"]:
        out["list_errors"] = metrics["list_errors"]
    if not metrics["format_ok"]:
        out["format_errors"] = metrics["format_errors"]

    if report_verbosity in {"standard", "full"}:
        out["matched_stack"] = metrics["matched_stack"]
        out["matched_total"] = metrics["matched_total"]

    if report_verbosity == "full":
        out["expected_stack"] = metrics["expected_stack"]
        out["expected_skills"] = metrics["expected_skills"]
        out["expected_total"] = metrics["expected_total"]

    return out


def _project_case_for_report(case: Dict[str, Any], report_verbosity: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cdm_file": case["cdm_file"],
        "vacancy_title": case["vacancy_title"],
        "vacancy_company": case["vacancy_company"],
        "predicted_keywords": case["predicted_keywords"],
        "passed": case["passed"],
        "strict_passed": case["strict_passed"],
        "strict_fail_reasons": case["strict_fail_reasons"],
        "metrics": _project_metrics_for_report(case["metrics"], report_verbosity=report_verbosity),
    }

    if report_verbosity in {"standard", "full"}:
        out["raw_extractor_output"] = case["raw_extractor_output"]

    if report_verbosity == "full":
        out["vacancy_text"] = case["vacancy_text"]

    return out


def run_vacancy_keywords_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    cases_count: Optional[int],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    seed: Optional[int],
    vacancy_gen_model: Optional[str],
    use_llm_vacancy_gen: bool,
    min_total_matches: int,
    min_stack_matches: int,
    require_all_in_text: bool,
    pass_score_threshold: float,
    report_verbosity: str,
    quiet: bool,
) -> pathlib.Path:
    ensure_dirs()

    if min_total_matches < 0 or min_stack_matches < 0:
        raise ValueError("--min-total-matches and --min-stack-matches must be >= 0")
    if not (0.0 <= pass_score_threshold <= 100.0):
        raise ValueError("--pass-score-threshold must be in [0, 100]")
    if report_verbosity not in REPORT_VERBOSITY_VALUES:
        raise ValueError(f"--report-verbosity must be one of: {', '.join(REPORT_VERBOSITY_VALUES)}")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    cfg: Dict[str, Any] = {}
    if CFG_PATH.is_file():
        cfg = load_yaml(CFG_PATH) or {}
        _log(quiet, f"[init] loaded cfg: {CFG_PATH}")
    else:
        _log(quiet, f"[init] cfg not found: {CFG_PATH} (ok, will use env/cli)")

    cfg_pid, cfg_pver, cfg_seed = _resolve_prompt_from_cfg(cfg)
    cfg_vac_model = _resolve_vacancy_gen_model_from_cfg(cfg)

    env_pid = os.environ.get("RESPONSIBILITIES_PARSER_PROMPT_ID")
    env_pver = os.environ.get("RESPONSIBILITIES_PARSER_PROMPT_VERSION")

    final_pid = prompt_id or cfg_pid or env_pid
    final_pver = prompt_version or cfg_pver or env_pver
    if not final_pid:
        raise EnvironmentError(
            "No prompt_id found. Provide --prompt-id, or set RESPONSIBILITIES_PARSER_PROMPT_ID, "
            "or add tests/tools/model.yaml -> responsibilities_parser.prompt_id"
        )

    final_seed = seed if seed is not None else cfg_seed
    rng = random.Random(final_seed)

    final_vac_model = vacancy_gen_model or cfg_vac_model or DEFAULT_VACANCY_GEN_MODEL

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)
    if cases_count is not None:
        if cases_count <= 0:
            raise ValueError("--cases-count must be > 0")
        # We'll sample from available CDMs
        sample_count = min(cases_count, len(cdm_paths))
        cdm_paths = rng.sample(cdm_paths, k=sample_count)

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
        f"cdm_count={cdm_count} "
        f"cases_count_requested={cases_count} "
        f"cases_count_selected={len(cdm_paths)} "
        f"seed={final_seed} "
        f"use_llm_vacancy_gen={use_llm_vacancy_gen} "
        f"vacancy_gen_model={final_vac_model if use_llm_vacancy_gen else 'template'} "
        f"min_total_matches={min_total_matches} "
        f"min_stack_matches={min_stack_matches} "
        f"require_all_in_text={require_all_in_text} "
        f"pass_score_threshold={pass_score_threshold} "
        f"report_verbosity={report_verbosity}",
    )
    _log(quiet, f"[init] prompt_id={final_pid} prompt_version={final_pver}")

    builder = VacancyTextBuilder()
    synthesizer = VacancyTextSynthesizerLLM(model=final_vac_model, seed=final_seed) if use_llm_vacancy_gen else None
    runner = VacancyKeywordsRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for cdm_path in cdm_paths:
        cdm = load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}
        v_title = vacancy.get("title")
        v_company = vacancy.get("company_name")

        expected_stack = _split_list_like(vacancy.get("vacancy_stack") or vacancy.get("stack") or "")
        expected_skills = _split_list_like(vacancy.get("vacancy_skills") or "")

        vacancy_text = ""
        raw_output = ""
        predicted: List[str] = []
        raw_error: Optional[str] = None

        try:
            if synthesizer is not None:
                vacancy_text = synthesizer.synthesize(vacancy)
                _accumulate_usage(token_usage_total, synthesizer.last_usage)
            else:
                vacancy_text = builder.build(vacancy)

            predicted, raw_output = runner.extract(vacancy_text)
            _accumulate_usage(token_usage_total, runner.last_usage)

        except Exception as e:
            raw_error = repr(e)

        if raw_error is not None:
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "error": raw_error,
                }
            )
            _log(quiet, f"[err] cdm={cdm_path.name} title={v_title} company={v_company} error={raw_error}")
            continue

        metrics = _case_metrics(
            predicted=predicted,
            vacancy_text=vacancy_text,
            expected_stack=expected_stack,
            expected_skills=expected_skills,
        )

        strict_fail_reasons = _strict_fail_reasons(
            metrics=metrics,
            min_total_matches=min_total_matches,
            min_stack_matches=min_stack_matches,
            require_all_in_text=require_all_in_text,
        )
        strict_passed = len(strict_fail_reasons) == 0

        passed = bool(
            metrics["list_ok"]
            and metrics["format_ok"]
            and float(metrics["final_score_pct"]) >= float(pass_score_threshold)
        )

        case = {
            "cdm_file": str(cdm_path),
            "vacancy_title": v_title,
            "vacancy_company": v_company,
            "vacancy_text": vacancy_text,
            "predicted_keywords": predicted,
            "raw_extractor_output": raw_output,
            "passed": passed,
            "strict_passed": strict_passed,
            "strict_fail_reasons": strict_fail_reasons,
            "metrics": metrics,
        }
        cases.append(case)

        _log(
            quiet,
            "[case] "
            f"cdm={cdm_path.name} "
            f"title={v_title} "
            f"company={v_company} "
            f"passed={passed} "
            f"strict_passed={strict_passed} "
            f"score={metrics['final_score_pct']} "
            f"matches={metrics['total_matches_count']} "
            f"stack_matches={metrics['stack_matches_count']} "
            f"precision={metrics['precision_pct']}% "
            f"in_text_ok={metrics['in_text_ok']}",
        )

    total = len(cases)
    passed_n = sum(1 for c in cases if c.get("passed"))
    pass_rate = round((passed_n / total * 100.0), 2) if total else 0.0
    strict_passed_n = sum(1 for c in cases if c.get("strict_passed"))
    strict_pass_rate = round((strict_passed_n / total * 100.0), 2) if total else 0.0

    avg_score = round(sum(float(c["metrics"]["final_score_pct"]) for c in cases) / total, 2) if total else 0.0
    avg_precision = round(sum(float(c["metrics"]["precision_pct"]) for c in cases) / total, 2) if total else 0.0
    avg_recall = round(sum(float(c["metrics"]["recall_pct"]) for c in cases) / total, 2) if total else 0.0
    avg_grounding = round(sum(float(c["metrics"]["grounding_pct"]) for c in cases) / total, 2) if total else 0.0
    avg_total_matches = (
        round(sum(int(c["metrics"]["total_matches_count"]) for c in cases) / total, 3) if total else 0.0
    )
    avg_stack_matches = (
        round(sum(int(c["metrics"]["stack_matches_count"]) for c in cases) / total, 3) if total else 0.0
    )

    format_bad = sum(1 for c in cases if not c["metrics"]["format_ok"])
    list_bad = sum(1 for c in cases if not c["metrics"]["list_ok"])
    in_text_bad = sum(1 for c in cases if not c["metrics"]["in_text_ok"])
    quality_levels = Counter((c["metrics"]["quality_level"] or "").strip() for c in cases)

    strict_only_in_text_fail_count = sum(
        1
        for c in cases
        if (
            not c["strict_passed"]
            and c["metrics"]["list_ok"]
            and c["metrics"]["format_ok"]
            and c["metrics"]["total_matches_count"] >= min_total_matches
            and c["metrics"]["stack_matches_count"] >= min_stack_matches
            and (require_all_in_text and not c["metrics"]["in_text_ok"])
        )
    )

    failed_cases = [
        {
            "cdm_file": c["cdm_file"],
            "vacancy_title": c["vacancy_title"],
            "vacancy_company": c["vacancy_company"],
            "predicted_keywords": c["predicted_keywords"],
            "strict_fail_reasons": c["strict_fail_reasons"],
            "final_score_pct": c["metrics"]["final_score_pct"],
            "quality_level": c["metrics"]["quality_level"],
            "not_in_text": c["metrics"]["not_in_text"],
        }
        for c in cases
        if not c.get("passed")
    ]

    cases_for_report = [_project_case_for_report(case=c, report_verbosity=report_verbosity) for c in cases]

    counts_by_company = Counter((c.get("vacancy_company") or "").strip() for c in cases)
    counts_by_title = Counter((c.get("vacancy_title") or "").strip() for c in cases)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_dir": str(cdm_dir),
        "cdm_count": cdm_count,
        "cases_count": len(cdm_paths),
        "cases_count_requested": cases_count,
        "seed": final_seed,
        "use_llm_vacancy_gen": use_llm_vacancy_gen,
        "vacancy_gen_model": (final_vac_model if use_llm_vacancy_gen else None),
        "vacancy_gen_retries": (VACANCY_GEN_MAX_RETRIES if use_llm_vacancy_gen else 0),
        "report_verbosity": report_verbosity,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "thresholds": {
            "min_total_matches": min_total_matches,
            "min_stack_matches": min_stack_matches,
            "require_all_in_text": require_all_in_text,
            "pass_score_threshold": pass_score_threshold,
        },
        "scoring": {
            "weights": SCORE_WEIGHTS,
            "quality_levels": {
                "strong_pass": "score >= 85",
                "pass": "70 <= score < 85",
                "borderline": "55 <= score < 70",
                "fail": "score < 55",
            },
        },
        "token_usage_total": token_usage_total,
        "summary": {
            "total_cases": total,
            "passed": passed_n,
            "pass_rate_pct": pass_rate,
            "strict_passed": strict_passed_n,
            "strict_pass_rate_pct": strict_pass_rate,
            "avg_final_score_pct": avg_score,
            "avg_precision_pct": avg_precision,
            "avg_recall_pct": avg_recall,
            "avg_grounding_pct": avg_grounding,
            "avg_total_matches": avg_total_matches,
            "avg_stack_matches": avg_stack_matches,
            "list_bad_count": list_bad,
            "format_bad_count": format_bad,
            "in_text_bad_count": in_text_bad,
            "strict_only_in_text_fail_count": strict_only_in_text_fail_count,
            "errors_count": len(errors),
            "failed_cases_count": len(failed_cases),
            "quality_levels": dict(quality_levels),
            "counts_by_company": dict(counts_by_company),
            "counts_by_title": dict(counts_by_title),
        },
        "cases": cases_for_report,
        "failed_cases": failed_cases,
        "errors": errors,
    }

    out_path = REPORTS_DIR / f"responsibilities_parser_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"total_cases={total} "
        f"passed={passed_n} "
        f"pass_rate={pass_rate:.2f}% "
        f"strict_passed={strict_passed_n} "
        f"strict_pass_rate={strict_pass_rate:.2f}% "
        f"avg_score={avg_score:.2f} "
        f"errors={len(errors)} "
        f"format_bad={format_bad} "
        f"in_text_bad={in_text_bad} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build detailed vacancy texts from CDM fixtures, run vacancy_keywords prompt, and report quality metrics."
    )
    parser.add_argument(
        "--cdm-dir",
        type=str,
        default=str(DEFAULT_CDM_DIR),
        help=f"CDM fixtures dir (default: {DEFAULT_CDM_DIR})",
    )
    parser.add_argument(
        "--cdm-count",
        type=int,
        default=None,
        help="Use first N CDM fixtures (sorted by filename). Default: all.",
    )
    parser.add_argument(
        "--cases-count",
        type=int,
        default=None,
        help="If set, sample N CDM fixtures from the available set (after --cdm-count). Default: use all selected.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides cfg seed if provided).",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Override vacancy_keywords prompt id (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override vacancy_keywords prompt version (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--use-llm-vacancy-gen",
        action="store_true",
        help="Use LLM to synthesize vacancy text (more realistic, less deterministic). Default: template-based builder.",
    )
    parser.add_argument(
        "--vacancy-gen-model",
        type=str,
        default=None,
        help=f"Vacancy synthesis model override (default: {DEFAULT_VACANCY_GEN_MODEL}). Only used with --use-llm-vacancy-gen.",
    )
    parser.add_argument(
        "--min-total-matches",
        type=int,
        default=2,
        help="Strict threshold only: minimum matched keywords vs (vacancy_stack U vacancy_skills).",
    )
    parser.add_argument(
        "--min-stack-matches",
        type=int,
        default=1,
        help="Strict threshold only: minimum matched keywords vs vacancy_stack.",
    )
    parser.add_argument(
        "--no-require-all-in-text",
        action="store_true",
        help="Disable strict check that every predicted keyword must appear in vacancy text (strict_passed only).",
    )
    parser.add_argument(
        "--pass-score-threshold",
        type=float,
        default=DEFAULT_PASS_SCORE_THRESHOLD,
        help=(
            "Main pass threshold for score-based evaluation [0..100]. "
            "Case passes when list/format are valid and final_score >= threshold."
        ),
    )
    parser.add_argument(
        "--report-verbosity",
        type=str,
        choices=list(REPORT_VERBOSITY_VALUES),
        default="compact",
        help=(
            "Report detail level: compact (smallest), standard (adds raw extractor output and matches), "
            "full (adds vacancy text and expected terms)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console progress output.",
    )

    args = parser.parse_args()

    run_vacancy_keywords_dataset(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        cases_count=args.cases_count,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        seed=args.seed,
        vacancy_gen_model=args.vacancy_gen_model,
        use_llm_vacancy_gen=bool(args.use_llm_vacancy_gen),
        min_total_matches=int(args.min_total_matches),
        min_stack_matches=int(args.min_stack_matches),
        require_all_in_text=not bool(args.no_require_all_in_text),
        pass_score_threshold=float(args.pass_score_threshold),
        report_verbosity=str(args.report_verbosity),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()

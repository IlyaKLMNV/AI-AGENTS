from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml
from openai import OpenAI

from adapters.adapters import to_input_form

ROOT = pathlib.Path(__file__).resolve().parents[1]
CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
REPORTS_DIR = ROOT / "tests" / "reports" / "telegram_generator"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"

DEFAULT_LIMIT = 10
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"

TELEGRAM_GEN_DIR = ROOT / "telegramMessageGenerator-main"
if TELEGRAM_GEN_DIR.is_dir():
    sys.path.append(str(TELEGRAM_GEN_DIR))

TELEGRAM_GENERATOR_AVAILABLE = False
TELEGRAM_IMPORT_ERROR: str | None = None
try:
    from telegramGenerator import InputForm as TGInputForm, TelegramMessageGenerator  # type: ignore

    TELEGRAM_GENERATOR_AVAILABLE = True
except Exception as exc:
    TGInputForm = None  # type: ignore
    TelegramMessageGenerator = None  # type: ignore
    TELEGRAM_GENERATOR_AVAILABLE = False
    TELEGRAM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass
class EvalResult:
    facts_present: Dict[str, bool]
    hallucinated_facts: List[str]
    question_present: bool
    comment: str


def ensure_dirs(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _normalize_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_json_substring(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    return None


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty json text")
    try:
        return json.loads(text)
    except Exception:
        extracted = _extract_json_substring(text)
        if not extracted:
            raise
        return json.loads(extracted)


def _extract_numbers(text: str) -> List[int]:
    nums: List[int] = []
    for raw in re.findall(r"\d+(?:[\s\u00A0]\d+)*", text or ""):
        cleaned = re.sub(r"[\s\u00A0]+", "", raw)
        try:
            nums.append(int(cleaned))
        except ValueError:
            continue
    return nums


def _extra_numbers(facts: Dict[str, str], message: str) -> List[str]:
    allowed: set[int] = set()
    for value in facts.values():
        allowed.update(_extract_numbers(str(value)))
    msg_nums = set(_extract_numbers(message))
    extra = sorted(n for n in msg_nums if n not in allowed)
    return [f"extra_numbers:{extra}"] if extra else []


def _normalize_for_match(text: str) -> str:
    t = _normalize_text(text).lower()
    t = re.sub(r"[\W_]+", "", t, flags=re.UNICODE)
    return t


def _contains_normalized(needle: str, haystack: str) -> bool:
    n = _normalize_for_match(needle)
    h = _normalize_for_match(haystack)
    return bool(n) and n in h


def _expected_work_mode_from_cdm(cdm: Dict[str, Any]) -> str | None:
    vacancy = (cdm or {}).get("vacancy") or {}
    for key in ("work_format", "workFormat", "work_mode", "workMode"):
        val = vacancy.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def _company_name_present_when_hidden(company_hidden: bool, original_company_name: str | None, message: str) -> bool:
    if not company_hidden:
        return False
    if not original_company_name or not original_company_name.strip():
        return False
    name = original_company_name.strip()
    if name.lower() == "скрыто":
        return False
    return _contains_normalized(name, message)


def _tech_tokens(text: str) -> set[str]:
    t = _normalize_text(text).lower()
    tokens = re.findall(r"[a-z0-9\+\#\.]{2,}", t, flags=re.IGNORECASE)
    return {tok for tok in tokens if tok}


def _stack_partially_mentioned(expected_stack: str, message: str) -> bool:
    exp = _tech_tokens(expected_stack)
    if not exp:
        return True
    msg = _tech_tokens(message)
    return bool(exp & msg)


def _build_expected_facts_report(input_form: TGInputForm, include_salary: bool) -> Dict[str, str]:
    facts: Dict[str, str] = {
        "company_name": str(getattr(input_form, "company_name", "") or "").strip(),
        "vacancy_name": str(getattr(input_form, "vacancy_name", "") or "").strip(),
        "company_description": str(getattr(input_form, "company_description", "") or "").strip(),
        "vacancy_responsibilities": str(getattr(input_form, "vacancy_responsibilities", "") or "").strip(),
        "vacancy_stack": str(getattr(input_form, "vacancy_stack", "") or "").strip(),
        "vacancy_skills": str(getattr(input_form, "vacancy_skills", "") or "").strip(),
    }
    if include_salary:
        facts["salary_range"] = str(getattr(input_form, "salary", "") or "").strip()
    return {k: v for k, v in facts.items() if v}


def _build_expected_facts_for_eval(expected_facts_report: Dict[str, str], company_hidden: bool) -> Dict[str, str]:
    facts = dict(expected_facts_report)
    # IMPORTANT:
    # - vacancy_name is always allowed and should always be evaluated
    # - company_name is not evaluated when hidden (we enforce hiding via separate forbidden-check)
    if company_hidden:
        facts.pop("company_name", None)
    return facts


def _build_allowed_context_facts(input_form: TGInputForm) -> Dict[str, str]:
    candidate_source = str(getattr(input_form, "candidate_source", "") or "").strip()
    reason = str(getattr(input_form, "reason_of_communication", "") or "").strip()

    allowed: Dict[str, str] = {}
    if candidate_source:
        allowed["candidate_source"] = candidate_source
    if reason:
        allowed["reason_of_communication"] = reason
    return allowed


def _required_keys(expected_facts_report: Dict[str, str], include_salary: bool) -> List[str]:
    # vacancy_name is ALWAYS required
    keys: List[str] = ["vacancy_name", "vacancy_responsibilities"]

    company_name = expected_facts_report.get("company_name", "")
    if company_name and company_name != "СКРЫТО":
        keys.insert(0, "company_name")

    if include_salary and "salary_range" in expected_facts_report:
        keys.append("salary_range")

    return keys


def _resolve_prompt_id() -> str:
    prompt_id = os.environ.get("FIRST_TOUCH_PROMPT_ID")
    if prompt_id:
        return str(prompt_id)

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")

    cfg = load_yaml(CFG_PATH)
    comp = _component_cfg(cfg, "first_touch")
    prompt_id = comp.get("prompt_id") if isinstance(comp, dict) else None
    if not prompt_id:
        raise RuntimeError(
            "Missing FIRST_TOUCH_PROMPT_ID env var and prompt_id in tests/tools/model.yaml (first_touch section)."
        )

    os.environ["FIRST_TOUCH_PROMPT_ID"] = str(prompt_id)
    return str(prompt_id)


def _eval_instruction() -> str:
    return (
        "Ты строгий QA-ревьюер первого сообщения рекрутера.\n"
        "Даны:\n"
        "- expected_facts: факты вакансии (то, с чем сравниваем наличие фактов в тексте)\n"
        "- allowed_context_facts: допустимые контекстные факты (источник кандидата, причина контакта), их НЕ считать галлюцинациями\n"
        "- generated_message: текст сообщения\n\n"
        "Задача:\n"
        "1) facts_present: для каждого ключа из expected_facts определить true/false.\n"
        "   true только если факт явно упомянут или ясно перефразирован.\n"
        "2) hallucinated_facts: список фактических утверждений про вакансию/условия,\n"
        "   которых нет в expected_facts и allowed_context_facts.\n"
        "   Не добавляй общие рекрутерские фразы типа 'заинтересовал ваш опыт', 'обратили внимание на профиль',\n"
        "   приветствия и вежливые вводные.\n"
        "   Если сомневаешься - лучше НЕ добавляй пункт.\n"
        "3) question_present: есть ли в сообщении вопросительный CTA.\n\n"
        "Верни строго JSON:\n"
        "{"
        '"facts_present": {"company_name": true/false, ...},'
        '"hallucinated_facts": ["..."],'
        '"question_present": true/false,'
        '"comment": "кратко на русском"'
        "}"
    )


def _eval_payload(expected_facts: Dict[str, str], allowed_context_facts: Dict[str, str], message: str) -> str:
    payload = {
        "instruction": _eval_instruction(),
        "expected_facts": expected_facts,
        "allowed_context_facts": allowed_context_facts,
        "generated_message": message,
    }
    return json.dumps(payload, ensure_ascii=False)


def evaluate_message(
    client: OpenAI,
    eval_model: str,
    expected_facts: Dict[str, str],
    allowed_context_facts: Dict[str, str],
    message: str,
) -> EvalResult:
    response = client.responses.create(
        model=eval_model,
        input=_eval_payload(expected_facts, allowed_context_facts, message),
    )
    raw = _normalize_text(getattr(response, "output_text", "") or "")
    data = _safe_json_loads(raw)

    facts_present: Dict[str, bool] = {}
    hallucinated_facts: List[str] = []
    question_present = "?" in message
    comment = ""

    facts_present_raw = data.get("facts_present") if isinstance(data, dict) else {}
    for key in expected_facts:
        if isinstance(facts_present_raw, dict):
            facts_present[key] = bool(facts_present_raw.get(key))
        else:
            facts_present[key] = False

    if isinstance(data, dict):
        h = data.get("hallucinated_facts") or []
        if not isinstance(h, list):
            h = [str(h)]
        hallucinated_facts = [str(x).strip() for x in h if str(x).strip()]

        if "question_present" in data:
            question_present = bool(data.get("question_present"))
        comment = str(data.get("comment") or "").strip()

    if "vacancy_stack" in expected_facts and not facts_present.get("vacancy_stack", False):
        if _stack_partially_mentioned(expected_facts.get("vacancy_stack", ""), message):
            facts_present["vacancy_stack"] = True

    return EvalResult(
        facts_present=facts_present,
        hallucinated_facts=hallucinated_facts,
        question_present=question_present,
        comment=comment,
    )


def _compute_summary(
    cases: List[Dict[str, Any]],
    requested_limit: int,
    fixtures_found: int,
    company_leaks_count: int,
) -> Dict[str, Any]:
    total = len(cases)
    hidden_cases = sum(1 for c in cases if c.get("company_hidden"))
    visible_cases = total - hidden_cases

    pass_count = sum(1 for c in cases if c.get("result", {}).get("passed"))
    strict_pass_count = sum(1 for c in cases if c.get("result", {}).get("passed_strict"))
    question_count = sum(1 for c in cases if c.get("result", {}).get("question_present"))
    halluc_free_count = sum(1 for c in cases if not (c.get("result", {}).get("hallucinated_facts") or []))

    missing_required_dist: Dict[str, int] = {}
    missing_optional_dist: Dict[str, int] = {}
    hallucinated_dist: Dict[str, int] = {}
    fail_reasons_dist: Dict[str, int] = {}

    for c in cases:
        r = c.get("result") or {}
        for k in r.get("missing_required_keys", []) or []:
            missing_required_dist[k] = missing_required_dist.get(k, 0) + 1
        for k in r.get("missing_optional_keys", []) or []:
            missing_optional_dist[k] = missing_optional_dist.get(k, 0) + 1
        for h in r.get("hallucinated_facts", []) or []:
            hallucinated_dist[h] = hallucinated_dist.get(h, 0) + 1
        for fr in r.get("fail_reasons", []) or []:
            fail_reasons_dist[fr] = fail_reasons_dist.get(fr, 0) + 1

    rate = (pass_count / total) if total else 0.0
    strict_rate = (strict_pass_count / total) if total else 0.0
    q_rate = (question_count / total) if total else 0.0
    h_rate = (halluc_free_count / total) if total else 0.0

    return {
        "requested_limit": requested_limit,
        "fixtures_found": fixtures_found,
        "actual_cases": total,
        "hidden_cases": hidden_cases,
        "visible_cases": visible_cases,
        "pass_rate": rate,
        "strict_pass_rate": strict_rate,
        "question_rate": q_rate,
        "hallucination_free_rate": h_rate,
        "company_leaks_count": company_leaks_count,
        "missing_required_distribution": missing_required_dist,
        "missing_optional_distribution": missing_optional_dist,
        "hallucinated_facts_distribution": hallucinated_dist,
        "fail_reasons_distribution": fail_reasons_dist,
    }


def _report_definitions(require_question: bool) -> Dict[str, Any]:
    pass_criteria = [
        "missing_required_keys is empty",
        "extra_numbers is empty",
    ]
    if require_question:
        pass_criteria.append("question_present=true")
    pass_criteria.append("if company_hidden=true then original_company_name must NOT be mentioned")

    return {
        "pass_criteria": pass_criteria,
        "strict_criteria": [
            "passed=true",
            "hallucinated_facts is empty",
        ],
        "notes": {
            "missing_optional_keys": "Диагностика. Не влияет на passed.",
            "hallucinated_facts": "Диагностика (passed_strict=false, но passed может быть true).",
            "allowed_context_facts": "Контекст (candidate_source, reason_of_communication) не считается галлюцинацией.",
        },
    }


def _status_digest(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed_count = 0
    failed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for c in cases:
        if "error" in c:
            errors.append({"cdm_file": c.get("cdm_file"), "error": c.get("error")})
            continue

        r = c.get("result") or {}
        if r.get("passed"):
            passed_count += 1
            continue

        failed.append(
            {
                "cdm_file": c.get("cdm_file"),
                "company_hidden": c.get("company_hidden"),
                "vacancy": c.get("vacancy"),
                "fail_reasons": r.get("fail_reasons") or [],
            }
        )

    return {
        "passed_count": passed_count,
        "failed_count": len(failed),
        "error_count": len(errors),
        "failed": failed,
        "errors": errors,
    }


def run_suite(
    limit: int,
    eval_model: str,
    require_question: bool,
    include_salary: bool,
    hide_company: bool,
    hide_company_ratio: float,
    seed: int | None,
    cdm_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    if not TELEGRAM_GENERATOR_AVAILABLE:
        raise RuntimeError(f"telegramGenerator import failed: {TELEGRAM_IMPORT_ERROR}")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    prompt_id = _resolve_prompt_id()

    all_files = sorted(cdm_dir.glob("*.json"))
    fixtures_found = len(all_files)
    if not all_files:
        raise FileNotFoundError(f"No CDM fixtures found in {cdm_dir}")

    cdm_files = all_files[:limit] if limit > 0 else all_files

    if seed is not None:
        random.seed(seed)

    print(
        "[init] "
        f"prompt_id={prompt_id}, eval_model={eval_model}, require_question={require_question}, "
        f"include_salary={include_salary}, hide_company={hide_company}, hide_company_ratio={hide_company_ratio}, seed={seed}"
    )
    print(f"[gen] fixtures_found={fixtures_found}, requested_limit={limit}, actual_cases={len(cdm_files)}")

    generator = TelegramMessageGenerator(api_key=os.environ.get("OPENAI_API_KEY"))
    client = OpenAI()

    cases: List[Dict[str, Any]] = []
    total_cases = len(cdm_files)
    company_leaks_count = 0

    for idx, cdm_path in enumerate(cdm_files, start=1):
        print(f"[run] case {idx}/{total_cases} ({cdm_path.name})")

        try:
            cdm = json.loads(cdm_path.read_text(encoding="utf-8"))
            form_dict = to_input_form(cdm)

            original_company_name = str(form_dict.get("company_name") or "").strip() or None
            expected_work_mode = _expected_work_mode_from_cdm(cdm)

            company_hidden = bool(hide_company) or (hide_company_ratio > 0.0 and random.random() < hide_company_ratio)
            if company_hidden:
                form_dict["company_name"] = "СКРЫТО"

            input_form = TGInputForm(**form_dict)

            expected_facts_report = _build_expected_facts_report(input_form, include_salary=include_salary)
            expected_facts_eval = _build_expected_facts_for_eval(expected_facts_report, company_hidden=company_hidden)
            allowed_context_facts = _build_allowed_context_facts(input_form)

            required_keys = _required_keys(expected_facts_report=expected_facts_report, include_salary=include_salary)
            optional_keys = [k for k in expected_facts_eval.keys() if k not in required_keys]

            message = _normalize_text(generator.generate_message(input_form))

            company_leaked = _company_name_present_when_hidden(company_hidden, original_company_name, message)
            if company_leaked:
                company_leaks_count += 1

            eval_result = evaluate_message(
                client=client,
                eval_model=eval_model,
                expected_facts=expected_facts_eval,
                allowed_context_facts=allowed_context_facts,
                message=message,
            )

            missing_required_keys = [k for k in required_keys if not bool(eval_result.facts_present.get(k, False))]
            missing_optional_keys = [k for k in optional_keys if not bool(eval_result.facts_present.get(k, False))]

            extra_numbers = _extra_numbers(expected_facts_report, message)

            fail_reasons: List[str] = []
            if missing_required_keys:
                fail_reasons.append("missing_required_keys")
            if extra_numbers:
                fail_reasons.append("extra_numbers")
            if require_question and not eval_result.question_present:
                fail_reasons.append("no_question")
            if company_leaked:
                fail_reasons.append("company_name_present_when_hidden")

            passed = not fail_reasons
            passed_strict = bool(passed and not eval_result.hallucinated_facts)

            case = {
                "cdm_file": cdm_path.name,
                "company_hidden": company_hidden,
                "original_company_name": original_company_name,
                "expected_work_mode": expected_work_mode,
                "vacancy": {
                    "company_name": expected_facts_report.get("company_name", ""),
                    "vacancy_name": expected_facts_report.get("vacancy_name", ""),
                },
                "expected_facts": expected_facts_report,
                "message": message,
                "result": {
                    "passed": passed,
                    "passed_strict": passed_strict,
                    "question_present": eval_result.question_present,
                    "missing_required_keys": missing_required_keys,
                    "missing_optional_keys": missing_optional_keys,
                    "hallucinated_facts": eval_result.hallucinated_facts,
                    "extra_numbers": extra_numbers,
                    "fail_reasons": fail_reasons,
                },
                "meta": {"comment": eval_result.comment},
            }
            cases.append(case)

        except Exception as exc:
            cases.append({"cdm_file": cdm_path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[warn] case failed: {cdm_path.name}: {type(exc).__name__}: {exc}")

    finished_at = datetime.datetime.now()

    summary = _compute_summary(
        cases=[c for c in cases if "result" in c],
        requested_limit=limit,
        fixtures_found=fixtures_found,
        company_leaks_count=company_leaks_count,
    )

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "prompt_id": prompt_id,
        "eval_model": eval_model,
        "definitions": _report_definitions(require_question=require_question),
        "status": _status_digest(cases),
        "summary": summary,
        "cases": cases,
    }

    ensure_dirs(out_dir)
    out_path = out_dir / f"telegram_generator_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] report saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram generator first-touch test")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--cdm-dir", type=pathlib.Path, default=CDM_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPORTS_DIR)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)

    rq = parser.add_mutually_exclusive_group()
    rq.add_argument("--require-question", dest="require_question", action="store_true", help="Require a question CTA")
    rq.add_argument("--no-require-question", dest="require_question", action="store_false", help="Do not require a question CTA")
    parser.set_defaults(require_question=True)

    parser.add_argument("--include-salary", action="store_true", help="Include salary_range into expected facts")
    parser.add_argument("--hide-company", action="store_true", help='Force company_name to "СКРЫТО" in all cases')
    parser.add_argument("--hide-company-ratio", type=float, default=0.0, help="Randomly hide company in N% cases (0..1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for hide-company-ratio")

    args = parser.parse_args()

    report_path = run_suite(
        limit=args.limit,
        eval_model=args.eval_model,
        require_question=bool(args.require_question),
        include_salary=bool(args.include_salary),
        hide_company=bool(args.hide_company),
        hide_company_ratio=float(args.hide_company_ratio or 0.0),
        seed=args.seed,
        cdm_dir=args.cdm_dir,
        out_dir=args.out_dir,
    )
    print("Report ->", report_path)


if __name__ == "__main__":
    main()

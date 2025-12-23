from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI
import yaml

from adapters.adapters import to_input_form

ROOT = pathlib.Path(__file__).resolve().parents[1]
CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
REPORTS_DIR = ROOT / "tests" / "reports" / "telegram_generator"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"

DEFAULT_LIMIT = 10
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
DEFAULT_COVERAGE_THRESHOLD = 1.0

TELEGRAM_GEN_DIR = ROOT / "telegramMessageGenerator-main"
if TELEGRAM_GEN_DIR.is_dir():
    sys.path.append(str(TELEGRAM_GEN_DIR))

TELEGRAM_GENERATOR_AVAILABLE = False
TELEGRAM_IMPORT_ERROR: str | None = None
try:
    from telegramGenerator import InputForm as TGInputForm, TelegramMessageGenerator  # type: ignore

    TELEGRAM_GENERATOR_AVAILABLE = True
    TELEGRAM_IMPORT_ERROR = None
except Exception as exc:
    TGInputForm = None  # type: ignore
    TelegramMessageGenerator = None  # type: ignore
    TELEGRAM_GENERATOR_AVAILABLE = False
    TELEGRAM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass
class EvalResult:
    facts_present: Dict[str, bool]
    missing_facts: List[str]
    hallucinated_facts: List[str]
    question_present: bool
    comment: str
    used_heuristics: bool
    coverage: float
    hallucination_free: bool
    passed: bool


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


def _tokenize(text: str) -> List[str]:
    t = _normalize_text(text).lower()
    t = re.sub(r"[^\w]+", " ", t, flags=re.UNICODE)
    return [p for p in t.split() if len(p) >= 3]


def _ensure_russian_comment(comment: str) -> str:
    text = (comment or "").strip()
    if not text:
        return "Комментарий отсутствует."
    if re.search(r"[А-Яа-яЁё]", text):
        return text
    return f"Комментарий (оригинал): {text}"


def _token_coverage(required: str, message: str) -> float:
    req = set(_tokenize(required))
    if not req:
        return 1.0
    msg = set(_tokenize(message))
    return len(req & msg) / len(req)


def _extract_numbers(text: str) -> List[int]:
    nums: List[int] = []
    for raw in re.findall(r"\d[\d\s]{2,}", text or ""):
        cleaned = re.sub(r"\s+", "", raw)
        try:
            nums.append(int(cleaned))
        except ValueError:
            continue
    return nums


def _salary_present(salary_text: str, message: str) -> bool:
    if not salary_text:
        return True
    required = _extract_numbers(salary_text)
    if not required:
        return False
    found = set(_extract_numbers(message))
    return all(n in found for n in required)


def _extra_numbers_hallucinations(facts: Dict[str, str], message: str) -> List[str]:
    allowed: set[int] = set()
    for value in facts.values():
        allowed.update(_extract_numbers(str(value)))
    msg_nums = set(_extract_numbers(message))
    extra = sorted(n for n in msg_nums if n not in allowed)
    if not extra:
        return []
    return [f"extra_numbers:{extra}"]


def _build_required_facts(input_form: TGInputForm) -> Dict[str, str]:
    facts = {
        "company_name": input_form.company_name,
        "vacancy_name": input_form.vacancy_name,
        "company_description": input_form.company_description or "",
        "vacancy_responsibilities": input_form.vacancy_responsibilities or "",
        "vacancy_stack": input_form.vacancy_stack or "",
        "vacancy_skills": input_form.vacancy_skills or "",
        "salary_range": input_form.salary or "",
    }
    return {k: str(v).strip() for k, v in facts.items() if str(v or "").strip()}


def _resolve_prompt_id() -> tuple[str, str]:
    prompt_id = os.environ.get("FIRST_TOUCH_PROMPT_ID")
    if prompt_id:
        return str(prompt_id), "env"

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")

    cfg = load_yaml(CFG_PATH)
    comp = _component_cfg(cfg, "first_touch")
    prompt_id = comp.get("prompt_id") if isinstance(comp, dict) else None
    if not prompt_id:
        raise RuntimeError(
            "Missing FIRST_TOUCH_PROMPT_ID env var and prompt_id in tests/tools/model.yaml "
            "(first_touch section)."
        )

    os.environ["FIRST_TOUCH_PROMPT_ID"] = str(prompt_id)
    return str(prompt_id), "config"


def _make_console_summary(summary: Dict[str, Any]) -> str:
    total = int(summary.get("всего") or 0)
    pass_rate = float(summary.get("доля_прошедших") or 0.0)
    coverage_avg = float(summary.get("среднее_покрытие") or 0.0)
    hallucination_free_rate = float(summary.get("доля_без_галлюцинаций") or 0.0)
    question_rate = float(summary.get("доля_с_вопросом") or 0.0)
    heuristics_rate = float(summary.get("доля_с_эвристикой") or 0.0)

    return (
        "[summary] "
        f"всего: {total}, "
        f"прошли: {pass_rate * 100:.1f}%, "
        f"покрытие: {coverage_avg * 100:.1f}%, "
        f"без_галлюцинаций: {hallucination_free_rate * 100:.1f}%, "
        f"с_вопросом: {question_rate * 100:.1f}%, "
        f"эвристика: {heuristics_rate * 100:.1f}%"
    )


def _eval_instruction() -> str:
    return (
        "Ты строгий QA-ревьюер для первого касания рекрутера.\n"
        "Даны факты вакансии и сгенерированное сообщение. Нужно определить:\n"
        "1) Для каждого факта вакансии — упомянут ли он явно или внятно перефразирован.\n"
        "2) Есть ли в сообщении новые фактические детали, которых нет в данных вакансии.\n"
        "Игнорируй приветствия, вежливые обороты и общие рекрутерские формулировки.\n"
        "Верни СТРОГО JSON с ключами:\n"
        "{"
        "\"facts_present\": {\"company_name\": true/false, ...},"
        "\"hallucinated_facts\": [\"...\"],"
        "\"question_present\": true/false,"
        "\"comment\": \"Комментарий на русском\""
        "}"
    )


def _eval_payload(facts: Dict[str, str], message: str) -> str:
    payload = {
        "instruction": _eval_instruction(),
        "vacancy_facts": facts,
        "generated_message": message,
    }
    return json.dumps(payload, ensure_ascii=False)


def evaluate_message(
    client: OpenAI,
    eval_model: str,
    facts: Dict[str, str],
    message: str,
    coverage_threshold: float,
    require_question: bool,
) -> EvalResult:
    question_present = "?" in message
    facts_present: Dict[str, bool] = {}
    missing_facts: List[str] = []
    hallucinated_facts: List[str] = []
    comment = ""
    used_heuristics = False

    try:
        response = client.responses.create(model=eval_model, input=_eval_payload(facts, message))
        raw = _normalize_text(getattr(response, "output_text", "") or "")
        data = _safe_json_loads(raw)

        facts_present_raw = data.get("facts_present") if isinstance(data, dict) else {}
        for key in facts:
            if isinstance(facts_present_raw, dict):
                facts_present[key] = bool(facts_present_raw.get(key))
            else:
                facts_present[key] = False

        hallucinated = []
        if isinstance(data, dict):
            hallucinated = data.get("hallucinated_facts") or []
            if "question_present" in data:
                question_present = bool(data.get("question_present"))
            comment = str(data.get("comment") or "").strip()

        if not isinstance(hallucinated, list):
            hallucinated = [str(hallucinated)]
        hallucinated_facts = [str(x).strip() for x in hallucinated if str(x).strip()]
        comment = _ensure_russian_comment(comment)

    except Exception as exc:
        for key, value in facts.items():
            if key in ("company_name", "vacancy_name"):
                facts_present[key] = _token_coverage(value, message) >= 0.9
            elif key in ("vacancy_stack", "vacancy_skills"):
                facts_present[key] = _token_coverage(value, message) >= 0.6
            elif key in ("company_description", "vacancy_responsibilities"):
                facts_present[key] = _token_coverage(value, message) >= 0.3
            elif key == "salary_range":
                facts_present[key] = _salary_present(value, message)
            else:
                facts_present[key] = _token_coverage(value, message) >= 0.5

        hallucinated_facts = _extra_numbers_hallucinations(facts, message)
        comment = f"оценка не удалась: {type(exc).__name__}: {exc}"
        comment = _ensure_russian_comment(comment)
        used_heuristics = True

    missing_facts = [k for k, v in facts_present.items() if not v]
    coverage = 1.0 if not facts else (len(facts) - len(missing_facts)) / len(facts)
    hallucination_free = not hallucinated_facts
    passed = coverage >= coverage_threshold and hallucination_free
    if require_question and not question_present:
        passed = False

    return EvalResult(
        facts_present=facts_present,
        missing_facts=missing_facts,
        hallucinated_facts=hallucinated_facts,
        question_present=question_present,
        comment=comment,
        used_heuristics=used_heuristics,
        coverage=coverage,
        hallucination_free=hallucination_free,
        passed=passed,
    )


def _compute_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    if total == 0:
        return {
            "всего": 0,
            "доля_прошедших": 0.0,
            "среднее_покрытие": 0.0,
            "доля_без_галлюцинаций": 0.0,
            "доля_с_вопросом": 0.0,
            "доля_с_эвристикой": 0.0,
            "распределение_пропусков": {},
            "распределение_галлюцинаций": {},
        }

    pass_count = sum(1 for c in cases if c.get("passed"))
    coverage_avg = sum(float(c.get("coverage") or 0.0) for c in cases) / total
    hallucination_free_rate = sum(1 for c in cases if c.get("hallucination_free")) / total
    question_rate = sum(1 for c in cases if c.get("question_present")) / total
    used_heuristics_rate = sum(1 for c in cases if c.get("used_heuristics")) / total

    missing_counts: Dict[str, int] = {}
    hallucinated_counts: Dict[str, int] = {}
    for c in cases:
        for m in c.get("missing_facts", []) or []:
            missing_counts[m] = missing_counts.get(m, 0) + 1
        for h in c.get("hallucinated_facts", []) or []:
            hallucinated_counts[h] = hallucinated_counts.get(h, 0) + 1

    return {
        "всего": total,
        "доля_прошедших": pass_count / total,
        "среднее_покрытие": coverage_avg,
        "доля_без_галлюцинаций": hallucination_free_rate,
        "доля_с_вопросом": question_rate,
        "доля_с_эвристикой": used_heuristics_rate,
        "распределение_пропусков": missing_counts,
        "распределение_галлюцинаций": hallucinated_counts,
    }


def run_suite(
    limit: int,
    eval_model: str,
    coverage_threshold: float,
    require_question: bool,
    cdm_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    if not TELEGRAM_GENERATOR_AVAILABLE:
        raise RuntimeError(f"telegramGenerator import failed: {TELEGRAM_IMPORT_ERROR}")

    prompt_id, prompt_source = _resolve_prompt_id()

    cdm_files = sorted(cdm_dir.glob("*.json"))
    if not cdm_files:
        raise FileNotFoundError(f"No CDM fixtures found in {cdm_dir}")

    if limit > 0:
        cdm_files = cdm_files[:limit]

    print(
        "[init] telegram generator: "
        f"prompt_id={prompt_id} ({prompt_source}), eval_model={eval_model}, "
        f"coverage_threshold={coverage_threshold}, require_question={require_question}"
    )
    print(f"[gen] loading CDM fixtures from {cdm_dir}...")
    print(f"[gen] found {len(cdm_files)} fixtures")

    generator = TelegramMessageGenerator(api_key=os.environ.get("OPENAI_API_KEY"))
    client = OpenAI()

    cases: List[Dict[str, Any]] = []
    total_cases = len(cdm_files)
    for idx, cdm_path in enumerate(cdm_files, start=1):
        print(f"[run] case {idx}/{total_cases} ({cdm_path.name})")
        case: Dict[str, Any] = {
            "cdm_file": cdm_path.name,
            "passed": False,
            "coverage": 0.0,
            "hallucination_free": False,
            "question_present": False,
            "used_heuristics": False,
            "missing_facts": [],
            "hallucinated_facts": [],
        }

        try:
            cdm = json.loads(cdm_path.read_text(encoding="utf-8"))
            form_dict = to_input_form(cdm)
            input_form = TGInputForm(**form_dict)
            facts = _build_required_facts(input_form)

            message = generator.generate_message(input_form)
            message = _normalize_text(message)

            eval_result = evaluate_message(
                client=client,
                eval_model=eval_model,
                facts=facts,
                message=message,
                coverage_threshold=coverage_threshold,
                require_question=require_question,
            )

            case.update(
                {
                    "facts": facts,
                    "message": message,
                    "missing_facts": eval_result.missing_facts,
                    "hallucinated_facts": eval_result.hallucinated_facts,
                    "question_present": eval_result.question_present,
                    "coverage": eval_result.coverage,
                    "hallucination_free": eval_result.hallucination_free,
                    "passed": eval_result.passed,
                    "used_heuristics": eval_result.used_heuristics,
                    "comment": eval_result.comment,
                }
            )

        except Exception as exc:
            case["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[warn] case failed: {cdm_path.name}: {case['error']}")

        cases.append(case)

    summary = _compute_summary(cases)
    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "prompt_id": prompt_id,
        "eval_model": eval_model,
        "coverage_threshold": coverage_threshold,
        "require_question": require_question,
        "cases": cases,
        "summary": summary,
    }

    ensure_dirs(out_dir)
    out_path = out_dir / f"telegram_generator_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(_make_console_summary(summary))
    print()
    print(f"[done] report saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram generator first-touch test")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--cdm-dir", type=pathlib.Path, default=CDM_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPORTS_DIR)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument("--require-question", action="store_true")
    args = parser.parse_args()

    report_path = run_suite(
        limit=args.limit,
        eval_model=args.eval_model,
        coverage_threshold=args.coverage_threshold,
        require_question=bool(args.require_question),
        cdm_dir=args.cdm_dir,
        out_dir=args.out_dir,
    )
    print("Report ->", report_path)


if __name__ == "__main__":
    main()

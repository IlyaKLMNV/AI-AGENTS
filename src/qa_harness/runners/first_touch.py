"""Раннер first_touch: генерация первого касания → LLM-судья фактов + эвристики.

Что тестируем — промпт `first_touch`: из payload (кандидат/вакансия/причина/стек/зарплата) генерит
сообщение первого касания. Без бэкенда. Оценка кейса:
- LLM-судья (отдельная модель): facts_present (каждый ожидаемый факт упомянут?), hallucinated_facts
  (выдуманные факты про вакансию/условия), question_present (есть CTA-вопрос);
- эвристики: extra_numbers (числа ≥5 цифр не из фактов), company_hidden (нет утечки названия).

Итог (passed) = все обязательные факты есть & нет лишних чисел & есть вопрос (если нужен) & при
company_hidden нет названия компании. Выдуманные факты (hallucinated) — СИГНАЛ, не gate (LLM-судья шумит
на общих фразах). quality ≠ infra: сбой генерации/судьи → errors.
`--offline` реплеит offline_message и использует эвристику вместо LLM-судьи (галлюцинации офлайн не ловятся).

  python -m qa_harness.runners.first_touch --offline                 # replay + эвристика, без сети
  python -m qa_harness.runners.first_touch                           # онлайн: генерация + LLM-судья
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, run_cases, usage_total
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.first_touch import (
    FactJudge,
    GoldenCase,
    company_name_leaked,
    extra_numbers,
    facts_present_heuristic,
    forbidden_phrases,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "first_touch" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "first_touch"
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
PAYLOAD_KEYS = (
    "candidate_name", "recruiter_name", "candidate_source", "reason_of_communication",
    "hiring_company_name", "vacancy_name", "vacancy_responsibilities", "message_formality",
    "company_description", "vacancy_stack", "salary_range",
)


def build_parser(default_component: str = "first_touch", default_golden: Path = DEFAULT_GOLDEN) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="first_touch QA runner (generate + LLM fact-judge + heuristics).")
    p.add_argument("--component", default=default_component, help="prompt-компонент из model.yaml: first_touch | first_touch_hh.")
    p.add_argument("--golden", type=Path, default=default_golden, help="Курируемые golden-кейсы (input + expected_facts).")
    p.add_argument("--offline", action="store_true", help="Replay offline_message + эвристика вместо LLM-судьи (без сети).")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель LLM-судьи (по умолчанию {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=4, help="Параллельных воркеров (2 LLM-вызова на кейс).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызовов LLM (генерация/судья), сек.")
    p.add_argument("--checkpoint-every", type=int, default=20, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _ending_check(text: str, phrase: str = "С уважением") -> str:
    """Срезать прощальную подпись (как в легаси FirstTouchGenerator)."""
    lines = (text or "").split("\n")
    if len(lines) >= 2 and any(phrase.lower() in ln.lower() for ln in lines[-2:]):
        del lines[-2:]
    return "\n".join(lines).strip()


def _process(case: GoldenCase, gen_client: Any, judge: Any, offline: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"case": case, "message": None, "usage": None, "judge_usage": None,
                           "facts_present": {}, "hallucinated": [], "question_present": False,
                           "call_error": None, "judge_error": None}
    if offline:
        if case.offline_message is None:
            res["call_error"] = "no_offline_message"
            return res
        msg = case.offline_message.strip()
        res["message"] = msg
        res["facts_present"] = facts_present_heuristic(case.expected_facts, msg)
        res["question_present"] = "?" in msg
        return res

    payload = dict(case.input)
    if case.company_hidden:
        payload["hiring_company_name"] = ""  # прод прячет название; промпт опирается на company_description
    try:
        text, usage = gen_client.run(json.dumps(payload, ensure_ascii=False))
        res["message"], res["usage"] = _ending_check(text), usage
    except Exception as e:  # noqa: BLE001 — сбой генерации = инфра
        res["call_error"] = f"generate:{type(e).__name__}:{e}"
        return res
    if not res["message"]:
        res["call_error"] = "generate:empty_output"
        return res
    # legit-контекст из input (его дали генератору) — НЕ галлюцинации
    allowed = dict(case.allowed_context_facts)
    for k in ("company_description", "vacancy_responsibilities", "reason_of_communication"):
        v = str(case.input.get(k) or "").strip()
        if v:
            allowed[k] = v
    try:
        verdict, jusage = judge.evaluate(case.expected_facts, allowed, res["message"])
        res["judge_usage"] = jusage
        res["facts_present"] = verdict.facts_present
        res["hallucinated"] = verdict.hallucinated_facts
        res["question_present"] = verdict.question_present
    except Exception as e:  # noqa: BLE001 — сбой судьи = инфра
        res["judge_error"] = f"judge:{type(e).__name__}:{e}"
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")
    component = args.component

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, component, cli_id=args.prompt_id, cli_version=args.prompt_version)

    gen_client = judge = None
    eval_model = None
    if not args.offline:
        from qa_harness.core.llm_client import ModelClient, StoredPromptClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")
        llm = get_client(timeout=args.step1_timeout)
        gen_client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=llm,
                                        text_format={"format": {"type": "text"}})
        eval_model = args.eval_model
        judge = FactJudge(ModelClient(eval_model, timeout=args.step1_timeout))

    cases = load_golden(args.golden)

    rb = ReportBuilder(
        runner=component,
        prompt_under_test={"component": component, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": prompt.prompt_id, "evaluator": eval_model},
        seed=prompt.seed,
        args={"offline": bool(args.offline), "golden": len(cases), "workers": args.workers, "eval_model": eval_model},
    )

    usage_bucket = blank_usage()
    m, reasons = Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"first_touch": dict(m), "reasons": dict(reasons)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, component, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        case: GoldenCase = res["case"]
        accumulate_usage(usage_bucket, res["usage"])
        accumulate_usage(usage_bucket, res["judge_usage"])
        cid = f"golden:{case.name}:v1"

        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {case.name}: {res['call_error']}")
            return
        if res["judge_error"]:
            rb.add_error(cid, res["judge_error"])
            reasons["judge_error"] += 1
            m["judge_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {case.name}: {res['judge_error']}")
            return

        m["total"] += 1
        facts_present = res["facts_present"]
        required = [k for k in case.expected_facts if k not in case.optional_facts]
        missing_required = [k for k in required if not facts_present.get(k)]
        extra_nums = extra_numbers(list(case.expected_facts.values()), res["message"])
        question_ok = res["question_present"] if case.require_question else True
        company_leak = case.company_hidden and company_name_leaked(str(case.input.get("hiring_company_name") or ""), res["message"])
        forbidden_in_msg = forbidden_phrases(res["message"], case.forbid_in_message)
        hallucinated = res["hallucinated"] or []

        # hallucinated — СИГНАЛ, не gate: LLM-судья шумит на общих фразах (в легаси это было strict-only)
        quality_passed = (not missing_required) and (not extra_nums) and question_ok and (not company_leak) and (not forbidden_in_msg)
        rc: List[str] = []
        if missing_required:
            reasons["missing_facts"] += 1
            rc += [f"missing_fact:{k}" for k in missing_required]
        else:
            m["facts_pass"] += 1
        if extra_nums:
            reasons["extra_numbers"] += 1
            rc += [f"extra_number:{n}" for n in extra_nums]
        else:
            m["no_extra_numbers"] += 1
        if question_ok:
            m["question_pass"] += 1
        else:
            reasons["no_question"] += 1
            rc.append("no_question")
        if company_leak:
            reasons["company_leak"] += 1
            rc.append("company_name_leaked")
        else:
            m["company_hidden_ok"] += 1
        if forbidden_in_msg:
            reasons["forbidden_in_message"] += 1
            rc += [f"forbidden_in_message:{p}" for p in forbidden_in_msg]
        else:
            m["no_forbidden_phrases"] += 1
        if hallucinated:
            reasons["hallucination"] += 1
            m["hallucination_total"] += len(hallucinated)
        else:
            m["no_hallucination"] += 1

        checks = [
            {"rule": "facts_required", "passed": not missing_required, "detail": ",".join(missing_required)},
            {"rule": "no_extra_numbers", "passed": not extra_nums, "detail": ",".join(extra_nums)},
            {"rule": "question", "passed": bool(question_ok), "detail": ""},
            {"rule": "company_hidden", "passed": not company_leak,
             "detail": str(case.input.get("hiring_company_name") or "") if company_leak else ""},
            {"rule": "no_forbidden_phrases", "passed": not forbidden_in_msg, "detail": ",".join(forbidden_in_msg)},
            {"rule": "no_hallucination(info)", "passed": not hallucinated, "detail": "; ".join(hallucinated[:5])},
        ]
        rb.add_case(CaseRecord(
            case_id=cid, source="golden", passed=quality_passed,
            inputs={"criterion": "facts present & no hallucination & no extra numbers & question & company_hidden",
                    "input": case.input, "expected_facts": case.expected_facts,
                    "company_hidden": case.company_hidden, "require_question": case.require_question},
            output={"raw": res["message"]},
            verdict={"evaluator": "first_touch_llm_judge" if eval_model else "first_touch_heuristic",
                     "model": eval_model, "passed": quality_passed, "reason_codes": rc},
            checks=checks,
        ))
        if not args.quiet:
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {case.name} "
                  f"facts_missing={len(missing_required)} halluc={len(hallucinated)} "
                  f"extra_nums={len(extra_nums)} q={question_ok}")

    total = len(cases)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        cases,
        work=lambda c: _process(c, gen_client, judge, args.offline),
        fold=_fold,
        max_workers=max(1, args.workers),
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_flush,
        on_interrupt=_on_interrupt,
    )

    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        import json as _json
        s = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        mode = "offline" if args.offline else "online"
        print(f"[{tag}] {mode} quality_cases={s['total']} passed={s['passed']} failed={s['failed']} "
              f"errors(infra)={s['errors']} done={outcome.done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Раннер first_touch_event: приглашение на фиксированное мероприятие (VK JT Go).

Промпт `first_touch_event_invite` из payload `{candidate_name}` генерит приглашение на ОДНО известное
мероприятие. Без бэкенда. Оценка кейса:
- EventJudge (отдельная модель) сверяет с эталоном: missing_facts / hallucinated_facts / forbidden_claims
  (время/цена/спикеры/день недели);
- эвристики: greeting «Имя, здравствуйте!», финальный вопрос про ссылку/регистрацию (≤14 слов), extra_numbers
  (разрешено только 4 = «4 апреля»).

Итог (passed) = greeting & final_question & нет missing & нет hallucinated & нет forbidden & нет extra_numbers.
quality ≠ infra: сбой генерации/судьи → errors. `--offline` реплеит offline_message + эвристика фактов (без судьи).

  python -m qa_harness.runners.first_touch_event --offline
  python -m qa_harness.runners.first_touch_event
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
from qa_harness.domain.first_touch_event import (
    EventJudge,
    GoldenCase,
    extra_numbers,
    facts_present_heuristic,
    final_question_ok,
    greeting_ok,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "first_touch_event" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "first_touch_event"
PROMPT_COMPONENT = "first_touch_event_invite"
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="first_touch_event QA runner (generate invite + event LLM-judge).")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Курируемые golden-кейсы (имена кандидатов).")
    p.add_argument("--offline", action="store_true", help="Replay offline_message + эвристика вместо судьи (без сети).")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель event-судьи (по умолчанию {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=4, help="Параллельных воркеров (2 LLM-вызова на кейс).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызовов LLM, сек.")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _process(case: GoldenCase, gen_client: Any, judge: Any, offline: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"case": case, "message": None, "usage": None, "judge_usage": None,
                           "missing": [], "hallucinated": [], "forbidden": [], "comment": "",
                           "call_error": None, "judge_error": None}
    if offline:
        if case.offline_message is None:
            res["call_error"] = "no_offline_message"
            return res
        msg = case.offline_message.strip()
        res["message"] = msg
        res["missing"] = facts_present_heuristic(msg)
        return res
    try:
        text, usage = gen_client.run(json.dumps({"candidate_name": case.candidate_name}, ensure_ascii=False))
        res["message"], res["usage"] = (text or "").strip(), usage
    except Exception as e:  # noqa: BLE001
        res["call_error"] = f"generate:{type(e).__name__}:{e}"
        return res
    if not res["message"]:
        res["call_error"] = "generate:empty_output"
        return res
    try:
        verdict, jusage = judge.evaluate(res["message"])
        res["judge_usage"] = jusage
        res["missing"] = verdict.missing_facts
        res["hallucinated"] = verdict.hallucinated_facts
        res["forbidden"] = verdict.forbidden_claims
        res["comment"] = verdict.comment
    except Exception as e:  # noqa: BLE001
        res["judge_error"] = f"judge:{type(e).__name__}:{e}"
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, PROMPT_COMPONENT, cli_id=args.prompt_id, cli_version=args.prompt_version)

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
        judge = EventJudge(ModelClient(eval_model, timeout=args.step1_timeout))

    cases = load_golden(args.golden)

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
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
        extra: Dict[str, Any] = {"first_touch_event": dict(m), "reasons": dict(reasons)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

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
        msg = res["message"]
        greeting = greeting_ok(msg, case.candidate_name)
        final_q = final_question_ok(msg)
        extra = extra_numbers(msg)
        missing, hallucinated, forbidden = res["missing"], res["hallucinated"], res["forbidden"]

        quality_passed = greeting and final_q and (not missing) and (not hallucinated) and (not forbidden) and (not extra)
        rc: List[str] = []
        if greeting:
            m["greeting_ok"] += 1
        else:
            reasons["bad_greeting"] += 1
            rc.append("bad_greeting")
        if final_q:
            m["final_question_ok"] += 1
        else:
            reasons["bad_final_question"] += 1
            rc.append("bad_final_question")
        if missing:
            reasons["missing_facts"] += 1
            rc += [f"missing:{x}" for x in missing[:6]]
        else:
            m["facts_pass"] += 1
        if hallucinated:
            reasons["hallucinated"] += 1
            rc += [f"hallucinated:{x}" for x in hallucinated[:5]]
        else:
            m["no_hallucination"] += 1
        if forbidden:
            reasons["forbidden_claims"] += 1
            rc += [f"forbidden:{x}" for x in forbidden[:5]]
        else:
            m["no_forbidden_claims"] += 1
        if extra:
            reasons["extra_numbers"] += 1
            rc += [f"extra_number:{n}" for n in extra]
        else:
            m["no_extra_numbers"] += 1

        checks = [
            {"rule": "greeting", "passed": bool(greeting), "detail": ""},
            {"rule": "final_question", "passed": bool(final_q), "detail": ""},
            {"rule": "facts_present", "passed": not missing, "detail": ",".join(missing[:6])},
            {"rule": "no_hallucination", "passed": not hallucinated, "detail": "; ".join(hallucinated[:5])},
            {"rule": "no_forbidden_claims", "passed": not forbidden, "detail": "; ".join(forbidden[:5])},
            {"rule": "no_extra_numbers", "passed": not extra, "detail": ",".join(str(n) for n in extra)},
        ]
        rb.add_case(CaseRecord(
            case_id=cid, source="golden", passed=quality_passed,
            inputs={"criterion": "greeting & final_question & facts & no_hallucination & no_forbidden & no_extra_numbers",
                    "candidate_name": case.candidate_name},
            output={"raw": msg},
            verdict={"evaluator": "first_touch_event_llm_judge" if eval_model else "first_touch_event_heuristic",
                     "model": eval_model, "passed": quality_passed, "reason_codes": rc,
                     "comment": res["comment"]},
            checks=checks,
        ))
        if not args.quiet:
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {case.name} greeting={int(greeting)} "
                  f"final_q={int(final_q)} missing={len(missing)} halluc={len(hallucinated)} "
                  f"forbidden={len(forbidden)} extra={len(extra)}")

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

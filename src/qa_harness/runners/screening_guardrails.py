"""Раннер screening_guardrails: гоняет ЖИВОЙ screening_assistant в мультитёрн-разговоре и ловит нарушения.

Промпт-под-тестом — `screening_assistant`. На курируемых разговорах (реплики кандидата) ассистент отвечает
по ходам (Conversations API), и каждый его ответ проверяется на 3 гардрейла:
- self_answer (пишет за кандидата), repeated_questions (повтор одного вопроса), premature_end (вопрос +
  тут же закрывает диалог). Онлайн — LLM-судья (+ heuristic-фолбэк); `--offline` — эвристики по replay.

Итог кейса (passed) = НИ ОДИН ответ ассистента не нарушил гардрейлы. quality ≠ infra: сбой разговора → errors.

  python -m qa_harness.runners.screening_guardrails --offline
  python -m qa_harness.runners.screening_guardrails
"""

from __future__ import annotations

import argparse
import datetime
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, run_cases, usage_total
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.screening import ScreeningConversation
from qa_harness.domain.screening_guardrails import (
    GoldenCase,
    GuardrailJudge,
    has_questions_in_reply,
    heuristic_premature_end,
    heuristic_repeated_questions,
    heuristic_self_answer,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "screening_guardrails" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_guardrails"
PROMPT_COMPONENT = "screening_assistant"
DEFAULT_EVAL_MODEL = "gpt-4.1"
DEFAULT_RECRUITER_NAME = "Анна"
DEFAULT_VACANCY_INFO: Dict[str, Any] = {
    "title": "Python Backend Developer",
    "company_name": "ExampleSoft",
    "responsibilities": "Поддержка и развитие микросервисов, интеграции с продуктами.",
    "work_format": "remote",
    "location": "Москва",
    "min_salary": 200000,
    "max_salary": 280000,
    "company_info": {
        "firm_description": "Продуктовая команда, развивающая b2b-платформу.",
        "vacancy_url": "https://example.com/vacancies/python-backend",
    },
    "questions": (
        "- Расскажите про опыт с Python и современными фреймворками?\n"
        "- Какие сервисы поддерживали под высокой нагрузкой?\n"
        "- Как часто используете SQL и для каких задач?"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="screening_guardrails QA runner (live screening_assistant multi-turn).")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--offline", action="store_true", help="Replay offline_turns + эвристики (без сети/судьи).")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель LLM-судьи (по умолчанию {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--prompt-id", default=None, help="Override screening_assistant prompt_id.")
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=2, help="Параллельных разговоров (каждый — несколько ходов).")
    p.add_argument("--step1-timeout", type=int, default=90, help="Таймаут вызовов LLM, сек.")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _heuristic_turn(candidate: str, reply: str, end: bool) -> Dict[str, Any]:
    has_q = has_questions_in_reply(reply)
    sa, _s = heuristic_self_answer(reply)
    rq, _r, topics = heuristic_repeated_questions(reply)
    pe, _p = heuristic_premature_end(reply, end)
    if not has_q:
        pe = False
    return {"candidate": candidate, "reply": reply, "end": end, "self_answer": sa, "repeated": rq,
            "premature": pe, "topics": topics, "comment": "heuristic", "used_heuristics": True,
            "usage": None, "judge_usage": None}


def _process(case: GoldenCase, client: Any, prompt: Any, judge: Any, offline: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"case": case, "turns": [], "call_error": None}
    if offline:
        for ot in case.offline_turns:
            res["turns"].append(_heuristic_turn(ot.candidate, ot.assistant_reply, ot.conversation_end))
        return res
    try:
        conv = ScreeningConversation(client, prompt.prompt_id, prompt.prompt_version,
                                     DEFAULT_VACANCY_INFO, DEFAULT_RECRUITER_NAME, case.candidate_name)
        conv.start()
    except Exception as e:  # noqa: BLE001
        res["call_error"] = f"conversation:{type(e).__name__}:{e}"
        return res
    for turn in case.candidate_turns:
        try:
            result = conv.respond(turn)
        except Exception as e:  # noqa: BLE001
            res["call_error"] = f"assistant:{type(e).__name__}:{e}"
            return res
        reply = str(result.response or "")
        verdict, jusage = judge.evaluate(turn, reply, result.conversation_end)
        res["turns"].append({"candidate": turn, "reply": reply, "end": result.conversation_end,
                             "self_answer": verdict.self_answer, "repeated": verdict.repeated_questions,
                             "premature": verdict.premature_end, "topics": verdict.repeated_topics,
                             "comment": verdict.comment, "used_heuristics": verdict.used_heuristics,
                             "usage": result.usage, "judge_usage": jusage})
        if result.conversation_end:
            break
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, PROMPT_COMPONENT, cli_id=args.prompt_id, cli_version=args.prompt_version)

    client = judge = None
    eval_model = None
    if not args.offline:
        from qa_harness.core.llm_client import ModelClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")
        client = get_client(timeout=args.step1_timeout)
        eval_model = args.eval_model
        judge = GuardrailJudge(ModelClient(eval_model, timeout=args.step1_timeout))

    cases = load_golden(args.golden)

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": PROMPT_COMPONENT, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
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
        extra: Dict[str, Any] = {"guardrails": dict(m), "reasons": dict(reasons)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        case: GoldenCase = res["case"]
        for t in res["turns"]:
            accumulate_usage(usage_bucket, t.get("usage"))
            accumulate_usage(usage_bucket, t.get("judge_usage"))
        cid = f"golden:{case.name}:v1"

        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {case.name}: {res['call_error']}")
            return
        if not res["turns"]:
            rb.add_error(cid, "no_turns")
            reasons["no_turns"] += 1
            return

        m["conversations"] += 1
        transcript: List[Dict[str, Any]] = []
        rc: List[str] = []
        violating = False
        used_heur_any = False
        for i, t in enumerate(res["turns"], start=1):
            transcript.append({"turn": 2 * i - 1, "role": "candidate", "text": t["candidate"]})
            flags = {"self_answer": bool(t["self_answer"]), "repeated_questions": bool(t["repeated"]),
                     "premature_end_after_questions": bool(t["premature"])}
            transcript.append({"turn": 2 * i, "role": "assistant", "text": t["reply"], "flags": flags})
            m["turns_total"] += 1
            if t["used_heuristics"]:
                used_heur_any = True
            if t["self_answer"]:
                m["self_answer"] += 1
                reasons["self_answer"] += 1
                rc.append(f"t{i}:self_answer")
                violating = True
            if t["repeated"]:
                m["repeated_questions"] += 1
                reasons["repeated_questions"] += 1
                rc.append(f"t{i}:repeated_questions")
                violating = True
            if t["premature"]:
                m["premature_end"] += 1
                reasons["premature_end"] += 1
                rc.append(f"t{i}:premature_end")
                violating = True

        passed = not violating
        if passed:
            m["passed_conversations"] += 1
        rb.add_case(CaseRecord(
            case_id=cid, source="suite", passed=passed,
            inputs={"criterion": "no self_answer/repeated_questions/premature_end in assistant turns",
                    "scenario": {"candidate_name": case.candidate_name}},
            transcript=transcript,
            verdict={"evaluator": "screening_guardrails_heuristic" if args.offline else "screening_guardrails_llm_judge",
                     "model": eval_model, "passed": passed, "reason_codes": rc,
                     "meta": {"used_heuristics": used_heur_any}},
        ))
        if not args.quiet:
            print(f"  [{'ok ' if passed else 'MISS'}] {case.name} turns={len(res['turns'])} "
                  f"violations={len(rc)}{' (heur)' if used_heur_any else ''}")

    total = len(cases)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        cases,
        work=lambda c: _process(c, client, prompt, judge, args.offline),
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
        print(f"[{tag}] {mode} conversations={s['total']} passed={s['passed']} failed={s['failed']} "
              f"errors(infra)={s['errors']} done={outcome.done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

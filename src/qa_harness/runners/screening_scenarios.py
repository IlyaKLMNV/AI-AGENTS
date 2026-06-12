"""Раннер screening_scenarios: сценарии из CSV → живой screening_assistant → LLM-судья против ожидания.

Промпт-под-тестом задаётся `--component` (`screening_assistant` или `screening_assistant_hh` — у каждого
свой CSV golden). Для каждого сценария берём реплики кандидата из примеров, гоняем ассистента мультитёрн
(Conversations API) и ScenarioJudge решает, отработал ли он как описано в `expected_behavior` сценария.
Заменяет ~4000 строк hardcoded-эвристик легаси LLM-судьёй.

Итог кейса (passed) = поведение ассистента соответствует ожидаемому. quality ≠ infra: сбой разговора/судьи
→ errors; сценарий без примеров диалога → skip (пробел golden, не сбой). Онлайн гоняет только сценарии с
примерами кандидата (в base CSV — 7 из 62; в hh CSV примеров пока нет). `--scenario-indices 1,7,12` —
точечно; `--sample 0` — все с примерами. `--offline` — плумбинг: грузит CSV и извлекает реплики, без сети.

  python -m qa_harness.runners.screening_scenarios --offline
  python -m qa_harness.runners.screening_scenarios --sample 5
  python -m qa_harness.runners.screening_scenarios --scenario-indices 1,7,12
  python -m qa_harness.runners.screening_scenarios --component screening_assistant_hh --offline
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, run_cases, usage_total
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.screening import ScreeningConversation
from qa_harness.domain.screening_scenarios import (
    END_MARKER,
    Scenario,
    ScenarioJudge,
    extract_candidate_examples,
    load_scenarios,
    parse_scenario_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_scenarios"
DEFAULT_COMPONENT = "screening_assistant"
# Дефолтный CSV golden по компоненту промпта (hh — отдельный набор сценариев HeadHunter).
COMPONENT_CSV = {
    "screening_assistant": FIXTURES / "screening_scenarios.csv",
    "screening_assistant_hh": FIXTURES / "screening_scenarios_hh.csv",
}
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
    "company_info": {"firm_description": "Продуктовая команда b2b-платформы.",
                     "vacancy_url": "https://example.com/vacancies/python-backend"},
    "questions": "- Опыт с Python и фреймворками?\n- Сервисы под нагрузкой?\n- Как используете SQL?",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="screening_scenarios QA runner (CSV scenarios + LLM judge vs expected).")
    p.add_argument("--component", default=DEFAULT_COMPONENT, choices=sorted(COMPONENT_CSV),
                   help="Промпт-под-тестом из model.yaml (hh — вариант HeadHunter).")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV сценариев (golden). По умолчанию — по --component.")
    p.add_argument("--sample", type=int, default=5, help="Случайная выборка N сценариев (0 = все ~90).")
    p.add_argument("--scenario-indices", default=None, help="Точечные номера строк CSV, напр. 1,7,12 (override --sample).")
    p.add_argument("--max-examples", type=int, default=4, help="Сколько реплик кандидата брать из примеров на сценарий.")
    p.add_argument("--offline", action="store_true", help="Плумбинг: грузим CSV + извлекаем реплики, без сети.")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель ScenarioJudge (по умолчанию {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=3, help="Параллельных сценариев (каждый — живой разговор).")
    p.add_argument("--step1-timeout", type=int, default=90)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _select(scenarios: List[Scenario], indices_raw: str, sample: int, seed: Any,
            max_examples: int, runnable_only: bool) -> List[Scenario]:
    if indices_raw:  # точечный выбор — честим как просили, фильтр не применяем
        wanted = parse_scenario_indices(indices_raw)
        by_idx = {s.index: s for s in scenarios}
        return [by_idx[i] for i in wanted if i in by_idx]
    pool = scenarios
    if runnable_only:  # онлайн: гонять можно лишь сценарии с примерами диалогов кандидата
        pool = [s for s in scenarios if extract_candidate_examples(s.examples_raw, max_examples)]
    if sample and sample > 0:
        rng = random.Random(seed)
        return rng.sample(pool, min(sample, len(pool)))
    return pool


def _process(scenario: Scenario, client: Any, prompt: Any, judge: Any, max_examples: int) -> Dict[str, Any]:
    res: Dict[str, Any] = {"scenario": scenario, "turns": [], "verdict": None, "judge_usage": None, "call_error": None}
    candidate_turns = extract_candidate_examples(scenario.examples_raw, max_examples)
    if not candidate_turns:
        res["call_error"] = "no_candidate_examples"
        return res
    try:
        conv = ScreeningConversation(client, prompt.prompt_id, prompt.prompt_version,
                                     DEFAULT_VACANCY_INFO, DEFAULT_RECRUITER_NAME, "Кандидат")
        conv.start()
    except Exception as e:  # noqa: BLE001
        res["call_error"] = f"conversation:{type(e).__name__}:{e}"
        return res
    for turn in candidate_turns:
        try:
            result = conv.respond(turn)
        except Exception as e:  # noqa: BLE001
            res["call_error"] = f"assistant:{type(e).__name__}:{e}"
            return res
        res["turns"].append({"candidate": turn, "reply": str(result.response or ""),
                             "end": result.conversation_end, "usage": result.usage})
        if result.conversation_end:
            break
    def _fmt_turn(t: Dict[str, Any]) -> str:
        reply = str(t["reply"] or "")
        if t["end"]:  # ассистент завершил диалог — служебный END в текст не попадает, метим маркером для судьи
            reply = (reply + " " + END_MARKER).strip()
        return f"[Кандидат] {t['candidate']}\n[Ассистент] {reply}"

    transcript_text = "\n".join(_fmt_turn(t) for t in res["turns"])
    try:
        verdict, jusage = judge.evaluate(scenario, transcript_text)
        res["verdict"], res["judge_usage"] = verdict, jusage
    except Exception as e:  # noqa: BLE001
        res["call_error"] = f"judge:{type(e).__name__}:{e}"
    return res


def _run_offline(args: argparse.Namespace, scenarios: List[Scenario], prompt: Any, run_id: str, started) -> Dict[str, Path]:
    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": args.component, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id, started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None}, seed=prompt.seed,
        args={"offline": True, "component": args.component, "scenarios": len(scenarios)},
    )
    rb.set_token_usage({"input": 0, "output": 0, "total": 0})
    with_ex = turns = 0
    for s in scenarios:
        ct = extract_candidate_examples(s.examples_raw, args.max_examples)
        if ct:
            with_ex += 1
            turns += len(ct)
        elif not args.quiet:
            print(f"  [skip] {s.index}:{s.name[:50]} — нет реплик кандидата")
    finished = datetime.datetime.now()
    md, cd = rb.finalize({"scenarios": {"loaded": len(scenarios), "with_candidate_examples": with_ex,
                                        "candidate_turns_total": turns}},
                         finished_at=finished.isoformat(timespec="seconds"),
                         duration_s=round((finished - started).total_seconds(), 3))
    mp, cpth = write_reports(args.out_dir, RUNNER, run_id, md, cd)
    if not args.quiet:
        print(f"[offline] scenarios={len(scenarios)} with_examples={with_ex} candidate_turns={turns} (плумбинг, без судьи)")
        print(f"[done] metrics -> {mp}")
    return {"metrics": mp, "cases": cpth}


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, args.component, cli_id=args.prompt_id, cli_version=args.prompt_version)
    seed = args.seed if args.seed is not None else prompt.seed
    csv_path = args.csv or COMPONENT_CSV[args.component]
    scenarios = _select(load_scenarios(csv_path), args.scenario_indices, args.sample, seed,
                        args.max_examples, runnable_only=not args.offline)

    if args.offline:
        return _run_offline(args, scenarios, prompt, run_id, started)

    from qa_harness.core.llm_client import ModelClient, get_client

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")
    client = get_client(timeout=args.step1_timeout)
    eval_model = args.eval_model
    judge = ScenarioJudge(ModelClient(eval_model, timeout=args.step1_timeout))

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": args.component, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id, started_at=started.isoformat(timespec="seconds"),
        models={"generator": prompt.prompt_id, "evaluator": eval_model}, seed=seed,
        args={"offline": False, "component": args.component, "scenarios": len(scenarios), "workers": args.workers,
              "max_examples": args.max_examples, "eval_model": eval_model},
    )

    usage_bucket = blank_usage()
    m, reasons = Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"scenarios": dict(m), "reasons": dict(reasons)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        s: Scenario = res["scenario"]
        for t in res["turns"]:
            accumulate_usage(usage_bucket, t.get("usage"))
        accumulate_usage(usage_bucket, res["judge_usage"])
        cid = f"scenario:{s.index}:{s.name[:40]}"

        if res["call_error"] == "no_candidate_examples":  # golden-пробел, не инфра-сбой
            m["skipped_no_examples"] += 1
            if not args.quiet:
                print(f"  [skip] {s.index} {s.name[:45]}: нет реплик кандидата")
            return
        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["errors"] += 1
            if not args.quiet:
                print(f"  [ERR ] {s.index} {s.name[:45]}: {res['call_error']}")
            return

        m["total"] += 1
        verdict = res["verdict"]
        transcript: List[Dict[str, Any]] = []
        for i, t in enumerate(res["turns"], start=1):
            transcript.append({"turn": 2 * i - 1, "role": "candidate", "text": t["candidate"]})
            a_turn: Dict[str, Any] = {"turn": 2 * i, "role": "assistant", "text": t["reply"]}
            if t["end"]:
                a_turn["ended"] = True  # ассистент завершил диалог (токен END / фильтр)
            transcript.append(a_turn)
        passed = bool(verdict.passed)
        if passed:
            m["passed"] += 1
        else:
            m["failed"] += 1
            for v in verdict.violations[:6]:
                reasons[v[:60]] += 1
        rb.add_case(CaseRecord(
            case_id=cid, source="suite", passed=passed,
            inputs={"criterion": s.expected_behavior or "expected behavior per scenario",
                    "scenario": {"index": s.index, "name": s.name, "description": s.description}},
            transcript=transcript,
            verdict={"evaluator": "screening_scenario_llm_judge", "model": eval_model, "passed": passed,
                     "reason_codes": verdict.violations[:8], "comment": verdict.comment},
        ))
        if not args.quiet:
            print(f"  [{'ok ' if passed else 'MISS'}] {s.index} {s.name[:45]} turns={len(res['turns'])} "
                  f"viol={len(verdict.violations)}")

    total = len(scenarios)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        scenarios,
        work=lambda sc: _process(sc, client, prompt, judge, args.max_examples),
        fold=_fold,
        max_workers=max(1, args.workers),
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_flush,
        on_interrupt=_on_interrupt,
    )

    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        import json as _json
        sm = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        print(f"[{tag}] online scenarios={sm['total']} passed={sm['passed']} failed={sm['failed']} "
              f"errors(infra)={sm['errors']} done={outcome.done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Раннер responsibilities_parser: текст вакансии → ключевые термины, оценка контракт + golden-семантика.

Что тестируем — промпт `responsibilities_parser`: на вход текст вакансии, на выход JSON-массив СТРОК
(1..5 ключевых требований/навыков, каждый 1..3 слова). Бэкенда НЕТ — только LLM, поэтому быстро/дёшево.
Кейс:
- contract (форма): 1..5 терминов, каждый 1..3 слова / без чисел-одиночек/запятых / ≤60, без дублей;
- semantic (golden): ожидаемые термины извлечены (expect ИЛИ-группы), запрещённые — нет (forbid);
- grounding (СИГНАЛ, не gate): найдены ли термины в тексте вакансии (возможная галлюцинация).

Итог кейса (passed) = contract & semantic. Сетевой сбой промпта → errors (не failed); невалидный
JSON-вывод → contract-фейл (качество). quality ≠ infra.

  python -m qa_harness.runners.responsibilities_parser --offline     # replay, без сети
  python -m qa_harness.runners.responsibilities_parser               # онлайн, реальный промпт (OPENAI_API_KEY)
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
from qa_harness.domain.responsibilities import (
    GoldenCase,
    check_contract,
    check_semantics,
    grounding_misses,
    load_golden,
    parse_keywords,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "responsibilities_parser" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "responsibilities_parser"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="responsibilities_parser QA runner (contract + golden semantic).")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Курируемые golden-вакансии с ожиданиями.")
    p.add_argument("--offline", action="store_true", help="Replay offline_output из golden (без сети).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=6, help="Параллельных воркеров (I/O-bound LLM-вызовы).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызова промпта, сек.")
    p.add_argument("--checkpoint-every", type=int, default=20, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _process(case: GoldenCase, client: Any, offline: bool) -> Dict[str, Any]:
    """Вакансия → ключевые слова (или ошибка). Сеть внутри; структурный результат."""
    res: Dict[str, Any] = {"case": case, "predicted": None, "raw": None,
                           "call_error": None, "parse_error": None, "usage": None}
    if offline:
        if case.offline_output is None:
            res["call_error"] = "no_offline_output"
        else:
            res["predicted"] = list(case.offline_output)
        return res
    try:
        raw, usage = client.run(case.vacancy)
        res["raw"], res["usage"] = raw, usage
    except Exception as e:  # noqa: BLE001 — сетевой/HTTP сбой = инфра
        res["call_error"] = f"parser:{type(e).__name__}:{e}"
        return res
    try:
        res["predicted"] = parse_keywords(raw)
    except ValueError as e:
        res["parse_error"] = f"invalid_json_output:{e}"
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)

    client = None
    if not args.offline:
        from qa_harness.core.llm_client import StoredPromptClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (промпт requires it)")
        client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=get_client(timeout=args.step1_timeout))

    cases = load_golden(args.golden)

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None},
        seed=prompt.seed,
        args={"offline": bool(args.offline), "golden": len(cases), "workers": args.workers},
    )

    usage_bucket = blank_usage()
    m, reasons = Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"responsibilities": dict(m), "reasons": dict(reasons)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        case: GoldenCase = res["case"]
        accumulate_usage(usage_bucket, res["usage"])
        cid = f"golden:{case.name}:v1"

        if res["call_error"]:                       # сетевой сбой / нет replay -> инфра
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {case.name}: {res['call_error']}")
            return

        m["total"] += 1
        rc: List[str] = []
        if res["parse_error"]:                      # не JSON-массив -> контракт-фейл (качество)
            contract_ok, contract_issues = False, ["invalid_json_output"]
            semantic_ok, sem_diffs, grounding, predicted = False, ["no_output"], [], []
            reasons["invalid_json_output"] += 1
            rc = ["invalid_json_output"]
        else:
            predicted = res["predicted"]
            contract_ok, contract_issues, _details = check_contract(predicted)
            semantic_ok, sem_diffs = check_semantics(predicted, case.expect, case.forbid)
            grounding = grounding_misses(predicted, case.vacancy)
            m["keywords_total"] += len(predicted)
            if contract_ok:
                m["contract_pass"] += 1
            else:
                reasons["contract_fail"] += 1
                rc += [f"contract:{i}" for i in contract_issues]
            if semantic_ok:
                m["semantic_pass"] += 1
            else:
                reasons["semantic_fail"] += 1
                rc += [f"semantic:{d}" for d in sem_diffs[:8]]
            if grounding:                           # СИГНАЛ (не gate)
                m["grounding_miss_cases"] += 1
                m["grounding_misses_total"] += len(grounding)

        quality_passed = bool(contract_ok) and bool(semantic_ok)
        checks = [
            {"rule": "format", "passed": bool(contract_ok), "detail": ",".join(contract_issues)},
            {"rule": "semantic", "passed": bool(semantic_ok), "detail": ",".join(sem_diffs[:8])},
            {"rule": "grounding(info)", "passed": not grounding, "detail": ",".join(grounding[:5])},
        ]
        rb.add_case(CaseRecord(
            case_id=cid, source="golden", passed=quality_passed,
            inputs={"criterion": "format(1..5 терминов, 1..3 слова) + semantic(golden expect/forbid)",
                    "vacancy": case.vacancy, "expect": case.expect, "forbid": case.forbid},
            output={"raw": res["raw"], "parsed": predicted},
            verdict={"evaluator": "responsibilities_quality", "passed": quality_passed, "reason_codes": rc},
            checks=checks,
        ))
        if not args.quiet:
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {case.name} "
                  f"contract={int(bool(contract_ok))} semantic={semantic_ok} "
                  f"keywords={len(predicted)} not_in_text={len(grounding)}")

    total = len(cases)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        cases,
        work=lambda c: _process(c, client, args.offline),
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

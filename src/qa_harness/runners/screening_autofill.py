"""Раннер screening_autofill: диалог → форма скрининга, оценка контракт + golden-ожидания + анти-утечка.

Что тестируем — промпт `screening_autofill`: на вход диалог рекрутер/кандидат (расплющенный в одну строку,
как в проде), на выход JSON-форма {preferred_location, min_salary, max_salary, work_format, additional_info}.
Бэкенда НЕТ (только LLM). Кейс:
- schema (форма): ключи/типы/enum work_format/digits зарплат/форма additional_info;
- expect (golden): ожидаемые поля совпали (work_format точным значением; зарплата/локация — `<nonempty>`);
- no_leak: в additional_info нет запрещённых тем (salary/location/work_format) и меток спикера;
- additional_info_nonempty (если кейс требует): доп. вопрос был — список не пуст.

Итог (passed) = schema & expect & no_leak & (ai_nonempty?). quality ≠ infra: сетевой сбой → errors;
невалидный JSON-вывод → schema-фейл (качество).

  python -m qa_harness.runners.screening_autofill --offline     # replay, без сети
  python -m qa_harness.runners.screening_autofill               # онлайн, реальный промпт (OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, run_cases, usage_total
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.screening_autofill import (
    GoldenCase,
    additional_info_leaks,
    check_expect,
    load_golden,
    parse_form,
    validate_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "screening_autofill" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_autofill"
AUTOFILL_INSTRUCTION = "Fill the screening form based on the dialogue below."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="screening_autofill QA runner (schema + golden expect + no-leak).")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Курируемые golden-диалоги с ожиданиями.")
    p.add_argument("--offline", action="store_true", help="Replay offline_output из golden (без сети).")
    p.add_argument("--no-flatten", action="store_true", help="НЕ расплющивать диалог в одну строку (по умолчанию плющим, как прод).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=6, help="Параллельных воркеров (I/O-bound LLM-вызовы).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызова промпта, сек.")
    p.add_argument("--checkpoint-every", type=int, default=20, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _flatten(dialogue: str) -> str:
    return re.sub(r"\s+", " ", (dialogue or "").replace("\r\n", "\n").replace("\r", "\n")).strip()


def _process(case: GoldenCase, client: Any, offline: bool, flatten: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"case": case, "parsed": None, "raw": None,
                           "call_error": None, "parse_error": None, "usage": None}
    if offline:
        if case.offline_output is None:
            res["call_error"] = "no_offline_output"
        else:
            res["parsed"] = case.offline_output
        return res
    dialogue = _flatten(case.dialogue) if flatten else case.dialogue
    try:
        raw, usage = client.run(f"{AUTOFILL_INSTRUCTION}\n\n{dialogue}")
        res["raw"], res["usage"] = raw, usage
    except Exception as e:  # noqa: BLE001 — сетевой/HTTP сбой = инфра
        res["call_error"] = f"autofill:{type(e).__name__}:{e}"
        return res
    try:
        res["parsed"] = parse_form(raw)
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
    flatten = not args.no_flatten

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None},
        seed=prompt.seed,
        args={"offline": bool(args.offline), "golden": len(cases), "workers": args.workers, "flatten": flatten},
    )

    usage_bucket = blank_usage()
    m, reasons = Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"autofill": dict(m), "reasons": dict(reasons)}
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

        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {case.name}: {res['call_error']}")
            return

        m["total"] += 1
        rc: List[str] = []
        if res["parse_error"]:
            parsed = {}
            schema_ok, schema_errors = False, ["invalid_json_output"]
            expect_ok, expect_diffs = False, ["no_output"]
            leaks, ai_ok = [], False
            reasons["invalid_json_output"] += 1
            rc = ["invalid_json_output"]
        else:
            parsed = res["parsed"]
            schema_errors = validate_schema(parsed)
            schema_ok = not schema_errors
            expect_ok, expect_diffs = check_expect(parsed, case.expect)
            leaks = additional_info_leaks(parsed, case.forbid_in_additional_info)
            ai = parsed.get("additional_info") if isinstance(parsed, dict) else None
            ai_ok = (isinstance(ai, list) and len(ai) > 0) if case.expect_additional_info_nonempty else True
            if schema_ok:
                m["schema_pass"] += 1
            else:
                reasons["schema_fail"] += 1
                rc += [f"schema:{e}" for e in schema_errors[:6]]
            if expect_ok:
                m["expect_pass"] += 1
            else:
                reasons["expect_fail"] += 1
                rc += [f"expect:{d}" for d in expect_diffs[:6]]
            if not leaks:
                m["no_leak_pass"] += 1
            else:
                reasons["leak"] += 1
                m["leak_total"] += len(leaks)
                rc += [f"leak:{x}" for x in leaks[:6]]
            if not ai_ok:
                reasons["additional_info_empty"] += 1
                rc.append("additional_info_empty")

        quality_passed = bool(schema_ok) and bool(expect_ok) and (not leaks) and bool(ai_ok)
        checks = [
            {"rule": "schema", "passed": bool(schema_ok), "detail": ",".join(schema_errors[:6])},
            {"rule": "expect", "passed": bool(expect_ok), "detail": ",".join(expect_diffs[:6])},
            {"rule": "no_leak", "passed": not leaks, "detail": ",".join(leaks[:6])},
        ]
        if case.expect_additional_info_nonempty:
            checks.append({"rule": "additional_info_nonempty", "passed": bool(ai_ok), "detail": ""})

        rb.add_case(CaseRecord(
            case_id=cid, source="golden", passed=quality_passed,
            inputs={"criterion": "schema(форма) + expect(golden) + no_leak(additional_info)",
                    "dialogue": case.dialogue, "expect": case.expect,
                    "forbid_in_additional_info": case.forbid_in_additional_info},
            output={"raw": res["raw"], "parsed": parsed},
            verdict={"evaluator": "screening_autofill_quality", "passed": quality_passed, "reason_codes": rc},
            checks=checks,
        ))
        if not args.quiet:
            wf = parsed.get("work_format") if isinstance(parsed, dict) else None
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {case.name} "
                  f"schema={int(bool(schema_ok))} expect={expect_ok} leaks={len(leaks)} wf={wf!r}")

    total = len(cases)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        cases,
        work=lambda c: _process(c, client, args.offline, flatten),
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

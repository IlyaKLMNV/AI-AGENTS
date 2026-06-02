"""Тонкий раннер extractor_agent — конвейер step1 (LLM parse) -> step2 (payload) -> step3 (backend).

Структурно другой раннер: НЕТ LLM-судьи, оценка контрактная; каждый кейс в отчёте — это
`stages[]` (step1_parse / step2_payload / step3_backend) с артефактом и pass/fail на каждом шаге.

  python -m qa_harness.runners.extractor_agent --steps 1                  # только промпт (нужен OPENAI_API_KEY)
  python -m qa_harness.runners.extractor_agent --steps 1,2,3 ...          # + backend (нужны AI_SEARCH_*)
"""

from __future__ import annotations

import argparse
import datetime
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, usage_total
from qa_harness.core.jsonio import safe_json_loads
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.pipeline import (
    BackendCfg,
    PromptCfg,
    build_step3_payload,
    call_backend_search_bool,
    call_openai_step1,
    make_base_payload,
    validate_step1_contract,
)
from qa_harness.pipeline.cases import build_suite_cases, build_synthetic_cases, load_cases_from_dir, parse_steps

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_DIR = REPO_ROOT / "tests" / "fixtures" / "extractor_agent"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
DEFAULT_MODEL = "gpt-4.1-mini"
RUNNER = "extractor_agent"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="extractor_agent QA runner (step1/2/3 pipeline, new architecture).")
    p.add_argument("--steps", default="1,2,3", help="Какие шаги гонять: 1 | 1,2 | 1,2,3.")
    p.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    p.add_argument("--cases-count", type=int, default=None, help="Сколько real-кейсов взять (по умолчанию все).")
    p.add_argument("--suite-count", type=int, default=0, help="Сколько встроенных suite-кейсов добавить.")
    p.add_argument("--synthetic-count", type=int, default=0, help="Сколько синтетических деградированных запросов.")
    p.add_argument("--mix-seed", type=int, default=42)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Модель для step1.")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    # backend (step3)
    p.add_argument("--base-url", default=None, help="AI search base url (или env AI_SEARCH_BASE_URL).")
    p.add_argument("--token", default=None, help="AI search token (или env AI_SEARCH_AUTH_TOKEN).")
    p.add_argument("--step3-path", default="/site/searchBool")
    p.add_argument("--timeout-s", type=int, default=30)
    p.add_argument("--step3-retries", type=int, default=1)
    p.add_argument("--token-in-body", action="store_true")
    p.add_argument("--no-sanitize-office-geo", action="store_true")
    # search flags
    p.add_argument("--only-russian", action="store_true")
    p.add_argument("--only-english", action="store_true")
    p.add_argument("--only-with-contacts", action="store_true")
    p.add_argument("--only-with-higher-education", action="store_true")
    p.add_argument("--current-position-title", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--highlight", action="store_true")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _gather_cases(args) -> List:
    real = load_cases_from_dir(args.cases_dir)
    if args.cases_count:
        real = real[: args.cases_count]
    suite = build_suite_cases(args.suite_count, args.mix_seed) if args.suite_count else []
    syn = build_synthetic_cases(real + suite, args.synthetic_count, args.mix_seed) if args.synthetic_count else []
    return real + suite + syn


def run(args: argparse.Namespace) -> Dict[str, Path]:
    steps = parse_steps(args.steps)
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set (step1 requires it)")

    backend: Optional[BackendCfg] = None
    token = ""
    if 3 in steps:
        base_url = args.base_url or os.environ.get("AI_SEARCH_BASE_URL")
        token = args.token or os.environ.get("AI_SEARCH_AUTH_TOKEN") or ""
        if not base_url or not token:
            raise EnvironmentError("--steps включает 3, но нет AI_SEARCH_BASE_URL/AI_SEARCH_AUTH_TOKEN (или --base-url/--token)")
        backend = BackendCfg(
            base_url=base_url, step3_path=args.step3_path, token_in_body=bool(args.token_in_body),
            timeout_s=args.timeout_s, retries=args.step3_retries,
        )

    base_payload = make_base_payload(
        only_russian=args.only_russian, only_english=args.only_english,
        only_with_contacts=args.only_with_contacts, only_with_higher_education=args.only_with_higher_education,
        current_position_title=args.current_position_title, limit=args.limit, offset=args.offset,
        shuffle=args.shuffle, highlight=args.highlight,
    )
    sanitize_geo = not args.no_sanitize_office_geo
    prompt_cfg = PromptCfg(prompt_id=prompt.prompt_id, prompt_version=prompt.prompt_version, model=args.model)

    cases = _gather_cases(args)
    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None},
        seed=args.mix_seed,
        args={"steps": steps, "cases": len(cases)},
    )

    usage_bucket = blank_usage()
    reason_hist: Counter = Counter()
    backend_stats = Counter()
    by_source: Dict[str, Dict[str, int]] = {}

    for case in cases:
        stages: List[Dict[str, Any]] = []
        reason_codes: List[str] = []
        extractor_json: Optional[dict] = None

        text, usage, err = call_openai_step1(api_key, prompt_cfg, case.input, timeout_s=args.timeout_s)
        accumulate_usage(usage_bucket, usage)

        if err:
            step1_ok = False
            reason_codes.append("step1_call_error")
            reason_hist["step1_call_error"] += 1
        else:
            obj, jerr = safe_json_loads(text or "", lenient=True)
            if jerr or not isinstance(obj, dict):
                step1_ok = False
                reason_codes.append("invalid_json")
                reason_hist["invalid_json"] += 1
            else:
                extractor_json = obj
                ok, errors, _warnings = validate_step1_contract(obj, case.input)
                step1_ok = ok
                if not ok:
                    reason_codes.extend(errors)
                    reason_hist["contract_fail"] += 1
        stages.append({"name": "step1_parse", "artifact": extractor_json, "passed": step1_ok, "reason_codes": list(reason_codes)})

        overall_ok = step1_ok
        payload: Optional[dict] = None
        if 2 in steps and step1_ok and extractor_json is not None:
            payload = build_step3_payload(extractor_json, case.input, base_payload, sanitize_geo)
            stages.append({"name": "step2_payload", "artifact": payload, "passed": True})

        if 3 in steps and backend is not None and payload is not None:
            kind, status, attempts, count, berr, _json = call_backend_search_bool(backend, token, payload)
            step3_ok = kind in ("success", "insufficient_search_terms")
            backend_stats["step3_calls"] += 1
            backend_stats[kind] += 1
            if kind == "success" and isinstance(count, int) and count > 0:
                backend_stats["retrieval_ok"] += 1
            s3_reasons = [] if kind == "success" else [kind]
            stages.append({
                "name": "step3_backend",
                "artifact": {"kind": kind, "status": status, "count": count},
                "passed": step3_ok,
                "reason_codes": s3_reasons,
            })
            if not step3_ok:
                reason_codes.append(kind)
                reason_hist[kind] += 1
            overall_ok = overall_ok and step3_ok

        src = case.source
        by_source.setdefault(src, {"total": 0, "passed": 0})
        by_source[src]["total"] += 1
        if overall_ok:
            by_source[src]["passed"] += 1

        rb.add_case(
            CaseRecord(
                case_id=f"{src}:{case.name}:v1",
                source=src,
                passed=overall_ok,
                inputs={"criterion": "step1 contract valid" + (" + step3 retrieval ok" if 3 in steps else ""), "query": case.input},
                output={"raw": text, "parsed": extractor_json},
                verdict={"evaluator": "pipeline", "passed": overall_ok, "reason_codes": reason_codes},
                stages=stages,
            )
        )
        if not args.quiet:
            print(f"  [{'ok' if overall_ok else 'MISS'}] {src}:{case.name} step1_ok={step1_ok}")

    rb.set_token_usage(usage_total(usage_bucket))
    metrics_extra: Dict[str, Any] = {"deterministic": dict(reason_hist), "by_source": by_source}
    if 3 in steps:
        metrics_extra["backend"] = dict(backend_stats)

    finished = datetime.datetime.now()
    metrics_doc, cases_doc = rb.finalize(
        metrics_extra,
        finished_at=finished.isoformat(timespec="seconds"),
        duration_s=round((finished - started).total_seconds(), 3),
    )
    metrics_path, cases_path = write_reports(args.out_dir, RUNNER, run_id, metrics_doc, cases_doc)

    if not args.quiet:
        s = metrics_doc["summary"]
        print(f"[summary] steps={steps} total={s['total']} passed={s['passed']} failed={s['failed']} pass_rate={s['pass_rate']}%")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

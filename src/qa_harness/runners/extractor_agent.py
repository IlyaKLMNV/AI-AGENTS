"""Раннер extractor_agent (пересобран): конвейер step1->step2->step3, оценка ПОЭТАПНО.

Ключевые принципы новой модели:
- кейсы — курируемые якоря с golden-ожиданиями (tests/fixtures/extractor_agent/anchors.yaml);
- каждый шаг оценивается своим критерием:
    step1 = contract (форма) + semantic (golden) + format (голый ли JSON);
    step2 = mapping integrity (ничего не потеряно при сборке payload);
    step3 = retrieval (success/insufficient/count) — ИНФОРМАЦИЯ, не pass/fail промпта;
- итог кейса (passed) = ТОЛЬКО качество промпта (contract & semantic & mapping);
- инфра-сбои (step1 сеть, step3 auth/timeout/http) идут в errors, а НЕ в failed;
- step1 через core StoredPromptClient (SDK, ретраи/бэкофф), раздельные таймауты step1/step3;
- конкурентность (пул потоков) + fail-fast по бэкенду + чекпоинты + сохранение при Ctrl+C.

  python -m qa_harness.runners.extractor_agent --steps 1                 # только промпт (OPENAI_API_KEY)
  python -m qa_harness.runners.extractor_agent --steps 1,2,3 ...         # + backend (AI_SEARCH_*)
"""

from __future__ import annotations

import argparse
import datetime
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_harness.core import (
    accumulate_usage,
    add_prompt_source_args,
    blank_usage,
    load_cfg,
    make_prompt_client,
    prompt_under_test_meta,
    resolve_prompt,
    resolve_source,
    run_cases,
    usage_total,
)
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.extractor import check_semantics
from qa_harness.pipeline import (
    BackendCfg,
    build_step3_payload,
    call_backend_search_bool,
    make_base_payload,
    mapping_report,
    parse_extractor_json,
    validate_step1_contract,
)
from qa_harness.pipeline.cases import Anchor, load_anchors, parse_steps

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANCHORS = REPO_ROOT / "tests" / "fixtures" / "extractor_agent" / "anchors.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "extractor_agent"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="extractor_agent QA runner (per-stage, new architecture).")
    p.add_argument("--steps", default="1,2,3", help="Какие шаги гонять: 1 | 1,2 | 1,2,3.")
    p.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS, help="Курируемые якоря с golden.")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    add_prompt_source_args(p)
    p.add_argument("--workers", type=int, default=6, help="Параллельных воркеров (I/O-bound).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызова step1 (LLM), сек.")
    p.add_argument("--step3-timeout", type=int, default=45, help="Таймаут вызова step3 (backend), сек.")
    p.add_argument("--backend-fail-fast", type=int, default=8, help="Стоп step3 после N инфра-ошибок подряд по бэкенду.")
    p.add_argument("--checkpoint-every", type=int, default=20, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    # backend (step3)
    p.add_argument("--base-url", default=None, help="AI search base url (или env AI_SEARCH_BASE_URL).")
    p.add_argument("--token", default=None, help="AI search token (или env AI_SEARCH_AUTH_TOKEN).")
    p.add_argument("--step3-path", default="/site/searchBool")
    p.add_argument("--step3-retries", type=int, default=1)
    p.add_argument("--token-in-body", action="store_true", help="Слать токен в теле (нужно для hlebusheck-бэкенда).")
    p.add_argument("--no-sanitize-office-geo", action="store_true")
    # search flags
    p.add_argument("--only-russian", action="store_true")
    p.add_argument("--only-english", action="store_true")
    p.add_argument("--only-with-contacts", action="store_true")
    p.add_argument("--only-with-higher-education", action="store_true")
    p.add_argument("--current-position-title", action="store_true")
    p.add_argument("--step3-limit", type=int, default=0,
                   help="limit профилей в step3: 0 = только count (быстро, ~сек). Профили для QA не нужны; "
                        "большой limit заставляет backend отдавать тяжёлую выдачу минутами.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _process(
    anchor: Anchor,
    client: Any,
    steps: List[int],
    base_payload: Dict[str, Any],
    sanitize_geo: bool,
    backend: Optional[BackendCfg],
    token: str,
    backend_down: threading.Event,
) -> Dict[str, Any]:
    """Полный конвейер для одного якоря. Сеть внутри; возвращает структурный результат."""
    res: Dict[str, Any] = {
        "anchor": anchor, "text": None, "extractor_json": None, "usage": None,
        "step1_status": None, "contract_ok": False, "contract_errors": [], "contract_warnings": [],
        "semantic_ok": None, "semantic_diffs": [], "mapping_ok": None, "mapping": None,
        "payload": None, "step3": None, "infra_error": None, "skipped_step3": False,
    }
    try:
        text, usage = client.run(anchor.input)
        res["text"], res["usage"] = text, usage
    except Exception as e:  # noqa: BLE001 — сетевой/HTTP сбой step1 = инфра
        res["step1_status"] = "call_error"
        res["infra_error"] = f"step1:{type(e).__name__}:{e}"
        return res

    obj, status = parse_extractor_json(text)
    res["step1_status"] = status
    if obj is None:
        return res  # invalid_json -> качество fail, дальше нечего

    res["extractor_json"] = obj
    ok, errors, warnings = validate_step1_contract(obj, anchor.input)
    res["contract_ok"], res["contract_errors"], res["contract_warnings"] = ok, errors, warnings
    sem_ok, diffs = check_semantics(obj, anchor.expect, anchor.forbid)
    res["semantic_ok"], res["semantic_diffs"] = sem_ok, diffs

    if 2 in steps:
        payload = build_step3_payload(obj, anchor.input, base_payload, sanitize_geo)
        rep = mapping_report(obj, payload)
        res["payload"], res["mapping"] = payload, rep
        # dropped = тихая потеря данных => fail; unmapped_fields (напр. business_spheres,
        # который payload пока не использует) => warning, качество не валит.
        res["mapping_ok"] = not rep["dropped"]

    if 3 in steps and backend is not None and res["payload"] is not None:
        if backend_down.is_set():
            res["skipped_step3"] = True
        else:
            kind, status_code, _att, count, berr, _json = call_backend_search_bool(backend, token, res["payload"])
            res["step3"] = {"kind": kind, "status": status_code, "count": count}
            if kind not in ("success", "insufficient_search_terms"):
                res["infra_error"] = f"step3:{kind}:{berr or status_code}"
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    steps = parse_steps(args.steps)
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    source = resolve_source(args.prompt_source)

    from qa_harness.core.llm_client import get_client

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set (step1 requires it)")
    client = make_prompt_client(prompt, source=source, local_version=args.local_prompt_version,
                                prompts_path=args.prompts_path, client=get_client(timeout=args.step1_timeout))

    backend: Optional[BackendCfg] = None
    token = ""
    if 3 in steps:
        base_url = args.base_url or os.environ.get("AI_SEARCH_BASE_URL")
        token = args.token or os.environ.get("AI_SEARCH_AUTH_TOKEN") or ""
        if not base_url or not token:
            raise EnvironmentError("--steps включает 3, но нет AI_SEARCH_BASE_URL/AI_SEARCH_AUTH_TOKEN (или --base-url/--token)")
        backend = BackendCfg(
            base_url=base_url, step3_path=args.step3_path, token_in_body=bool(args.token_in_body),
            timeout_s=args.step3_timeout, retries=args.step3_retries,
        )

    base_payload = make_base_payload(
        only_russian=args.only_russian, only_english=args.only_english,
        only_with_contacts=args.only_with_contacts, only_with_higher_education=args.only_with_higher_education,
        current_position_title=args.current_position_title, limit=args.step3_limit, offset=args.offset,
    )
    sanitize_geo = not args.no_sanitize_office_geo
    anchors = load_anchors(args.anchors)

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test=prompt_under_test_meta(prompt, source, args.local_prompt_version),
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None},
        seed=prompt.seed,
        args={"steps": steps, "anchors": len(anchors), "workers": args.workers},
    )

    usage_bucket = blank_usage()
    m1, m2, m3 = Counter(), Counter(), Counter()
    reasons: Counter = Counter()
    backend_down = threading.Event()
    state = {"infra_step3": 0}

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        metrics_extra: Dict[str, Any] = {"step1": dict(m1), "reasons": dict(reasons)}
        if 2 in steps:
            metrics_extra["step2"] = dict(m2)
        if 3 in steps:
            metrics_extra["step3"] = dict(m3)
        if interrupted:
            metrics_extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(metrics_extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        anchor: Anchor = res["anchor"]
        accumulate_usage(usage_bucket, res["usage"])
        cid = f"anchor:{anchor.name}:v1"

        # step1 сетевой сбой -> инфра, НЕ кейс качества
        if res["step1_status"] == "call_error":
            rb.add_error(cid, res["infra_error"])
            reasons["step1_call_error"] += 1
            m1["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {anchor.name}: {res['infra_error']}")
            return

        m1["total"] += 1
        rc: List[str] = []
        if res["step1_status"] == "invalid":
            reasons["invalid_json"] += 1
            m1["invalid_json"] += 1
            rc.append("invalid_json")
        elif res["step1_status"] == "dirty":
            reasons["output_not_bare_json"] += 1
            m1["dirty_output"] += 1
            rc.append("output_not_bare_json")  # warning, не валит качество

        contract_ok = res["contract_ok"]
        semantic_ok = res["semantic_ok"]
        mapping_ok = res["mapping_ok"]
        if contract_ok:
            m1["contract_pass"] += 1
        elif res["extractor_json"] is not None:
            reasons["contract_fail"] += 1
            rc += [f"contract:{e}" for e in res["contract_errors"][:6]]
        if semantic_ok is True:
            m1["semantic_pass"] += 1
        elif semantic_ok is False:
            reasons["semantic_fail"] += 1
            rc += [f"semantic:{d}" for d in res["semantic_diffs"][:6]]

        if 2 in steps and mapping_ok is not None:
            if mapping_ok:
                m2["mapping_pass"] += 1
            else:
                reasons["mapping_fail"] += 1
            rep = res["mapping"] or {}
            m2["dropped_total"] += len(rep.get("dropped", []))
            m2["sanitized_total"] += len(rep.get("sanitized", []))
            m2["unmapped_total"] += len(rep.get("unmapped_fields", []))
            rc += [f"dropped:{x}" for x in rep.get("dropped", [])[:4]]
            rc += [f"unmapped:{x}" for x in rep.get("unmapped_fields", [])]

        # итог качества: contract & semantic & mapping (None = шаг не гонялся -> не валит)
        quality_passed = bool(contract_ok) and (semantic_ok is not False) and (mapping_ok is not False)

        stages: List[Dict[str, Any]] = [
            {"name": "step1_parse", "artifact": res["extractor_json"], "passed": contract_ok,
             "reason_codes": (["invalid_json"] if res["step1_status"] == "invalid" else []) + res["contract_errors"][:6]},
        ]
        checks = [
            {"rule": "contract", "passed": bool(contract_ok), "detail": ",".join(res["contract_errors"][:6])},
            {"rule": "semantic", "passed": bool(semantic_ok), "detail": ",".join(res["semantic_diffs"][:8])},
            {"rule": "output_bare_json", "passed": res["step1_status"] != "dirty"},
        ]
        if 2 in steps and mapping_ok is not None:
            stages.append({"name": "step2_payload", "artifact": res["payload"], "passed": bool(mapping_ok),
                           "reason_codes": [f"dropped:{x}" for x in (res['mapping'] or {}).get('dropped', [])]
                                           + [f"unmapped:{x}" for x in (res['mapping'] or {}).get('unmapped_fields', [])]})
            checks.append({"rule": "mapping", "passed": bool(mapping_ok), "detail": str(res["mapping"])})

        # step3 — retrieval-инфо + инфра отдельно (НЕ гейтит quality)
        if 3 in steps:
            if res["skipped_step3"]:
                m3["skipped"] += 1
            elif res["step3"] is not None:
                kind = res["step3"]["kind"]
                count = res["step3"]["count"]
                m3[kind] += 1
                if kind == "success" and (count or 0) == 0:
                    m3["zero_count"] += 1
                stages.append({"name": "step3_backend", "artifact": res["step3"],
                               "passed": kind in ("success", "insufficient_search_terms"),
                               "reason_codes": [] if kind == "success" else [kind]})
                if res["infra_error"]:
                    m3["infra_errors"] += 1
                    reasons[f"step3:{kind}"] += 1
                    rb.add_error(cid, res["infra_error"])
                    state["infra_step3"] += 1
                    if state["infra_step3"] >= args.backend_fail_fast:
                        backend_down.set()

        rb.add_case(CaseRecord(
            case_id=cid, source="anchor", passed=quality_passed,
            inputs={"criterion": "contract + semantic(golden) + mapping integrity",
                    "query": anchor.input, "expect": anchor.expect, "forbid": anchor.forbid},
            output={"raw": res["text"], "parsed": res["extractor_json"]},
            verdict={"evaluator": "extractor_quality", "passed": quality_passed, "reason_codes": rc},
            checks=checks, stages=stages,
        ))
        if not args.quiet:
            s3 = f" step3={res['step3']['kind']}/{res['step3']['count']}" if res.get("step3") else ""
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {anchor.name} contract={int(bool(contract_ok))} semantic={semantic_ok} mapping={mapping_ok}{s3}")

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    # Оркестрация (пул/порядок/чекпоинты/Ctrl+C) — в core.run_loop; fail-fast (backend_down)
    # остаётся здесь как shared state между _process и _fold, циклу про него знать не нужно.
    outcome = run_cases(
        anchors,
        work=lambda a: _process(a, client, steps, base_payload, sanitize_geo, backend, token, backend_down),
        fold=_fold,
        max_workers=max(1, args.workers),
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_flush,
        on_interrupt=_on_interrupt,
    )
    interrupted = outcome.interrupted
    done, total = outcome.done, outcome.total

    metrics_path, cases_path = _flush(interrupted=interrupted)
    if not args.quiet:
        # summary берём из только что записанного metrics
        import json as _json
        s = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if interrupted else "summary"
        print(f"[{tag}] steps={steps} quality_cases={s['total']} passed={s['passed']} failed={s['failed']} errors(infra)={s['errors']} done={done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

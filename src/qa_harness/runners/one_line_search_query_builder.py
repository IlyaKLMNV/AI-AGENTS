"""Раннер one_line_search_query_builder: вакансия -> однострочный boolean-запрос, оценка ПОЭТАПНО.

Что тестируем — промпт-БИЛДЕР (one_line_search_query_builder): он превращает текст
вакансии в одну строку boolean-запроса. Конвейер кейса:
- step1 (builder, промпт-под-тестом): вакансия -> строка запроса;
    качество билдера = format (одна строка, не пусто, без JSON) + no_leakage (нет
    упоминаний формата работы/зарплаты/процесса найма) + semantic по golden (запрос
    содержит ожидаемые термины и не содержит запрещённые);
- step2 (extractor, downstream-ИНСТРУМЕНТ): запрос -> extractor_json -> payload —
    проверка контракта/маппинга, ИНФО (это другой промпт, билдера не валит);
- step3 (backend): searchBool -> count — retrieval-ИНФО.

Итог кейса (passed) = ТОЛЬКО качество билдера (format & no_leakage & semantic).
Инфра-сбои (builder сеть, extractor сеть, backend auth/timeout/http) идут в errors,
а НЕ в failed. Бэкенд-count — информация, не гейт (принцип quality≠infra, как у extractor).

  python -m qa_harness.runners.one_line_search_query_builder --offline          # без сети (replay)
  python -m qa_harness.runners.one_line_search_query_builder                      # step1: промпт билдера (OPENAI_API_KEY)
  python -m qa_harness.runners.one_line_search_query_builder --steps 1,2,3 --token-in-body  # + extractor + backend count
  python -m qa_harness.runners.one_line_search_query_builder --generate --variants 5   # вариативно: LLM-вакансия с засеянными core, только step1
"""

from __future__ import annotations

import argparse
import datetime
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import random

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
from qa_harness.domain.generators import (
    SOFT_NOISE,
    TECH_VOCAB,
    GenerationPolicy,
    ResponsibilitiesGenerator,
    ResponsibilitiesSpec,
    generate_valid,
)
from qa_harness.domain.query_builder import (
    GoldenCase,
    build_query_checks,
    check_query_semantics,
    detect_leakage,
    load_golden,
)
from qa_harness.pipeline import (
    BackendCfg,
    build_step3_payload,
    call_backend_search_bool,
    make_base_payload,
    mapping_report,
    parse_extractor_json,
    validate_step1_contract,
)
from qa_harness.pipeline.cases import parse_steps

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "one_line_search_query_builder" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "one_line_search_query_builder"
EXTRACTOR_COMPONENT = "extractor_agent"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="one_line_search_query_builder QA runner (per-stage, new architecture).")
    p.add_argument("--steps", default="1",
                   help="1 | 1,2 | 1,2,3. Качество билдера полностью оценивается на step1; "
                        "2 (extractor) и 3 (backend count) добавляют downstream-инфо.")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Курируемые golden-вакансии с ожиданиями.")
    p.add_argument("--offline", action="store_true", help="Replay offline_query из golden (без сети; принудительно steps=1).")
    # --- режим вариативной генерации (seeded-вакансия с известными терминами; принудительно steps=1) ---
    p.add_argument("--generate", action="store_true",
                   help="Вариативный режим: вакансию генерит LLM с засеянными core-терминами (expect из них), только step1.")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help=f"Модель генератора вакансии (по умолч. {DEFAULT_GEN_MODEL}).")
    p.add_argument("--gen-seed", type=int, default=None, help="Seed выборки домена/терминов.")
    p.add_argument("--variants", type=int, default=5, help="Сколько вакансий сгенерить (--generate).")
    p.add_argument("--core-terms", type=int, default=2, help="Сколько core-терминов засевать (ожидаем в запросе).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature генератора (--generate).")
    p.add_argument("--gen-retries", type=int, default=2, help="Повторов генерации при провале валидации.")
    p.add_argument("--prompt-id", default=None, help="Override prompt_id билдера.")
    p.add_argument("--prompt-version", default=None, help="Override prompt_version билдера.")
    p.add_argument("--extractor-prompt-id", default=None, help="Override prompt_id extractor (step2).")
    p.add_argument("--extractor-prompt-version", default=None, help="Override prompt_version extractor (step2).")
    add_prompt_source_args(p)
    p.add_argument("--workers", type=int, default=6, help="Параллельных воркеров (I/O-bound).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызовов LLM (builder/extractor), сек.")
    p.add_argument("--step3-timeout", type=int, default=45, help="Таймаут вызова backend (step3), сек.")
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
                   help="limit профилей в step3: 0 = только count (быстро). Профили для QA не нужны.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _process(
    case: GoldenCase,
    builder_client: Any,
    extractor_client: Any,
    steps: List[int],
    base_payload: Dict[str, Any],
    sanitize_geo: bool,
    backend: Optional[BackendCfg],
    token: str,
    backend_down: threading.Event,
    offline: bool,
) -> Dict[str, Any]:
    """Полный конвейер для одной golden-вакансии. Сеть внутри; возвращает структурный результат."""
    res: Dict[str, Any] = {
        "case": case, "query": "", "usage": None, "extractor_usage": None,
        "step1_status": None, "format_ok": False, "format_errors": [],
        "leakage": [], "semantic_ok": None, "semantic_diffs": [],
        "extractor_status": None, "extractor_json": None,
        "extractor_contract_ok": None, "extractor_contract_errors": [],
        "payload": None, "mapping": None, "step3": None,
        "skipped_step3": False, "infra_error": None,
    }

    # --- step1: builder -> строка запроса ---
    if offline:
        if case.offline_query is None:
            res["step1_status"] = "no_offline_query"
            res["infra_error"] = "builder:no_offline_query"
            return res
        res["query"] = case.offline_query.strip()
    else:
        try:
            text, usage = builder_client.run(case.vacancy)
            res["query"], res["usage"] = (text or "").strip(), usage
        except Exception as e:  # noqa: BLE001 — сетевой/HTTP сбой builder = инфра
            res["step1_status"] = "call_error"
            res["infra_error"] = f"builder:{type(e).__name__}:{e}"
            return res
    res["step1_status"] = "ok"
    query = res["query"]

    # качество билдера: формат + анти-утечка + семантика (golden)
    fmt = build_query_checks(query)
    res["format_ok"], res["format_errors"] = fmt["ok"], fmt["errors"]
    res["leakage"] = detect_leakage(query)
    sem_ok, diffs = check_query_semantics(query, case.expect, case.forbid)
    res["semantic_ok"], res["semantic_diffs"] = sem_ok, diffs

    # --- step2: extractor (downstream-инструмент, ИНФО) ---
    if 2 in steps and not offline:
        try:
            etext, eusage = extractor_client.run(query)
            res["extractor_usage"] = eusage
        except Exception as e:  # noqa: BLE001 — сбой extractor = инфра (билдера не валит)
            res["infra_error"] = f"extractor:{type(e).__name__}:{e}"
            return res
        obj, status = parse_extractor_json(etext)
        res["extractor_status"], res["extractor_json"] = status, obj
        if obj is not None:
            ok, errors, _warnings = validate_step1_contract(obj, query)
            res["extractor_contract_ok"], res["extractor_contract_errors"] = ok, errors
            payload = build_step3_payload(obj, query, base_payload, sanitize_geo)
            res["payload"], res["mapping"] = payload, mapping_report(obj, payload)

    # --- step3: backend count (ИНФО) ---
    if 3 in steps and not offline and backend is not None and res["payload"] is not None:
        if backend_down.is_set():
            res["skipped_step3"] = True
        else:
            kind, status_code, _att, count, berr, _json = call_backend_search_bool(backend, token, res["payload"])
            res["step3"] = {"kind": kind, "status": status_code, "count": count}
            if kind not in ("success", "insufficient_search_terms"):
                res["infra_error"] = f"step3:{kind}:{berr or status_code}"
    return res


def _process_generate(variant: int, *, builder_client: Any, base_payload: Dict[str, Any], sanitize_geo: bool,
                      backend_down: Any, gen_client: Any, gen_policy: GenerationPolicy,
                      gen_seed: int, core_n: int) -> Dict[str, Any]:
    """Один вариант: засеваем core/soft → LLM пишет вакансию → step1 билдера → судья expect(core)/forbid(soft)."""
    rng = random.Random(f"{gen_seed}:{variant}")
    domain = rng.choice(list(TECH_VOCAB))
    core = rng.sample(TECH_VOCAB[domain], min(core_n, len(TECH_VOCAB[domain])))
    soft = rng.sample(SOFT_NOISE, 2)
    gen = ResponsibilitiesGenerator(gen_client)
    gr = generate_valid(lambda _a: (gen.generate(ResponsibilitiesSpec(domain, core, soft, noise_level=variant % 3)), None),
                        policy=gen_policy)
    base = {"mode": "generate", "variant": variant, "gen_source": gr.source, "gen_usage": dict(gen.usage)}
    if not gr.ok:
        return {**base, "case": None, "query": "", "usage": None, "extractor_usage": None,
                "step1_status": "gen_failed", "infra_error": f"vacancy_gen_failed:{'; '.join(gr.errors[-2:]) or 'unknown'}"}
    case = GoldenCase(name=f"v{variant}_{domain}", vacancy=str(gr.item),
                      expect=[[t] for t in core], forbid=list(soft))
    res = _process(case, builder_client, None, [1], base_payload, sanitize_geo, None, "", backend_down, offline=False)
    res.update(base)
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    steps = [1] if (args.offline or args.generate) else parse_steps(args.steps)
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    builder = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    source = resolve_source(args.prompt_source)
    extractor = None
    if 2 in steps:
        extractor = resolve_prompt(
            cfg, EXTRACTOR_COMPONENT, cli_id=args.extractor_prompt_id, cli_version=args.extractor_prompt_version
        )

    builder_client = extractor_client = None
    if not args.offline:
        from qa_harness.core.llm_client import get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (step1 builder requires it)")
        llm = get_client(timeout=args.step1_timeout)
        builder_client = make_prompt_client(builder, source=source, local_version=args.local_prompt_version,
                                            prompts_path=args.prompts_path, client=llm)
        if extractor is not None:
            # helper extractor в local-режиме — на боевой версии из pointer.yaml (свой --local-prompt-version не пробрасываем)
            extractor_client = make_prompt_client(extractor, source=source, prompts_path=args.prompts_path, client=llm)

    backend: Optional[BackendCfg] = None
    token = ""
    if 3 in steps and not args.offline:
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
    backend_down = threading.Event()

    gen_setup: Dict[str, Any] = {}
    if args.generate:
        from qa_harness.core.llm_client import ModelClient

        gen_seed = args.gen_seed if args.gen_seed is not None else (builder.seed if builder.seed is not None else 0)
        gen_setup = dict(
            builder_client=builder_client, base_payload=base_payload, sanitize_geo=sanitize_geo,
            backend_down=backend_down,
            gen_client=ModelClient(args.gen_model, timeout=args.step1_timeout, temperature=args.temperature),
            gen_policy=GenerationPolicy(max_retries=args.gen_retries, temperature=args.temperature, seed=gen_seed),
            gen_seed=gen_seed, core_n=args.core_terms,
        )
        work_items: List[Any] = list(range(max(1, args.variants)))
        models = {"vacancy_generator": args.gen_model, "evaluator": None}
        run_args = {"mode": "generate", "steps": [1], "variants": args.variants, "core_terms": args.core_terms,
                    "gen_model": args.gen_model, "gen_seed": gen_seed, "temperature": args.temperature,
                    "workers": args.workers}
    else:
        work_items = list(load_golden(args.golden))
        models = {"generator": None, "evaluator": None}
        run_args = {
            "mode": "golden", "steps": steps, "golden": len(work_items), "workers": args.workers,
            "offline": bool(args.offline),
            "extractor_prompt": (
                {"prompt_id": extractor.prompt_id, "prompt_version": extractor.prompt_version} if extractor else None
            ),
        }

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test=prompt_under_test_meta(builder, source, args.local_prompt_version),
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models=models,
        seed=builder.seed,
        args=run_args,
    )

    usage_bucket = blank_usage()
    gen_usage_bucket = blank_usage()
    m1, m2, m3 = Counter(), Counter(), Counter()
    reasons: Counter = Counter()
    gen_sources: Counter = Counter()
    state = {"infra_step3": 0}

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        metrics_extra: Dict[str, Any] = {"builder": dict(m1), "reasons": dict(reasons)}
        if args.generate:
            metrics_extra["generation"] = {"usage": usage_total(gen_usage_bucket), "sources": dict(gen_sources)}
        if 2 in steps:
            metrics_extra["step2_extractor"] = dict(m2)
        if 3 in steps:
            metrics_extra["step3_backend"] = dict(m3)
        if interrupted:
            metrics_extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(metrics_extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        is_gen = res.get("mode") == "generate"
        accumulate_usage(usage_bucket, res["usage"])
        accumulate_usage(usage_bucket, res["extractor_usage"])
        accumulate_usage(usage_bucket, res.get("gen_usage"))
        accumulate_usage(gen_usage_bucket, res.get("gen_usage"))
        if res.get("gen_source"):
            gen_sources[res["gen_source"]] += 1
        if is_gen:
            label = f"v{res['variant']}"
            cid = f"vacancy:v{res['variant']}"
        else:
            label = res["case"].name
            cid = f"golden:{res['case'].name}:v1"

        # builder сетевой сбой / нет replay / провал генерации -> инфра, НЕ кейс качества
        if res["step1_status"] in ("call_error", "no_offline_query", "gen_failed"):
            rb.add_error(cid, res["infra_error"])
            reasons[res["step1_status"]] += 1
            m1["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {label}: {res['infra_error']}")
            return
        case: GoldenCase = res["case"]

        m1["total"] += 1
        rc: List[str] = []
        format_ok, leakage, semantic_ok = res["format_ok"], res["leakage"], res["semantic_ok"]
        if format_ok:
            m1["format_pass"] += 1
        else:
            reasons["format_fail"] += 1
            rc += [f"format:{e}" for e in res["format_errors"]]
        if not leakage:
            m1["leakage_clean"] += 1
        else:
            reasons["leakage"] += 1
            m1["leakage_total"] += len(leakage)
            rc += [f"leakage:{x}" for x in leakage]
        if semantic_ok:
            m1["semantic_pass"] += 1
        else:
            reasons["semantic_fail"] += 1
            rc += [f"semantic:{d}" for d in res["semantic_diffs"][:8]]

        quality_passed = bool(format_ok) and (not leakage) and bool(semantic_ok)

        checks = [
            {"rule": "format", "passed": bool(format_ok), "detail": ",".join(res["format_errors"])},
            {"rule": "no_leakage", "passed": not leakage, "detail": ",".join(leakage)},
            {"rule": "semantic", "passed": bool(semantic_ok), "detail": ",".join(res["semantic_diffs"][:8])},
        ]
        stages: List[Dict[str, Any]] = [
            {"name": "step1_builder", "artifact": res["query"], "passed": quality_passed, "reason_codes": rc},
        ]

        # step2 extractor — downstream-инфо (контракт/маппинг), билдера не гейтит
        if 2 in steps and res["extractor_status"] is not None:
            m2["total"] += 1
            ec_ok = res["extractor_contract_ok"]
            if res["extractor_status"] == "invalid":
                m2["invalid_json"] += 1
            if ec_ok:
                m2["contract_pass"] += 1
            rep = res["mapping"] or {}
            m2["dropped_total"] += len(rep.get("dropped", []))
            stages.append({
                "name": "step2_extractor", "artifact": res["payload"], "passed": bool(ec_ok),
                "reason_codes": (["invalid_json"] if res["extractor_status"] == "invalid" else [])
                + (res["extractor_contract_errors"] or [])[:6],
            })
            checks.append({"rule": "extractor_contract(info)", "passed": bool(ec_ok),
                           "detail": ",".join((res["extractor_contract_errors"] or [])[:6])})

        # step3 backend — retrieval-инфо
        if 3 in steps:
            if res["skipped_step3"]:
                m3["skipped"] += 1
            elif res["step3"] is not None:
                kind, count = res["step3"]["kind"], res["step3"]["count"]
                m3[kind] += 1
                if kind == "success" and (count or 0) == 0:
                    m3["zero_count"] += 1
                stages.append({"name": "step3_backend", "artifact": res["step3"],
                               "passed": kind in ("success", "insufficient_search_terms"),
                               "reason_codes": [] if kind == "success" else [kind]})

        # инфра step2/step3 -> errors (НЕ failed); fail-fast по бэкенду
        if res["infra_error"]:
            rb.add_error(cid, res["infra_error"])
            reasons[res["infra_error"].split(":")[0] + "_infra"] += 1
            if res["infra_error"].startswith("step3:"):
                m3["infra_errors"] += 1
                state["infra_step3"] += 1
                if state["infra_step3"] >= args.backend_fail_fast:
                    backend_down.set()

        inputs: Dict[str, Any] = {"criterion": "format + no_leakage + semantic(golden)",
                                  "vacancy": case.vacancy, "expect": case.expect, "forbid": case.forbid}
        if is_gen:
            inputs["variant"] = res["variant"]
            inputs["gen_source"] = res.get("gen_source")
        rb.add_case(CaseRecord(
            case_id=cid, source="synthetic" if is_gen else "golden", passed=quality_passed,
            inputs=inputs,
            output={"raw": res["query"]},
            verdict={"evaluator": "one_line_quality", "passed": quality_passed, "reason_codes": rc},
            checks=checks, stages=stages,
        ))
        if not args.quiet:
            s2 = f" step2={res['extractor_status']}" if (2 in steps and res["extractor_status"]) else ""
            s3 = f" step3={res['step3']['kind']}/{res['step3']['count']}" if res.get("step3") else ""
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {label} "
                  f"format={int(bool(format_ok))} leak={len(leakage)} semantic={semantic_ok}{s2}{s3}")

    total = len(work_items)

    if args.generate:
        def _work(it):
            return _process_generate(it, **gen_setup)
    else:
        def _work(c):
            return _process(c, builder_client, extractor_client, steps, base_payload,
                            sanitize_geo, backend, token, backend_down, args.offline)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        work_items,
        work=_work,
        fold=_fold,
        max_workers=max(1, args.workers),
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_flush,
        on_interrupt=_on_interrupt,
    )
    interrupted = outcome.interrupted
    done = outcome.done

    metrics_path, cases_path = _flush(interrupted=interrupted)
    if not args.quiet:
        import json as _json
        s = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if interrupted else "summary"
        mode = "offline" if args.offline else "online"
        print(f"[{tag}] {mode} steps={steps} quality_cases={s['total']} passed={s['passed']} "
              f"failed={s['failed']} errors(infra)={s['errors']} done={done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

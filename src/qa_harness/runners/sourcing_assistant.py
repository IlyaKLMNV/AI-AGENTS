"""Раннер sourcing_assistant: кандидат ↔ требования вакансии, оценка по КОНТРАКТУ вывода.

Что тестируем — промпт `sourcing_assistant`: на вход `{requirements:[...], profile:{...}}`, на выход
JSON-массив 1:1 к требованиям, по объекту `{requirement, comment, passed:0|1}` на каждое. Конвейер кейса:
- кейс = ВАКАНСИЯ; требования (1..5) берём из CDM (`key_requirements`/stack+skills);
- backend-поиск РЕАЛЬНЫХ кандидатов по `vacancy.extractor_entities` (limit=pool) → сэмпл N профилей;
- на каждого кандидата зовём промпт → проверяем КОНТРАКТ вывода (форма/длина/echo), это `subjects[]`.

Итог кейса (passed) = ВСЕ оценённые кандидаты прошли контракт. Семантики «реально ли подходит» нет
(кандидаты живые, без разметки). quality ≠ infra: backend-сбои / нет entities / нет кандидатов / сетевые
ошибки промпта → errors (кейс не в passed/failed). Бэкенд-профили (limit>0) — медленный путь, не раздуваем.

  python -m qa_harness.runners.sourcing_assistant --offline                         # replay, без сети
  python -m qa_harness.runners.sourcing_assistant --cases-count 1 --candidate-sample-size 2 --token-in-body  # дёшево, онлайн
  python -m qa_harness.runners.sourcing_assistant --generate --variants 5           # вариативно: LLM-профиль + засеянные requirements, БЕЗ backend

Режим `--generate`: профиль кандидата генерит LLM, requirements засеяны из словаря → промпт sourcing →
проверка КОНТРАКТА (1:1 echo). Backend НЕ нужен (кандидаты не ищутся, а генерятся) — снимает backend-зависимость.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, run_cases, usage_total
from qa_harness.core.cdm import load_cdm_files, load_json
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.generators import (
    TECH_VOCAB,
    CandidateProfileGenerator,
    CandidateProfileSpec,
    GenerationPolicy,
    generate_valid,
)
from qa_harness.domain.sourcing import (
    build_candidate_profile,
    check_contract,
    load_offline_cases,
    parse_sourcing_output,
    requirements_from_cdm,
)
from qa_harness.pipeline import BackendCfg, build_step3_payload, call_backend_search_bool, make_base_payload

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CDM_DIR = REPO_ROOT / "tests" / "fixtures" / "cdm" / "std"
DEFAULT_OFFLINE = REPO_ROOT / "tests" / "fixtures" / "sourcing_assistant" / "offline.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "sourcing_assistant"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sourcing_assistant QA runner (subjects[], contract, new architecture).")
    p.add_argument("--offline", action="store_true", help="Replay канонных кандидатов из фикстуры (без сети).")
    p.add_argument("--offline-fixture", type=Path, default=DEFAULT_OFFLINE)
    # --- режим вариативной генерации (LLM-профиль + засеянные requirements; BACKEND НЕ нужен) ---
    p.add_argument("--generate", action="store_true",
                   help="Вариативный режим: профиль кандидата генерит LLM, requirements засеяны; без backend.")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help=f"Модель генератора профиля (по умолч. {DEFAULT_GEN_MODEL}).")
    p.add_argument("--gen-seed", type=int, default=None, help="Seed выборки домена/требований.")
    p.add_argument("--variants", type=int, default=5, help="Сколько профилей сгенерить (--generate).")
    p.add_argument("--req-count", type=int, default=3, help="Сколько requirements засевать (echo-контракт).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature генератора профиля (--generate).")
    p.add_argument("--gen-retries", type=int, default=2, help="Повторов генерации профиля при провале валидации.")
    p.add_argument("--cdm-dir", type=Path, default=DEFAULT_CDM_DIR, help="CDM-вакансии (нужны extractor_entities + key_requirements).")
    p.add_argument("--cdm-count", type=int, default=None, help="Взять первые N CDM (по сортировке).")
    p.add_argument("--cases-count", type=int, default=None, help="Сэмплировать N вакансий из набора.")
    p.add_argument("--requirements-source", choices=["cdm_key_requirements", "stack_skills"], default="cdm_key_requirements")
    p.add_argument("--candidate-pool-size", type=int, default=10, help="limit профилей из backend на вакансию (limit>0 = медленно!).")
    p.add_argument("--candidate-sample-size", type=int, default=5, help="Сколько профилей оценить промптом.")
    p.add_argument("--sample-mode", choices=["first", "random"], default="first")
    p.add_argument("--count-only", action="store_true",
                   help="Только ЧИСЛО кандидатов на вакансию (limit=0, быстро, без таймаутов): профили НЕ тянем "
                        "и промпт НЕ зовём. Диагностика доступности, не тест качества.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=4, help="Параллельных вакансий (бэкенд-профили тяжёлые — не задирай).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызова промпта sourcing, сек.")
    p.add_argument("--step3-timeout", type=int, default=60, help="Таймаут backend-поиска (профили медленные), сек.")
    p.add_argument("--backend-fail-fast", type=int, default=5, help="Стоп после N инфра-ошибок подряд по бэкенду.")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    # backend
    p.add_argument("--base-url", default=None, help="AI search base url (или env AI_SEARCH_BASE_URL).")
    p.add_argument("--token", default=None, help="AI search token (или env AI_SEARCH_AUTH_TOKEN).")
    p.add_argument("--step3-path", default="/site/searchBool")
    p.add_argument("--step3-retries", type=int, default=1)
    p.add_argument("--token-in-body", action="store_true", help="Слать токен в теле (нужно для hlebusheck-бэкенда).")
    p.add_argument("--no-sanitize-office-geo", action="store_true")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _sample(profiles: List[Dict[str, Any]], n: int, mode: str, rng: random.Random) -> List[Dict[str, Any]]:
    clean = [p for p in profiles if isinstance(p, dict)]
    if n <= 0 or not clean:
        return []
    take = min(n, len(clean))
    if mode == "first" or take == len(clean):
        return clean[:take]
    return [clean[i] for i in sorted(rng.sample(range(len(clean)), take))]


def _process_online(
    idx_vacancy,
    sourcing_client: Any,
    requirements_source: str,
    base_payload: Dict[str, Any],
    sanitize_geo: bool,
    backend: BackendCfg,
    token: str,
    backend_down: threading.Event,
    sample_size: int,
    sample_mode: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    idx, cdm_path = idx_vacancy
    cdm = load_json(cdm_path)
    vacancy = cdm.get("vacancy") or {}
    name = Path(cdm_path).stem
    res: Dict[str, Any] = {"name": name, "title": str(vacancy.get("title") or ""),
                           "requirements": [], "case_infra": None, "candidates": []}

    requirements = requirements_from_cdm(vacancy, requirements_source)
    res["requirements"] = requirements
    if not requirements:
        res["case_infra"] = "no_requirements"
        return res

    title = str(vacancy.get("title") or "").strip()
    entities = vacancy.get("extractor_entities")
    if not title:
        res["case_infra"] = "no_title_in_cdm"
        return res
    if not isinstance(entities, dict):
        res["case_infra"] = "no_search_entities_in_cdm"
        return res

    payload = build_step3_payload(entities, title, base_payload, sanitize_office_geo=sanitize_geo)
    if not any(k in payload for k in ("positions", "skills", "keys")):
        res["case_infra"] = "no_search_entities_in_cdm"
        return res

    if backend_down.is_set():
        res["case_infra"] = "backend_down(skipped)"
        return res

    kind, status_code, _att, count, berr, response = call_backend_search_bool(backend, token, payload)
    if kind not in ("success", "insufficient_search_terms"):
        res["case_infra"] = f"backend:{kind}:{berr or status_code}"
        return res
    profiles = []
    if isinstance(response, dict) and isinstance(response.get("profiles"), list):
        profiles = [p for p in response["profiles"] if isinstance(p, dict)]
    if not profiles:
        res["case_infra"] = "no_candidates_found"
        return res

    rng = random.Random((seed or 0) * 100003 + idx)
    for cand in _sample(profiles, sample_size, sample_mode, rng):
        cand_id = str(cand.get("id") or cand.get("name") or "candidate")
        entry: Dict[str, Any] = {"id": cand_id, "predicted": None, "infra_error": None, "parse_error": None, "usage": None}
        profile = build_candidate_profile(cand)
        try:
            raw, usage = sourcing_client.run(json.dumps({"requirements": requirements, "profile": profile}, ensure_ascii=False))
            entry["usage"] = usage
        except Exception as e:  # noqa: BLE001 — сетевой/HTTP сбой = инфра
            entry["infra_error"] = f"sourcing:{type(e).__name__}:{e}"
            res["candidates"].append(entry)
            continue
        try:
            entry["predicted"] = parse_sourcing_output(raw)
        except ValueError as e:
            entry["parse_error"] = f"invalid_json_output:{e}"
        res["candidates"].append(entry)
    return res


def _process_offline(idx_case) -> Dict[str, Any]:
    _idx, case = idx_case
    res: Dict[str, Any] = {"name": case.name, "title": "", "requirements": list(case.requirements),
                           "case_infra": None, "candidates": []}
    for cand in case.candidates:
        res["candidates"].append({"id": cand.id, "predicted": list(cand.output),
                                  "infra_error": None, "parse_error": None, "usage": None})
    return res


def _process_generate(variant: int, *, sourcing_client: Any, gen_client: Any,
                      gen_policy: GenerationPolicy, gen_seed: int, req_count: int) -> Dict[str, Any]:
    """Один вариант: засеваем requirements + LLM-профиль кандидата → промпт sourcing → контракт (без backend)."""
    rng = random.Random(f"{gen_seed}:{variant}")
    domain = rng.choice(list(TECH_VOCAB))
    requirements = rng.sample(TECH_VOCAB[domain], min(req_count, len(TECH_VOCAB[domain])))
    res: Dict[str, Any] = {"name": f"v{variant}", "title": f"{domain} вакансия", "requirements": requirements,
                           "case_infra": None, "candidates": [], "mode": "generate", "variant": variant,
                           "gen_source": None, "gen_usage": None}
    gen = CandidateProfileGenerator(gen_client)
    gr = generate_valid(lambda _a: (gen.generate(CandidateProfileSpec(domain, requirements, noise_level=variant % 3)), None),
                        policy=gen_policy)
    res["gen_source"] = gr.source
    res["gen_usage"] = dict(gen.usage)
    if not gr.ok:
        res["case_infra"] = f"profile_gen_failed:{'; '.join(gr.errors[-2:]) or 'unknown'}"
        return res
    profile = gr.item
    entry: Dict[str, Any] = {"id": f"cand_v{variant}", "predicted": None, "infra_error": None,
                             "parse_error": None, "usage": None}
    try:
        raw, usage = sourcing_client.run(json.dumps({"requirements": requirements, "profile": profile}, ensure_ascii=False))
        entry["usage"] = usage
    except Exception as e:  # noqa: BLE001 — сетевой сбой промпта = инфра
        entry["infra_error"] = f"sourcing:{type(e).__name__}:{e}"
        res["candidates"].append(entry)
        return res
    try:
        entry["predicted"] = parse_sourcing_output(raw)
    except ValueError as e:
        entry["parse_error"] = f"invalid_json_output:{e}"
    res["candidates"].append(entry)
    return res


def _run_count_only(args, items, prompt, run_id, started, backend, token, sanitize_geo) -> Dict[str, Path]:
    """Диагностика: число кандидатов на вакансию (limit=0, быстро). Профили не тянем, промпт не зовём."""
    count_payload = make_base_payload(only_with_contacts=True, current_position_title=True, limit=0, offset=0)
    backend_down = threading.Event()
    counts: Dict[str, int] = {}
    errors: List[Any] = []
    state = {"infra": 0}

    def work(item):
        _idx, cdm_path = item
        vacancy = (load_json(cdm_path).get("vacancy") or {})
        name = Path(cdm_path).stem
        r: Dict[str, Any] = {"name": name, "count": None, "infra": None}
        title = str(vacancy.get("title") or "").strip()
        entities = vacancy.get("extractor_entities")
        if not title:
            r["infra"] = "no_title_in_cdm"
            return r
        if not isinstance(entities, dict):
            r["infra"] = "no_search_entities_in_cdm"
            return r
        payload = build_step3_payload(entities, title, count_payload, sanitize_office_geo=sanitize_geo)
        if not any(k in payload for k in ("positions", "skills", "keys")):
            r["infra"] = "no_search_entities_in_cdm"
            return r
        if backend_down.is_set():
            r["infra"] = "backend_down(skipped)"
            return r
        kind, status, _a, count, berr, _j = call_backend_search_bool(backend, token, payload)
        if kind not in ("success", "insufficient_search_terms"):
            r["infra"] = f"backend:{kind}:{berr or status}"
            return r
        r["count"] = int(count or 0)
        return r

    def fold(r):
        if r["infra"]:
            errors.append((r["name"], r["infra"]))
            if r["infra"].startswith("backend:"):
                state["infra"] += 1
                if state["infra"] >= args.backend_fail_fast:
                    backend_down.set()
            if not args.quiet:
                print(f"  [ERR ] {r['name']}: {r['infra']}")
        else:
            counts[r["name"]] = r["count"]
            if not args.quiet:
                print(f"  [count] {r['name']} count={r['count']}")

    outcome = run_cases(items, work=work, fold=fold, max_workers=max(1, args.workers), checkpoint_every=0,
                        on_interrupt=(lambda: None) if args.quiet else (lambda: print("\n[interrupted]")))

    vals = sorted(counts.values())
    nonzero = [v for v in vals if v > 0]
    search_counts = {
        "per_vacancy": counts, "vacancies": len(counts), "with_candidates": len(nonzero),
        "zero": len(vals) - len(nonzero), "sum": sum(vals),
        "min": (vals[0] if vals else 0), "max": (vals[-1] if vals else 0),
        "avg": round(sum(vals) / len(vals), 1) if vals else 0,
    }
    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id, started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None}, seed=prompt.seed,
        args={"count_only": True, "cases": len(items), "workers": args.workers},
    )
    rb.set_token_usage({"input": 0, "output": 0, "total": 0})
    for name, reason in errors:
        rb.add_error(f"cdm:{name}:v1", reason)
    finished = datetime.datetime.now()
    md, cd = rb.finalize({"search_counts": search_counts}, finished_at=finished.isoformat(timespec="seconds"),
                         duration_s=round((finished - started).total_seconds(), 3))
    metrics_path, cases_path = write_reports(args.out_dir, RUNNER, run_id, md, cd)
    if not args.quiet:
        sc = search_counts
        print(f"[count-only] vacancies={sc['vacancies']} with_candidates={sc['with_candidates']} zero={sc['zero']} "
              f"sum={sc['sum']} avg={sc['avg']} min={sc['min']} max={sc['max']} errors={len(errors)} done={outcome.done}/{len(items)}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    seed = args.seed if args.seed is not None else prompt.seed
    if args.count_only and args.offline:
        raise ValueError("--count-only требует онлайн-бэкенд (несовместим с --offline)")

    sourcing_client = None
    backend: Optional[BackendCfg] = None
    token = ""
    gen_setup: Dict[str, Any] = {}
    if args.generate:
        if args.offline or args.count_only:
            raise ValueError("--generate несовместим с --offline/--count-only")
        from qa_harness.core.llm_client import ModelClient, StoredPromptClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (sourcing prompt requires it)")
        sourcing_client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=get_client(timeout=args.step1_timeout))
        gen_seed = args.gen_seed if args.gen_seed is not None else (seed if seed is not None else 0)
        gen_setup = dict(
            sourcing_client=sourcing_client,
            gen_client=ModelClient(args.gen_model, timeout=args.step1_timeout, temperature=args.temperature),
            gen_policy=GenerationPolicy(max_retries=args.gen_retries, temperature=args.temperature, seed=gen_seed),
            gen_seed=gen_seed, req_count=args.req_count,
        )
        items = list(range(max(1, args.variants)))  # int-варианты; _process_generate берёт variant
        case_source = "synthetic"
    elif args.offline:
        items = list(enumerate(load_offline_cases(args.offline_fixture)))
        case_source = "suite"
    else:
        from qa_harness.core.llm_client import StoredPromptClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (sourcing prompt requires it)")
        sourcing_client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=get_client(timeout=args.step1_timeout))

        base_url = args.base_url or os.environ.get("AI_SEARCH_BASE_URL")
        token = args.token or os.environ.get("AI_SEARCH_AUTH_TOKEN") or ""
        if not base_url or not token:
            raise EnvironmentError("онлайн-режим требует AI_SEARCH_BASE_URL/AI_SEARCH_AUTH_TOKEN (или --base-url/--token)")
        backend = BackendCfg(base_url=base_url, step3_path=args.step3_path, token_in_body=bool(args.token_in_body),
                             timeout_s=args.step3_timeout, retries=args.step3_retries)

        paths = load_cdm_files(args.cdm_dir, args.cdm_count)
        if args.cases_count is not None and args.cases_count > 0:
            rng = random.Random(seed)
            paths = rng.sample(paths, k=min(args.cases_count, len(paths)))
        items = list(enumerate(paths))
        case_source = "cdm"

    base_payload = make_base_payload(only_with_contacts=True, current_position_title=True,
                                     limit=args.candidate_pool_size, offset=0)
    sanitize_geo = not args.no_sanitize_office_geo

    if args.count_only:
        return _run_count_only(args, items, prompt, run_id, started, backend, token, sanitize_geo)

    if args.generate:
        models = {"profile_generator": args.gen_model, "evaluator": None}
        run_args = {"mode": "generate", "variants": args.variants, "req_count": args.req_count,
                    "gen_model": args.gen_model, "gen_seed": gen_setup["gen_seed"],
                    "temperature": args.temperature, "workers": args.workers}
    else:
        models = {"generator": None, "evaluator": None}
        run_args = {"mode": "golden", "offline": bool(args.offline), "cases": len(items),
                    "requirements_source": args.requirements_source,
                    "candidate_pool_size": args.candidate_pool_size, "candidate_sample_size": args.candidate_sample_size,
                    "sample_mode": args.sample_mode, "workers": args.workers}

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models=models,
        seed=seed,
        args=run_args,
    )

    usage_bucket = blank_usage()
    gen_usage_bucket = blank_usage()
    mc, reasons, issues, gen_sources = Counter(), Counter(), Counter(), Counter()
    backend_down = threading.Event()
    state = {"infra_backend": 0}

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"candidates": dict(mc), "contract_issues": dict(issues), "reasons": dict(reasons)}
        if args.generate:
            extra["generation"] = {"usage": usage_total(gen_usage_bucket), "sources": dict(gen_sources)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        name = res["name"]
        cid = f"{case_source}:{name}:v1"
        requirements = res["requirements"]
        if res.get("gen_usage"):                          # учёт генерации профиля (--generate)
            accumulate_usage(usage_bucket, res["gen_usage"])
            accumulate_usage(gen_usage_bucket, res["gen_usage"])
        if res.get("gen_source"):
            gen_sources[res["gen_source"]] += 1

        # инфра/данные на уровне всей вакансии -> errors, без кейса качества
        if res["case_infra"]:
            rb.add_error(cid, res["case_infra"])
            reasons[res["case_infra"].split(":")[0]] += 1
            if res["case_infra"].startswith("backend:"):
                state["infra_backend"] += 1
                if state["infra_backend"] >= args.backend_fail_fast:
                    backend_down.set()
            if not args.quiet:
                print(f"  [ERR ] {name}: {res['case_infra']}")
            return

        subjects: List[Dict[str, Any]] = []
        case_issue_codes: set = set()
        quality_count = 0
        for cand in res["candidates"]:
            accumulate_usage(usage_bucket, cand.get("usage"))
            cid_cand = f"{cid}:{cand['id']}"
            if cand["infra_error"]:                       # сетевой сбой промпта -> инфра, не subject
                rb.add_error(cid_cand, cand["infra_error"])
                mc["infra"] += 1
                continue
            quality_count += 1
            mc["evaluated"] += 1
            if cand["parse_error"]:                       # не JSON-массив -> контракт-фейл
                passed, codes = False, ["invalid_json_output"]
                req_results = None
            else:
                passed, case_issues, _details = check_contract(requirements, cand["predicted"])
                codes = list(case_issues)
                req_results = []
                for it in cand["predicted"]:
                    if isinstance(it, dict):
                        req_results.append({"requirement": str(it.get("requirement") or ""),
                                            "passed": bool(it.get("passed") in (1, True)),
                                            "comment": str(it.get("comment") or "")})
            if passed:
                mc["passed"] += 1
            else:
                mc["contract_fail"] += 1
                for c in codes:
                    issues[c] += 1
                    case_issue_codes.add(c)
            subj: Dict[str, Any] = {"id": cand["id"], "passed": bool(passed),
                                    "verdict": {"evaluator": "sourcing_contract", "passed": bool(passed), "reason_codes": codes}}
            if req_results is not None:
                subj["requirement_results"] = req_results
            subjects.append(subj)

        if quality_count == 0:                            # все кандидаты инфра -> не кейс качества
            rb.add_error(cid, "all_candidates_infra")
            reasons["all_candidates_infra"] += 1
            if not args.quiet:
                print(f"  [ERR ] {name}: all_candidates_infra")
            return

        mc["cases"] += 1
        case_passed = all(s["passed"] for s in subjects)
        rb.add_case(CaseRecord(
            case_id=cid, source=case_source, passed=case_passed,
            inputs={"criterion": "sourcing output contract: array 1:1 to requirements, item shape {requirement,comment,passed}",
                    "requirements": requirements, "vacancy_title": res["title"]},
            verdict={"evaluator": "sourcing_contract", "passed": case_passed, "reason_codes": sorted(case_issue_codes)},
            subjects=subjects,
        ))
        if not args.quiet:
            n_pass = sum(1 for s in subjects if s["passed"])
            print(f"  [{'ok ' if case_passed else 'MISS'}] {name} reqs={len(requirements)} "
                  f"candidates={len(subjects)} passed={n_pass} issues={sorted(case_issue_codes)}")

    total = len(items)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    if args.generate:
        def work(item):
            return _process_generate(item, **gen_setup)
    elif args.offline:
        work = _process_offline
    else:
        def work(item):
            return _process_online(item, sourcing_client, args.requirements_source, base_payload, sanitize_geo,
                                   backend, token, backend_down, args.candidate_sample_size, args.sample_mode, seed)

    outcome = run_cases(items, work=work, fold=_fold, max_workers=max(1, args.workers),
                        checkpoint_every=args.checkpoint_every, on_checkpoint=_flush, on_interrupt=_on_interrupt)

    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        s = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        mode = "offline" if args.offline else "online"
        print(f"[{tag}] {mode} cases(eval)={s['total']} passed={s['passed']} failed={s['failed']} "
              f"errors(infra)={s['errors']} candidates(eval/pass)={mc['evaluated']}/{mc['passed']} done={outcome.done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

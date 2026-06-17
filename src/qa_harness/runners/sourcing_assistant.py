"""Раннер sourcing_assistant: КОНСЕРВАТИВНАЯ 0/1-оценка резюме по требованиям.

Что тестируем — промпт `sourcing_assistant`: на вход `{requirements:[...предложения...], candidate_data:"..."}`
(резюме / профиль / анкета), на выход JSON-массив 1:1 к требованиям, объект `{requirement(echo), comment,
passed:0|1}` на каждое. Правило: 1 при ЯВНОМ ИЛИ близком по смыслу подтверждении в данных кандидата
(не только дословно); 0 — если подтверждения нет/косвенное (роль ≠ технология)/из текста вакансии.

Режимы (см. матрицу в docs/MIGRATION_STATUS):
- `--golden`  — реальный промпт на golden candidate_data-кейсах: contract + semantic(passed == expect_passed);
- `--offline` — replay offline_output из golden: contract + semantic (без сети);
- `--generate`— LLM-резюме с известными positive/negative навыками: contract + semantic; BACKEND НЕ нужен;
- (без флага) — backend: поиск ЖИВЫХ кандидатов по CDM → contract-ONLY (эталона passed нет; НЕ проверка
  качества scoring, а смоук формы). `--count-only` — только число кандидатов (диагностика);
- `--search`  — поиск ЖИВЫХ кандидатов по РЕАЛЬНЫМ вакансиям (vacancies.yaml): extractor(вакансия)→сущности
  →backend (limit=`--candidates`, дефолт 5)→профили→scoring каждого против requirements вакансии. На шаг
  дальше count-only (тянем профили, а не число); contract-only (люди живые, эталона passed нет).

contract: массив 1:1, точный echo, элемент {requirement,comment,passed}; passed — integer 0/1 (НЕ bool/строка);
comment без переносов, ≤2 предложений. semantic: passed совпадает с эталоном expect_passed (gate);
согласованность комментария с меткой — сигнал. quality ≠ infra: backend/сеть/генерация → errors.

  python -m qa_harness.runners.sourcing_assistant --offline                # replay golden, без сети
  python -m qa_harness.runners.sourcing_assistant --golden                 # online scoring на golden (OPENAI_API_KEY)
  python -m qa_harness.runners.sourcing_assistant --generate --variants 5  # вариативно: LLM-резюме, БЕЗ backend
  python -m qa_harness.runners.sourcing_assistant --cases-count 1 --candidate-sample-size 2 --token-in-body  # backend smoke (contract-only)
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
    GenerationPolicy,
    ResumeGenerator,
    ResumeSpec,
    generate_valid,
)
from qa_harness.domain.sourcing import (
    GoldenScoreCase,
    SearchVacancy,
    build_candidate_profile,
    check_contract,
    check_passed_labels,
    comment_inconsistencies,
    load_golden_score,
    load_search_vacancies,
    parse_sourcing_output,
    requirements_from_cdm,
)
from qa_harness.pipeline import (
    BackendCfg,
    build_step3_payload,
    call_backend_search_bool,
    make_base_payload,
    parse_extractor_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CDM_DIR = REPO_ROOT / "tests" / "fixtures" / "cdm" / "std"
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "sourcing_assistant" / "golden.yaml"
DEFAULT_VACANCIES = REPO_ROOT / "tests" / "fixtures" / "sourcing_assistant" / "vacancies.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "sourcing_assistant"
EXTRACTOR_COMPONENT = "extractor_agent"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"


def resume_text_from_profile(profile: Dict[str, Any]) -> str:
    """Сплющить backend-профиль {about, skills[], positions[]} в текст резюме (вход новой задачи)."""
    parts: List[str] = []
    about = str(profile.get("about") or "").strip()
    if about:
        parts.append(about)
    skills = [str(s.get("skill")).strip() for s in (profile.get("skills") or []) if isinstance(s, dict) and s.get("skill")]
    if skills:
        parts.append("Навыки: " + ", ".join(skills) + ".")
    for pos in (profile.get("positions") or [])[:4]:
        if isinstance(pos, dict):
            seg = " ".join(str(pos.get(k) or "").strip() for k in ("pos", "name", "description") if pos.get(k))
            if seg.strip():
                parts.append(seg.strip())
    return "\n".join(parts).strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sourcing_assistant QA runner (subjects[], contract, new architecture).")
    # --- scoring-режимы (contract + semantic по expect_passed); BACKEND НЕ нужен ---
    p.add_argument("--golden", action="store_true",
                   help="Scoring online: реальный промпт на golden resume_text-кейсах (contract + semantic).")
    p.add_argument("--offline", action="store_true", help="Replay offline_output из golden (без сети; contract + semantic).")
    p.add_argument("--generate", action="store_true",
                   help="Вариативный scoring: LLM-резюме с известными positive/negative навыками; без backend.")
    p.add_argument("--golden-file", type=Path, default=DEFAULT_GOLDEN, help="Golden-кейсы scoring (resume_text + expect_passed).")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help=f"Модель генератора резюме (по умолч. {DEFAULT_GEN_MODEL}).")
    p.add_argument("--gen-seed", type=int, default=None, help="Seed выборки домена/навыков.")
    p.add_argument("--variants", type=int, default=5, help="Сколько резюме сгенерить (--generate).")
    p.add_argument("--positive-skills", type=int, default=2, help="Сколько positive-навыков засевать (expect_passed=1).")
    p.add_argument("--negative-skills", type=int, default=1, help="Сколько negative-навыков засевать (expect_passed=0).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature генератора резюме (--generate).")
    p.add_argument("--gen-retries", type=int, default=2, help="Повторов генерации резюме при провале валидации.")
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
    # --- ЖИВОЙ поиск кандидатов по реальным вакансиям (extractor → backend → scoring) ---
    p.add_argument("--search", action="store_true",
                   help="Искать ЖИВЫХ кандидатов по реальным вакансиям (vacancies.yaml): extractor→backend→scoring.")
    p.add_argument("--vacancies-file", type=Path, default=DEFAULT_VACANCIES,
                   help="Файл вакансий для --search (текст вакансии + требования).")
    p.add_argument("--candidates", type=int, default=5, help="Сколько кандидатов искать и оценивать на вакансию (--search).")
    p.add_argument("--extractor-prompt-id", default=None, help="Override prompt_id extractor (--search).")
    p.add_argument("--extractor-prompt-version", default=None)
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
        candidate_data = resume_text_from_profile(profile)  # новый вход промпта — данные кандидата текстом
        try:
            raw, usage = sourcing_client.run(json.dumps({"requirements": requirements, "candidate_data": candidate_data}, ensure_ascii=False))
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


def _process_search(vac: SearchVacancy, extractor_client: Any, sourcing_client: Any,
                    base_payload: Dict[str, Any], sanitize_geo: bool, backend: BackendCfg, token: str,
                    backend_down: threading.Event, n_candidates: int) -> Dict[str, Any]:
    """Реальная вакансия → extractor (сущности) → backend-поиск ЖИВЫХ кандидатов → scoring каждого.

    На шаг дальше, чем --count-only: тянем профили (limit=N), а не только count, и каждого оцениваем
    промптом sourcing против requirements вакансии. Эталона passed нет (люди живые) → contract-качество.
    """
    res: Dict[str, Any] = {"name": vac.name, "title": vac.title, "requirements": list(vac.requirements),
                           "case_infra": None, "candidates": []}
    if not vac.requirements:
        res["case_infra"] = "no_requirements"
        return res
    try:                                                  # extractor: вакансия → сущности для поиска
        etext, _eu = extractor_client.run(vac.vacancy)
    except Exception as e:  # noqa: BLE001
        res["case_infra"] = f"extractor:{type(e).__name__}:{e}"
        return res
    entities, status = parse_extractor_json(etext)
    if entities is None:
        res["case_infra"] = f"extractor_invalid_json:{status}"
        return res
    payload = build_step3_payload(entities, vac.title or vac.vacancy[:80], base_payload, sanitize_office_geo=sanitize_geo)
    if not any(k in payload for k in ("positions", "skills", "keys")):
        res["case_infra"] = "no_search_entities"
        return res
    if backend_down.is_set():
        res["case_infra"] = "backend_down(skipped)"
        return res
    kind, status_code, _att, _count, berr, response = call_backend_search_bool(backend, token, payload)
    if kind not in ("success", "insufficient_search_terms"):
        res["case_infra"] = f"backend:{kind}:{berr or status_code}"
        return res
    profiles = [p for p in (response.get("profiles") or []) if isinstance(p, dict)] if isinstance(response, dict) else []
    if not profiles:
        res["case_infra"] = "no_candidates_found"
        return res
    for cand in profiles[:max(1, n_candidates)]:          # каждого найденного кандидата → scoring
        cid = str(cand.get("id") or cand.get("name") or "candidate")
        entry: Dict[str, Any] = {"id": cid, "predicted": None, "infra_error": None, "parse_error": None, "usage": None}
        candidate_data = resume_text_from_profile(build_candidate_profile(cand))
        try:
            raw, usage = sourcing_client.run(
                json.dumps({"requirements": vac.requirements, "candidate_data": candidate_data}, ensure_ascii=False))
            entry["usage"] = usage
        except Exception as e:  # noqa: BLE001
            entry["infra_error"] = f"sourcing:{type(e).__name__}:{e}"
            res["candidates"].append(entry)
            continue
        try:
            entry["predicted"] = parse_sourcing_output(raw)
        except ValueError as e:
            entry["parse_error"] = f"invalid_json_output:{e}"
        res["candidates"].append(entry)
    return res


# ---------------- scoring-режимы (golden / offline / generate): contract + semantic ----------------

# Требования-категории для generate: (требование, anchor-фраза для данных кандидата). Нейтральный стиль.
# Готовность/устройство — близкие формулировки (рассматриваю/возможно), не только дословное «готов».
_READINESS_POS = [
    ("Готовность к редким командировкам в Ереван.", "командировки в Ереван возможны"),
    ("Готовность работать в офисе в Москве.", "проживаю в Москве, офисный формат рассматриваю"),
    ("Готовность к релокации в Ереван.", "релокацию в Ереван рассматриваю"),
]
_DEVICE_POS = [
    ("Наличие стабильного интернет-соединения со скоростью не менее 20 Мбит/с.", "домашний интернет 100 Мбит/с"),
    ("Наличие смартфона или планшета с Android или iOS.", "есть смартфон на Android"),
]
_READINESS_NEG = [
    ("Готовность к релокации в Кипр.", "Кипр"),
    ("Готовность к частым командировкам в Сочи.", "Сочи"),
]


def _build_generate_case(variant: int, *, gen_client: Any, gen_policy: GenerationPolicy,
                         gen_seed: int, pos_n: int, neg_n: int):
    """Контролируемый кейс СМЕШАННЫХ типов: навыки + готовность/устройство (близкие формулировки).

    positive (expect=1): навык-в-данных + готовность/устройство (явно подтверждены в candidate_data);
    negative (expect=0): навык-отсутствует + готовность-отсутствует. LLM пишет candidate_data, упоминая
    positive-anchors и НЕ упоминая negative-anchors (валидация в ResumeSpec.parse).
    """
    rng = random.Random(f"{gen_seed}:{variant}")
    domain = rng.choice(list(TECH_VOCAB))
    techs = rng.sample(TECH_VOCAB[domain], min(3, len(TECH_VOCAB[domain])))
    pos_skill, neg_skill, decoy = techs[0], techs[1], techs[2]
    avail = rng.choice(_READINESS_POS + _DEVICE_POS)       # одно подтверждаемое условие
    neg_avail = rng.choice(_READINESS_NEG)                  # одно отсутствующее условие

    requirements = [f"Опыт работы с {pos_skill}.", avail[0], f"Опыт работы с {neg_skill}.", neg_avail[0]]
    expect = [1, 1, 0, 0]
    must_mention = [pos_skill, avail[1], decoy]             # навык + anchor условия + decoy (смежное, не требование)
    must_not_mention = [neg_skill, neg_avail[1]]

    gen = ResumeGenerator(gen_client)
    gr = generate_valid(
        lambda _a: (gen.generate(ResumeSpec(domain, must_mention, must_not_mention, noise_level=variant % 3)), None),
        policy=gen_policy)
    candidate_data = str(gr.item) if gr.ok else ""
    case = GoldenScoreCase(name=f"v{variant}_{domain}", requirements=requirements,
                           candidate_data=candidate_data, expect_passed=expect)
    return case, gr, dict(gen.usage)


def _process_score(case: GoldenScoreCase, sourcing_client: Any, replay: bool) -> Dict[str, Any]:
    """Один scoring-кейс: реплей offline_output ИЛИ вызов промпта на {requirements, candidate_data}."""
    res: Dict[str, Any] = {"case": case, "predicted": None, "parse_error": None,
                           "infra_error": None, "usage": None}
    if replay:
        res["predicted"] = list(case.offline_output)
        return res
    try:
        raw, usage = sourcing_client.run(
            json.dumps({"requirements": case.requirements, "candidate_data": case.candidate_data}, ensure_ascii=False))
        res["usage"] = usage
    except Exception as e:  # noqa: BLE001 — сетевой сбой промпта = инфра
        res["infra_error"] = f"sourcing:{type(e).__name__}:{e}"
        return res
    try:
        res["predicted"] = parse_sourcing_output(raw)
    except ValueError as e:
        res["parse_error"] = f"invalid_json_output:{e}"
    return res


def _run_scoring(args, prompt, run_id, started) -> Dict[str, Path]:
    """Scoring-режимы (golden / offline / generate): contract + semantic(expect_passed). БЕЗ backend."""
    mode = "offline" if args.offline else ("generate" if args.generate else "golden")
    online = mode != "offline"
    sourcing_client = None
    gen_setup: Dict[str, Any] = {}
    if online:
        from qa_harness.core.llm_client import ModelClient, StoredPromptClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (sourcing prompt requires it)")
        sourcing_client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=get_client(timeout=args.step1_timeout))

    if mode == "generate":
        gen_seed = args.gen_seed if args.gen_seed is not None else (prompt.seed if prompt.seed is not None else 0)
        gen_setup = dict(
            gen_client=ModelClient(args.gen_model, timeout=args.step1_timeout, temperature=args.temperature),
            gen_policy=GenerationPolicy(max_retries=args.gen_retries, temperature=args.temperature, seed=gen_seed),
            gen_seed=gen_seed, pos_n=args.positive_skills, neg_n=args.negative_skills)
        items: List[Any] = list(range(max(1, args.variants)))
        models = {"resume_generator": args.gen_model, "evaluator": None}
        run_args = {"mode": "generate", "variants": args.variants, "positive": args.positive_skills,
                    "negative": args.negative_skills, "gen_model": args.gen_model, "gen_seed": gen_seed,
                    "temperature": args.temperature, "workers": args.workers}
        case_source = "synthetic"
    else:
        items = load_golden_score(args.golden_file)
        models = {"generator": None, "evaluator": None}
        run_args = {"mode": mode, "golden": len(items), "workers": args.workers}
        case_source = "suite" if mode == "offline" else "golden"

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id, started_at=started.isoformat(timespec="seconds"),
        models=models, seed=prompt.seed, args=run_args,
    )
    usage_bucket = blank_usage()
    gen_usage_bucket = blank_usage()
    mc, reasons, gen_sources = Counter(), Counter(), Counter()
    signals: Counter = Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"scoring": dict(mc), "reasons": dict(reasons), "comment_signals": dict(signals)}
        if mode == "generate":
            extra["generation"] = {"usage": usage_total(gen_usage_bucket), "sources": dict(gen_sources)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _work(item):
        if mode == "generate":
            case, gr, gen_usage = _build_generate_case(item, **gen_setup)
            if not gr.ok:
                return {"case": case, "predicted": None, "parse_error": None,
                        "infra_error": f"resume_gen_failed:{'; '.join(gr.errors[-2:]) or 'unknown'}",
                        "usage": None, "gen_source": gr.source, "gen_usage": gen_usage}
            res = _process_score(case, sourcing_client, replay=False)
            res["gen_source"] = gr.source
            res["gen_usage"] = gen_usage
            return res
        return _process_score(item, sourcing_client, replay=(mode == "offline"))

    def _fold(res: Dict[str, Any]) -> None:
        case: GoldenScoreCase = res["case"]
        requirements, expect = case.requirements, case.expect_passed
        accumulate_usage(usage_bucket, res.get("usage"))
        if res.get("gen_usage"):
            accumulate_usage(usage_bucket, res["gen_usage"])
            accumulate_usage(gen_usage_bucket, res["gen_usage"])
        if res.get("gen_source"):
            gen_sources[res["gen_source"]] += 1
        cid = f"{case_source}:{case.name}:v1"

        if res.get("infra_error"):                        # сбой генерации/сети → errors (не качество)
            rb.add_error(cid, res["infra_error"])
            reasons[res["infra_error"].split(":")[0]] += 1
            mc["errors"] += 1
            if not args.quiet:
                print(f"  [ERR ] {case.name}: {res['infra_error']}")
            return

        mc["total"] += 1
        rc: List[str] = []
        if res["parse_error"]:
            predicted = []
            contract_ok, c_issues = False, ["invalid_json_output"]
            semantic_ok, sem_diffs, comment_sig = False, ["no_output"], []
            reasons["invalid_json_output"] += 1
        else:
            predicted = res["predicted"]
            contract_ok, c_issues, _det = check_contract(requirements, predicted)
            semantic_ok, sem_diffs = check_passed_labels(predicted, expect)
            comment_sig = comment_inconsistencies(predicted)
        if contract_ok:
            mc["contract_pass"] += 1
        else:
            reasons["contract_fail"] += 1
            rc += [f"contract:{i}" for i in c_issues]
        if semantic_ok:
            mc["semantic_pass"] += 1
        else:
            reasons["semantic_fail"] += 1
            rc += [f"semantic:{d}" for d in sem_diffs[:8]]
        for s in comment_sig:                              # СИГНАЛ (не gate)
            signals[s.split(":")[-1]] += 1

        passed = bool(contract_ok) and bool(semantic_ok)
        if passed:
            mc["passed"] += 1
        subjects: List[Dict[str, Any]] = []
        for i, req in enumerate(requirements):
            item = predicted[i] if i < len(predicted) and isinstance(predicted[i], dict) else {}
            subjects.append({"requirement": req, "expected_passed": expect[i] if i < len(expect) else None,
                             "actual_passed": item.get("passed"), "comment": item.get("comment")})
        rb.add_case(CaseRecord(
            case_id=cid, source=case_source, passed=passed,
            inputs={"criterion": "contract(1:1 echo) + semantic(passed == expect_passed)",
                    "requirements": requirements, "candidate_data": case.candidate_data, "expect_passed": expect},
            output={"raw": None, "parsed": predicted},
            verdict={"evaluator": "sourcing_scoring", "passed": passed, "reason_codes": rc},
            subjects=subjects,
            checks=[{"rule": "contract", "passed": bool(contract_ok), "detail": ",".join(c_issues)},
                    {"rule": "semantic_passed", "passed": bool(semantic_ok), "detail": ",".join(sem_diffs[:8])},
                    {"rule": "comment_consistency(info)", "passed": not comment_sig, "detail": ",".join(comment_sig[:5])}],
        ))
        if not args.quiet:
            print(f"  [{'ok ' if passed else 'MISS'}] {case.name} reqs={len(requirements)} "
                  f"contract={int(bool(contract_ok))} semantic={semantic_ok} signals={len(comment_sig)}")

    total = len(items)
    outcome = run_cases(items, work=_work, fold=_fold, max_workers=max(1, args.workers),
                        checkpoint_every=args.checkpoint_every, on_checkpoint=_flush,
                        on_interrupt=(lambda: None) if args.quiet else (lambda: print("\n[interrupted]")))
    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        s = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        print(f"[{tag}] {mode} cases={s['total']} passed={s['passed']} failed={s['failed']} "
              f"errors(infra)={s['errors']} done={outcome.done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


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

    # scoring-режимы (golden / offline / generate): contract + semantic(expect_passed), БЕЗ backend
    if args.offline or args.generate or args.golden:
        return _run_scoring(args, prompt, run_id, started)

    # дефолт: backend (поиск ЖИВЫХ кандидатов) — contract-only (нет эталона passed для живых профилей)
    from qa_harness.core.llm_client import StoredPromptClient, get_client

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set (sourcing prompt requires it)")
    sourcing_client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=get_client(timeout=args.step1_timeout))

    base_url = args.base_url or os.environ.get("AI_SEARCH_BASE_URL")
    token = args.token or os.environ.get("AI_SEARCH_AUTH_TOKEN") or ""
    if not base_url or not token:
        raise EnvironmentError("backend-режим требует AI_SEARCH_BASE_URL/AI_SEARCH_AUTH_TOKEN (или --base-url/--token)")
    backend: Optional[BackendCfg] = BackendCfg(
        base_url=base_url, step3_path=args.step3_path, token_in_body=bool(args.token_in_body),
        timeout_s=args.step3_timeout, retries=args.step3_retries)

    extractor_client = None
    if args.search:
        if args.count_only:
            raise ValueError("--search несовместим с --count-only")
        extractor = resolve_prompt(cfg, EXTRACTOR_COMPONENT, cli_id=args.extractor_prompt_id,
                                   cli_version=args.extractor_prompt_version)
        extractor_client = StoredPromptClient(extractor.prompt_id, extractor.prompt_version,
                                              client=get_client(timeout=args.step1_timeout))
        items = list(load_search_vacancies(args.vacancies_file))
        case_source = "vacancy"
        base_payload = make_base_payload(only_with_contacts=True, current_position_title=True,
                                         limit=max(1, args.candidates), offset=0)
    else:
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

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None},
        seed=seed,
        args=({"mode": "search", "vacancies": len(items), "candidates": args.candidates,
               "extractor": {"prompt_id": extractor.prompt_id, "prompt_version": extractor.prompt_version},
               "workers": args.workers, "scoring": "contract_only(live candidates, no expect_passed)"}
              if args.search else
              {"mode": "backend", "cases": len(items), "requirements_source": args.requirements_source,
               "candidate_pool_size": args.candidate_pool_size, "candidate_sample_size": args.candidate_sample_size,
               "sample_mode": args.sample_mode, "workers": args.workers, "scoring": "contract_only(no expect_passed)"}),
    )

    usage_bucket = blank_usage()
    mc, reasons, issues = Counter(), Counter(), Counter()
    backend_down = threading.Event()
    state = {"infra_backend": 0}

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"candidates": dict(mc), "contract_issues": dict(issues), "reasons": dict(reasons)}
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

    if args.search:
        def work(item):
            return _process_search(item, extractor_client, sourcing_client, base_payload, sanitize_geo,
                                   backend, token, backend_down, args.candidates)
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

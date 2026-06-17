"""Раннер first_touch: генерация первого касания → LLM-судья фактов + эвристики.

Что тестируем — промпт `first_touch`: из payload (кандидат/вакансия/причина/стек/зарплата) генерит
сообщение первого касания. Без бэкенда. Оценка кейса:
- LLM-судья (отдельная модель): facts_present (каждый ожидаемый факт упомянут?), hallucinated_facts
  (выдуманные факты про вакансию/условия), question_present (есть CTA-вопрос);
- эвристики: extra_numbers (числа ≥5 цифр не из фактов), company_hidden (нет утечки названия).

Итог (passed) = все обязательные факты есть & нет лишних чисел & есть вопрос (если нужен) & при
company_hidden нет названия компании. Выдуманные факты (hallucinated) — СИГНАЛ, не gate (LLM-судья шумит
на общих фразах). quality ≠ infra: сбой генерации/судьи → errors.
`--offline` реплеит offline_message и использует эвристику вместо LLM-судьи (галлюцинации офлайн не ловятся).

Режимы: golden (дефолт, payload из golden.yaml) · `--generate` (LLM генерит входную вакансию, expected_facts
выводятся из неё; домен/сениорити/контекст варьируются по seed; `--variants` шт.) · `--offline` (replay).

  python -m qa_harness.runners.first_touch --offline                 # replay + эвристика, без сети
  python -m qa_harness.runners.first_touch                           # онлайн golden: генерация + LLM-судья
  python -m qa_harness.runners.first_touch --generate --variants 4 --temperature 0.8
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import random

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, run_cases, usage_total
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.generators import (
    DOMAINS,
    SENIORITIES,
    GenerationPolicy,
    VacancyGenerator,
    VacancySpec,
    generate_valid,
)
from qa_harness.domain.first_touch import (
    FactJudge,
    GoldenCase,
    company_name_leaked,
    extra_numbers,
    facts_present_heuristic,
    forbidden_phrases,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "first_touch" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "first_touch"
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"
_GEN_NAMES = ("Иван", "Мария", "Дмитрий", "Ольга", "Сергей", "Анна", "Павел", "Екатерина")
_GEN_SOURCES = ("LinkedIn", "HeadHunter", "рекомендация коллеги", "GitHub", "профильное сообщество")
_GEN_REASONS = ("заинтересовал ваш опыт и стек", "профиль хорошо подходит под вакансию",
                "рекомендация по вашим проектам")
PAYLOAD_KEYS = (
    "candidate_name", "recruiter_name", "candidate_source", "reason_of_communication",
    "hiring_company_name", "vacancy_name", "vacancy_responsibilities", "message_formality",
    "company_description", "vacancy_stack", "salary_range",
)


def build_parser(default_component: str = "first_touch", default_golden: Path = DEFAULT_GOLDEN) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="first_touch QA runner (generate + LLM fact-judge + heuristics).")
    p.add_argument("--component", default=default_component, help="prompt-компонент из model.yaml: first_touch | first_touch_hh.")
    p.add_argument("--golden", type=Path, default=default_golden, help="Курируемые golden-кейсы (input + expected_facts).")
    p.add_argument("--offline", action="store_true", help="Replay offline_message + эвристика вместо LLM-судьи (без сети).")
    # --- режим вариативной генерации (LLM генерит входную вакансию) ---
    p.add_argument("--generate", action="store_true",
                   help="Вариативный режим: входную вакансию генерит LLM (а не из golden); expected_facts из неё.")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help=f"Модель генератора вакансии (по умолч. {DEFAULT_GEN_MODEL}).")
    p.add_argument("--gen-seed", type=int, default=None, help="Seed выборки домена/сениорити/контекста.")
    p.add_argument("--variants", type=int, default=4, help="Сколько вакансий сгенерить (--generate).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature генератора вакансии (--generate).")
    p.add_argument("--gen-retries", type=int, default=2, help="Повторов генерации вакансии при провале валидации.")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель LLM-судьи (по умолчанию {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--workers", type=int, default=4, help="Параллельных воркеров (2 LLM-вызова на кейс).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызовов LLM (генерация/судья), сек.")
    p.add_argument("--checkpoint-every", type=int, default=20, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _ending_check(text: str, phrase: str = "С уважением") -> str:
    """Срезать прощальную подпись (как в легаси FirstTouchGenerator)."""
    lines = (text or "").split("\n")
    if len(lines) >= 2 and any(phrase.lower() in ln.lower() for ln in lines[-2:]):
        del lines[-2:]
    return "\n".join(lines).strip()


def _process(case: GoldenCase, gen_client: Any, judge: Any, offline: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"case": case, "message": None, "usage": None, "judge_usage": None,
                           "facts_present": {}, "hallucinated": [], "question_present": False,
                           "call_error": None, "judge_error": None}
    if offline:
        if case.offline_message is None:
            res["call_error"] = "no_offline_message"
            return res
        msg = case.offline_message.strip()
        res["message"] = msg
        res["facts_present"] = facts_present_heuristic(case.expected_facts, msg)
        res["question_present"] = "?" in msg
        return res

    payload = dict(case.input)
    if case.company_hidden:
        payload["hiring_company_name"] = ""  # прод прячет название; промпт опирается на company_description
    try:
        text, usage = gen_client.run(json.dumps(payload, ensure_ascii=False))
        res["message"], res["usage"] = _ending_check(text), usage
    except Exception as e:  # noqa: BLE001 — сбой генерации = инфра
        res["call_error"] = f"generate:{type(e).__name__}:{e}"
        return res
    if not res["message"]:
        res["call_error"] = "generate:empty_output"
        return res
    # legit-контекст из input (его дали промпту) — упоминание этих фактов НЕ галлюцинация
    allowed = dict(case.allowed_context_facts)
    allowed_keys = ["company_description", "vacancy_responsibilities", "vacancy_text",
                    "reason_of_communication", "vacancy_stack", "salary_range"]
    if not case.company_hidden:  # при hidden имя компании скрыто (его утечку ловит отдельный чек)
        allowed_keys.append("hiring_company_name")
    for k in allowed_keys:
        v = str(case.input.get(k) or "").strip()
        if v:
            allowed[k] = v
    try:
        verdict, jusage = judge.evaluate(case.expected_facts, allowed, res["message"])
        res["judge_usage"] = jusage
        res["facts_present"] = verdict.facts_present
        res["hallucinated"] = verdict.hallucinated_facts
        res["question_present"] = verdict.question_present
    except Exception as e:  # noqa: BLE001 — сбой судьи = инфра
        res["judge_error"] = f"judge:{type(e).__name__}:{e}"
    return res


def _process_generate(variant: int, *, gen_client: Any, judge: Any, vacancy_client: Any,
                      gen_policy: GenerationPolicy, gen_seed: int, component: str) -> Dict[str, Any]:
    """Один вариант: LLM генерит вакансию → собираем payload + expected_facts → существующий _process."""
    rng = random.Random(f"{gen_seed}:{variant}")
    domain = rng.choice(DOMAINS)
    seniority = rng.choice(SENIORITIES)
    candidate_name = rng.choice(_GEN_NAMES)
    source = rng.choice(_GEN_SOURCES)
    reason = rng.choice(_GEN_REASONS)
    formality = rng.choice(["formal", "informal"])
    hidden = rng.random() < 0.4

    vgen = VacancyGenerator(vacancy_client)
    gr = generate_valid(lambda _a: (vgen.generate(VacancySpec(domain, seniority, noise_level=variant % 3)), None),
                        policy=gen_policy)
    base = {"mode": "generate", "variant": variant, "gen_source": gr.source, "gen_usage": dict(vgen.usage)}
    if not gr.ok:
        return {**base, "case": None, "message": None, "usage": None, "judge_usage": None,
                "facts_present": {}, "hallucinated": [], "question_present": False,
                "call_error": f"vacancy_gen_failed:{'; '.join(gr.errors[-2:]) or 'unknown'}", "judge_error": None}
    vac = gr.item
    payload = {
        "candidate_name": candidate_name, "recruiter_name": "Анна", "candidate_source": source,
        "reason_of_communication": reason, "hiring_company_name": vac["hiring_company_name"],
        "vacancy_name": vac["vacancy_name"], "vacancy_responsibilities": vac["vacancy_responsibilities"],
        "message_formality": formality, "company_description": vac["company_description"],
        "vacancy_stack": vac["vacancy_stack"], "salary_range": vac["salary_range"],
    }
    expected = {"vacancy_name": vac["vacancy_name"], "candidate_name": candidate_name}
    optional: List[str] = []
    if vac["salary_range"]:
        expected["salary"] = vac["salary_range"]
        optional.append("salary")
    # hh-промпт не должен упоминать источник кандидата → ловим утечку источника
    forbid = [source] if "hh" in component else []
    case = GoldenCase(name=f"v{variant}_{domain.split()[0]}_{seniority}", input=payload,
                      expected_facts=expected, optional_facts=optional, forbid_in_message=forbid,
                      company_hidden=hidden, require_question=True)
    res = _process(case, gen_client, judge, offline=False)
    res.update(base)
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")
    component = args.component

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, component, cli_id=args.prompt_id, cli_version=args.prompt_version)

    if args.generate and args.offline:
        raise ValueError("--generate несовместим с --offline (генерация требует сети).")

    gen_client = judge = None
    eval_model = None
    gen_setup: Dict[str, Any] = {}
    if not args.offline:
        from qa_harness.core.llm_client import ModelClient, StoredPromptClient, get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")
        llm = get_client(timeout=args.step1_timeout)
        gen_client = StoredPromptClient(prompt.prompt_id, prompt.prompt_version, client=llm,
                                        text_format={"format": {"type": "text"}})
        eval_model = args.eval_model
        judge = FactJudge(ModelClient(eval_model, timeout=args.step1_timeout))

    if args.generate:
        from qa_harness.core.llm_client import ModelClient

        gen_seed = args.gen_seed if args.gen_seed is not None else (prompt.seed if prompt.seed is not None else 0)
        gen_setup = dict(
            vacancy_client=ModelClient(args.gen_model, timeout=args.step1_timeout, temperature=args.temperature),
            gen_policy=GenerationPolicy(max_retries=args.gen_retries, temperature=args.temperature, seed=gen_seed),
            gen_seed=gen_seed,
            component=component,
        )
        work_items: List[Any] = list(range(max(1, args.variants)))
        models = {"vacancy_generator": args.gen_model, "assistant": prompt.prompt_id, "evaluator": eval_model}
        run_args = {"mode": "generate", "component": component, "variants": args.variants,
                    "gen_model": args.gen_model, "gen_seed": gen_seed, "temperature": args.temperature,
                    "workers": args.workers, "eval_model": eval_model}
    else:
        work_items = list(load_golden(args.golden))
        models = {"generator": prompt.prompt_id, "evaluator": eval_model}
        run_args = {"mode": "golden", "offline": bool(args.offline), "golden": len(work_items),
                    "workers": args.workers, "eval_model": eval_model}

    rb = ReportBuilder(
        runner=component,
        prompt_under_test={"component": component, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models=models,
        seed=prompt.seed,
        args=run_args,
    )

    usage_bucket = blank_usage()
    gen_usage_bucket = blank_usage()
    m, reasons, gen_sources = Counter(), Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"first_touch": dict(m), "reasons": dict(reasons)}
        if args.generate:
            extra["generation"] = {"usage": usage_total(gen_usage_bucket), "sources": dict(gen_sources)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, component, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        is_gen = res.get("mode") == "generate"
        accumulate_usage(usage_bucket, res["usage"])
        accumulate_usage(usage_bucket, res["judge_usage"])
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

        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {label}: {res['call_error']}")
            return
        if res["judge_error"]:
            rb.add_error(cid, res["judge_error"])
            reasons["judge_error"] += 1
            m["judge_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {label}: {res['judge_error']}")
            return
        case: GoldenCase = res["case"]

        m["total"] += 1
        facts_present = res["facts_present"]
        required = [k for k in case.expected_facts if k not in case.optional_facts]
        missing_required = [k for k in required if not facts_present.get(k)]
        extra_nums = extra_numbers(list(case.expected_facts.values()), res["message"])
        question_ok = res["question_present"] if case.require_question else True
        company_leak = case.company_hidden and company_name_leaked(str(case.input.get("hiring_company_name") or ""), res["message"])
        forbidden_in_msg = forbidden_phrases(res["message"], case.forbid_in_message)
        hallucinated = res["hallucinated"] or []

        # hallucinated — СИГНАЛ, не gate: LLM-судья шумит на общих фразах (в легаси это было strict-only)
        quality_passed = (not missing_required) and (not extra_nums) and question_ok and (not company_leak) and (not forbidden_in_msg)
        rc: List[str] = []
        if missing_required:
            reasons["missing_facts"] += 1
            rc += [f"missing_fact:{k}" for k in missing_required]
        else:
            m["facts_pass"] += 1
        if extra_nums:
            reasons["extra_numbers"] += 1
            rc += [f"extra_number:{n}" for n in extra_nums]
        else:
            m["no_extra_numbers"] += 1
        if question_ok:
            m["question_pass"] += 1
        else:
            reasons["no_question"] += 1
            rc.append("no_question")
        if company_leak:
            reasons["company_leak"] += 1
            rc.append("company_name_leaked")
        else:
            m["company_hidden_ok"] += 1
        if forbidden_in_msg:
            reasons["forbidden_in_message"] += 1
            rc += [f"forbidden_in_message:{p}" for p in forbidden_in_msg]
        else:
            m["no_forbidden_phrases"] += 1
        if hallucinated:
            reasons["hallucination"] += 1
            m["hallucination_total"] += len(hallucinated)
        else:
            m["no_hallucination"] += 1

        checks = [
            {"rule": "facts_required", "passed": not missing_required, "detail": ",".join(missing_required)},
            {"rule": "no_extra_numbers", "passed": not extra_nums, "detail": ",".join(extra_nums)},
            {"rule": "question", "passed": bool(question_ok), "detail": ""},
            {"rule": "company_hidden", "passed": not company_leak,
             "detail": str(case.input.get("hiring_company_name") or "") if company_leak else ""},
            {"rule": "no_forbidden_phrases", "passed": not forbidden_in_msg, "detail": ",".join(forbidden_in_msg)},
            {"rule": "no_hallucination(info)", "passed": not hallucinated, "detail": "; ".join(hallucinated[:5])},
        ]
        inputs: Dict[str, Any] = {"criterion": "facts present & no hallucination & no extra numbers & question & company_hidden",
                                  "input": case.input, "expected_facts": case.expected_facts,
                                  "company_hidden": case.company_hidden, "require_question": case.require_question}
        if is_gen:
            inputs["variant"] = res["variant"]
            inputs["gen_source"] = res.get("gen_source")
        rb.add_case(CaseRecord(
            case_id=cid, source="synthetic" if is_gen else "golden", passed=quality_passed,
            inputs=inputs,
            output={"raw": res["message"]},
            verdict={"evaluator": "first_touch_llm_judge" if eval_model else "first_touch_heuristic",
                     "model": eval_model, "passed": quality_passed, "reason_codes": rc},
            checks=checks,
        ))
        if not args.quiet:
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {label} "
                  f"facts_missing={len(missing_required)} halluc={len(hallucinated)} "
                  f"extra_nums={len(extra_nums)} q={question_ok}")

    total = len(work_items)

    if args.generate:
        def _work(it):
            return _process_generate(it, gen_client=gen_client, judge=judge, **gen_setup)
    else:
        def _work(c):
            return _process(c, gen_client, judge, args.offline)

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

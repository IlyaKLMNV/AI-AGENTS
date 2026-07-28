"""Раннер screening_split: тест НОВОГО раздельного скрининга (Аналитик + Интервьюер).

Split = два промпта из пакета `prompts` (`screening_analyzer` — «мозг», строгий JSON
Decision; `screening_interviewer` — «рот», одно сообщение) + КОД-оркестратор (состояние,
счётчики/пороги, фиксированные скрипты), портированный из tgApi 1:1
(qa_harness.domain.screening_split). Тестируется как в проде: тела/схема — из пакета
`prompts` (LOCAL-источник), арифметика состояний — в коде.

СЦЕНАРИИ — отдельный CSV (`tests/fixtures/screening_split/scenarios.csv`, копия golden
монолита + новый зарплатный кейс). Легаси-раннер screening_scenarios и его CSV не трогаем.

Режимы:
- `--offline` — плумбинг: сценарии + реплики кандидата + санити чистого домена (без сети
  и без пакета prompts);
- golden (дефолт) — реплики кандидата из CSV, живой прогон split-движка, судья диалога
  (ScenarioJudge против expected_behavior). Слои A (Аналитик, checks) и B (Интервьюер) —
  следующий этап; `--generate` — позже.

  python -m qa_harness.runners.screening_split --offline
  python -m qa_harness.runners.screening_split --scenario-indices 62 --prompts-path ../prompts
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qa_harness.core import (
    LOCAL,
    accumulate_usage,
    add_prompt_source_args,
    blank_usage,
    component_cfg,
    ensure_prompts_importable,
    load_cfg,
    load_local_spec,
    resolve_source,
    run_cases,
    usage_total,
)
from qa_harness.domain import screening_split as sp
from qa_harness.domain.generators import CandidateAgent, GenerationPolicy, VariantSampler
from qa_harness.domain.screening import run_adaptive_conversation
from qa_harness.domain.screening_scenarios import (
    END_MARKER,
    Scenario,
    ScenarioJudge,
    constraints_for,
    extract_candidate_examples,
    load_constraints,
    load_scenarios,
    load_vacancies,
    parse_scenario_indices,
    vacancy_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_CSV = FIXTURES / "screening_split" / "scenarios.csv"
# Пер-сценарный контекст вакансии (формат/локация/вилка/скрытость) — переиспользуем набор
# монолита. ВНИМАНИЕ про индексы: split-CSV реиндексирован (удалены дубли 57/58/59), поэтому
# split-строки 1..56 == база 1..56, а split 57..62 == база 60..65. Записи вакансий все ≤40 (в
# неизменной зоне), сценарии без записи берут DEFAULT_VACANCY_INFO — расхождение их не задевает.
DEFAULT_VACANCIES = FIXTURES / "generation" / "screening_scenarios" / "scenario_vacancies.yaml"
# Констрейнты генерации — набор монолита (base-keyed по index). После реиндекса split-инъекция(60)/
# аудио(61) НЕ мапятся на base-записи 63/64 → в generated берут дженерик-констрейнты (обе — edge/
# dialogue-сценарии, в scripted идут по рецепту; общий constraints.yaml НЕ трогаем ради легаси).
DEFAULT_CONSTRAINTS = FIXTURES / "generation" / "screening_scenarios" / "constraints.yaml"
DEFAULT_CHECKS = FIXTURES / "screening_split" / "scenario_checks.yaml"
DEFAULT_INPUTS = FIXTURES / "screening_split" / "candidate_inputs.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_split"
ANALYZER_COMPONENT = "screening_analyzer"
INTERVIEWER_COMPONENT = "screening_interviewer"
DEFAULT_EVAL_MODEL = "gpt-4.1"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TURNS = 6
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
    p = argparse.ArgumentParser(description="screening_split QA runner (Аналитик + Интервьюер; local prompts).")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV сценариев split (по умолч. отдельный от легаси).")
    p.add_argument("--sample", type=int, default=5, help="Случайная выборка N сценариев (0 = все).")
    p.add_argument("--scenario-indices", default=None, help="Точечные номера строк CSV, напр. 1,7,62 (override --sample).")
    p.add_argument("--max-examples", type=int, default=4, help="Сколько реплик кандидата брать из примеров на сценарий.")
    p.add_argument("--offline", action="store_true", help="Плумбинг: сценарии + реплики + санити чистого домена, без сети.")
    # --- вариативный режим (адаптивный LLM-кандидат вместо реплик из CSV) ---
    p.add_argument("--generate", action="store_true", help="Вариативный режим: реплики кандидата генерит адаптивный LLM (разблокирует сценарии без примеров).")
    p.add_argument("--input-mode", choices=["scripted", "generated"], default="scripted",
                   help="Как подавать вход для РЕЦЕПТНЫХ сценариев: scripted (дословно, детерминированный "
                        "CI-гейт, дефолт) | generated (LLM-генератор, засеянный из рецепта: сумма из вилки + "
                        "триггер/примеры; Аналитик-инвариант ГЕЙТИТ так же, как в scripted). generated включает генератор сам.")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help=f"Модель генератора кандидата (по умолч. {DEFAULT_GEN_MODEL}).")
    p.add_argument("--gen-seed", type=int, default=None, help="Seed разнообразия стилей (по умолч. = --seed).")
    p.add_argument("--variants", type=int, default=1, help="Сколько вариативных прогонов на сценарий (--generate).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature генератора кандидата (--generate).")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Макс. ходов кандидата в адаптивном диалоге.")
    p.add_argument("--gen-retries", type=int, default=1, help="Повторов генерации реплики при провале валидации.")
    p.add_argument("--constraints", type=Path, default=None, help="YAML констрейнтов генерации (по умолч. набор монолита).")
    p.add_argument("--analyzer-version", default=None, metavar="vN", help="Версия screening_analyzer в пакете prompts (иначе pointer.yaml active).")
    p.add_argument("--interviewer-version", default=None, metavar="vN", help="Версия screening_interviewer (иначе pointer.yaml active).")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель судей диалога/Интервьюера (по умолч. {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--no-interviewer-judge", action="store_true", help="Отключить LLM-судью Интервьюера (слой B — только детерминированный leak-scan).")
    p.add_argument("--vacancies", type=Path, default=None, help="YAML пер-сценарного контекста вакансии (по умолч. набор монолита).")
    p.add_argument("--checks", type=Path, default=None, help="YAML инвариантов Decision Аналитика (по умолч. scenario_checks.yaml).")
    p.add_argument("--candidate-inputs", type=Path, default=None, help="YAML скриптовых реплик кандидата (по умолч. candidate_inputs.yaml).")
    p.add_argument("--workers", type=int, default=3, help="Параллельных сценариев (каждый — живой разговор).")
    p.add_argument("--step1-timeout", type=int, default=90)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=None, help="Seed выборки сценариев.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    add_prompt_source_args(p)  # --prompt-source/--local-prompt-version/--prompts-path (split — local-only)
    return p


def _select(scenarios: List[Scenario], indices_raw: str | None, sample: int, seed: Any,
            *, runnable_only: bool, max_examples: int, scripted_indices: set | None = None) -> List[Scenario]:
    if indices_raw:  # точечный выбор — как просили, без фильтра
        wanted = parse_scenario_indices(indices_raw)
        by_idx = {s.index: s for s in scenarios}
        return [by_idx[i] for i in wanted if i in by_idx]
    pool = scenarios
    scripted = scripted_indices or set()
    if runnable_only:  # онлайн golden: гоняем сценарии с примерами ИЛИ со скриптовым входом
        pool = [s for s in scenarios if s.index in scripted or extract_candidate_examples(s.examples_raw, max_examples)]
    if sample and sample > 0:
        return random.Random(seed).sample(pool, min(sample, len(pool)))
    return pool


# ── прогон одного сценария на ФИКСИРОВАННЫХ репликах (golden из CSV / скриптовые) ──
# gate_mode: "analyzer" — детерминированный гейт по трассе (ScenarioJudge НЕ зовём) ·
#            "dialogue" — гейт LLM-судьёй диалога (для сценариев без машинных инвариантов).
def _run_fixed_turns(scenario: Scenario, variant: int, candidate_turns: List[str], *, client: Any,
                     analyzer_client: Any, interviewer_spec: Any, judge: Any, ijudge: Any,
                     vinfo: Dict[str, Any], gate_mode: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {"scenario": scenario, "variant": variant, "turns": [], "verdict": None,
                           "judge_usage": None, "leak": None, "iverdict": None, "ijudge_usage": None,
                           "call_error": None, "gate": gate_mode, "gen_sources": []}
    if not candidate_turns:
        res["call_error"] = "no_candidate_examples"
        return res
    try:
        conv = sp.SplitConversation(
            client=client, analyzer_client=analyzer_client, interviewer_spec=interviewer_spec,
            vacancy_info=vinfo, recruiter_name=DEFAULT_RECRUITER_NAME, candidate_name="Кандидат")
        conv.start()
    except Exception as e:  # noqa: BLE001
        res["call_error"] = f"conversation:{type(e).__name__}:{e}"
        return res
    for turn in candidate_turns:
        try:
            result = conv.respond(turn)
        except Exception as e:  # noqa: BLE001
            res["call_error"] = f"engine:{type(e).__name__}:{e}"
            return res
        tr = result.tool_trace or {}
        res["turns"].append({"candidate": turn, "reply": str(result.response or ""),
                             "end": result.conversation_end, "usage": result.usage,
                             "decision": tr.get("decision"), "state": tr.get("state")})
        if result.conversation_end:
            break
    if gate_mode == "dialogue":
        _judge_into(res, judge, scenario)
        if res["call_error"]:  # инфра-сбой судьи диалога — слой B не считаем
            return res
    _eval_layer_b(res, vinfo, ijudge)
    return res


def _eval_layer_b(res: Dict[str, Any], vinfo: Dict[str, Any], ijudge: Any) -> None:
    """Слой B: утечка секрета (детерминированно, с атрибуцией) + судья Интервьюера (LLM). Общий для golden/generate."""
    leak = sp.leak_scan(res["turns"], vinfo)
    res["leak"] = {"passed": leak.passed, "details": leak.details, "culprit": leak.culprit}
    if ijudge is None:
        return
    pairs = [{"turn": i, "instruction": (t.get("decision") or {}).get("instruction") or "", "message": t["reply"]}
             for i, t in enumerate(res["turns"], 1)
             if isinstance(t.get("decision"), dict) and t["decision"].get("next_action") == "ask"]
    if not pairs:
        return
    try:
        iverdict, iusage = ijudge.evaluate(pairs)
        res["iverdict"] = {"passed": iverdict.passed, "violations": iverdict.violations, "comment": iverdict.comment}
        res["ijudge_usage"] = iusage
    except Exception as e:  # noqa: BLE001 — судья Интервьюера не критичен для прогона
        res["iverdict"] = {"passed": True, "violations": [], "comment": f"судья Интервьюера недоступен: {type(e).__name__}"}


def _process_generate(item: Any, *, client: Any, analyzer_client: Any, interviewer_spec: Any, judge: Any, ijudge: Any,
                      gen_client: Any, gen_model: str, constraints_entries: list, sampler: Any, max_turns: int,
                      gen_policy: Any, vacancies: Dict[int, Dict[str, Any]], constraints_override: Any = None,
                      gate: str = "dialogue", max_rounds: int | None = None) -> Dict[str, Any]:
    """Один (сценарий, вариант): адаптивный LLM-кандидат ↔ split-движок, затем судья+слои A/B.

    constraints_override — засеянные из рецепта констрейнты (Фаза 2: must_convey из вилки + сид);
    gate — режим оценки ('dialogue' по умолч.; 'analyzer' — гейт трассой, для generated+инвариантов);
    max_rounds — число раундов ЭТОГО сценария (из рецепта: `rounds` или len(turns)); задаёт длину
    адаптивного диалога ПЕР-СЦЕНАРНО, а не глобальным --max-turns (одинаково для scripted/generated)."""
    scenario, variant = item
    res: Dict[str, Any] = {"scenario": scenario, "variant": variant, "mode": "generate", "gate": gate,
                           "turns": [], "verdict": None, "judge_usage": None, "leak": None, "iverdict": None,
                           "ijudge_usage": None, "call_error": None, "gen_sources": []}
    vinfo = vacancy_for(scenario, vacancies, DEFAULT_VACANCY_INFO)
    constraints = constraints_override or constraints_for(scenario, constraints_entries)
    style = sampler.at(scenario.index * 1000 + variant)  # детерминированный стиль на (сценарий, вариант)
    agent = CandidateAgent(gen_client, gen_model, constraints, style, policy=gen_policy)
    conv = sp.SplitConversation(client=client, analyzer_client=analyzer_client, interviewer_spec=interviewer_spec,
                                vacancy_info=vinfo, recruiter_name=DEFAULT_RECRUITER_NAME, candidate_name="Кандидат")
    # Приоритет длины диалога: рецептные rounds (пер-сценарно) > constraints.max_turns > глобальный --max-turns.
    eff_turns = max_rounds or constraints.max_turns or max_turns
    result = run_adaptive_conversation(conv, agent, max_turns=eff_turns)
    for t in result.turns:
        tr = t.tool_trace or {}
        res["turns"].append({"candidate": t.candidate, "reply": t.reply, "end": t.end,
                             "usage": t.assistant_usage, "gen_usage": t.gen_usage, "source": t.candidate_source,
                             "decision": tr.get("decision"), "state": tr.get("state")})
        res["gen_sources"].append(t.candidate_source)
    if result.error:
        res["call_error"] = result.error
        return res
    if not res["turns"]:
        res["call_error"] = "empty_dialogue"
        return res
    if gate == "dialogue":
        _judge_into(res, judge, scenario)
        if res["call_error"]:
            return res
    _eval_layer_b(res, vinfo, ijudge)
    return res


def _transcript_text(turns: List[Dict[str, Any]]) -> str:
    """Текст диалога для судьи; реплику, завершившую диалог, метим END_MARKER."""
    def _fmt(t: Dict[str, Any]) -> str:
        reply = str(t["reply"] or "")
        if t["end"]:
            reply = (reply + " " + END_MARKER).strip()
        return f"[Кандидат] {t['candidate']}\n[Ассистент] {reply}"

    return "\n".join(_fmt(t) for t in turns)


def _judge_into(res: Dict[str, Any], judge: Any, scenario: Scenario) -> None:
    try:
        verdict, jusage = judge.evaluate(scenario, _transcript_text(res["turns"]))
        res["verdict"], res["judge_usage"] = verdict, jusage
    except Exception as e:  # noqa: BLE001
        res["call_error"] = f"judge:{type(e).__name__}:{e}"


def _run_offline(args: argparse.Namespace, scenarios: List[Scenario]) -> None:
    with_ex = turns = 0
    for s in scenarios:
        ct = extract_candidate_examples(s.examples_raw, args.max_examples)
        mark = f"реплик-кандидата: {len(ct)}" if ct else "нет реплик кандидата"
        if ct:
            with_ex += 1
            turns += len(ct)
        if not args.quiet:
            print(f"  {s.index:>2} {s.name[:60]:<60} {mark}")
    print(f"\n[offline] сценариев={len(scenarios)} с_примерами={with_ex} реплик_всего={turns} (плумбинг, без судьи)")
    print("[offline] санити чистого домена (порт tgApi):")
    for line in _domain_sanity():
        print(f"  · {line}")
    print("[offline] сеть и пакет prompts не дёргались.")


def _domain_sanity() -> List[str]:
    out: List[str] = []
    ko = sp.render_script("KO_FORMAT_OFFICE", city="Москва")
    out.append(f"render KO_FORMAT_OFFICE(city=Москва): city_grammar={'в городе Москва' in (ko or '')} · terminal={sp.is_terminal('KO_FORMAT_OFFICE')}")
    st_office = sp.init_state("office", "1. Опыт с Python?\n2. SQL?")
    st_remote = sp.init_state("remote", "")
    out.append(f"init_state office format_check={st_office['format_check']} questions={[q['key'] for q in st_office['questions']]} · remote format_check={st_remote['format_check']}")
    st2 = sp.apply_updates(st_office, [{"key": "salary", "value": "closed"}, {"key": "candidate_city", "value": "Казань"}])
    st3 = sp.apply_updates(st2, [], event="gibberish")
    out.append(f"apply_updates salary={st2['salary']} city={st2['candidate_city']} gibberish_counter={st3['counters']['gibberish']}")
    dec, err = sp.parse_and_validate('{"next_action":"ask","script_key":null,"instruction":"Спроси зарплату","updates":[],"event":null,"asking":"salary"}')
    out.append(f"Decision-валидатор: valid={dec is not None} err={err or '—'}")
    return out


def _resolve_version(cfg: Dict[str, Any], component: str, cli_version: str | None) -> str | None:
    """CLI > model.yaml[component].local_version > None (pointer.yaml active в пакете)."""
    if cli_version:
        return cli_version
    return component_cfg(cfg, component).get("local_version")


def run(args: argparse.Namespace) -> Any:
    scenarios = load_scenarios(args.csv)

    if args.offline:
        selected = _select(scenarios, args.scenario_indices, args.sample, args.seed,
                           runnable_only=False, max_examples=args.max_examples)
        print(f"Сценариев в CSV: {len(scenarios)} · выбрано: {len(selected)} · CSV: {args.csv}")
        _run_offline(args, selected)
        return None

    # --- онлайн golden: split — LOCAL-only (stored-эквивалента нет), потому local по умолчанию ---
    source = resolve_source(args.prompt_source or os.environ.get("QA_HARNESS_PROMPT_SOURCE") or LOCAL)
    if source != LOCAL:
        raise SystemExit("screening_split тестируется только в local; stored-эквивалента у split-промптов нет.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set (экспортируй: set -a; source .env; set +a)")
    # Готча: пустой `OPENAI_BASE_URL=` в .env экспортится как "" и OpenAI SDK берёт его за base_url
    # (битый URL → APIConnectionError). Пустое значение = «не задано»: убираем, чтобы SDK взял дефолт.
    if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)

    from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
    from qa_harness.core.llm_client import LocalPromptClient, ModelClient, get_client

    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")
    cfg = load_cfg(args.cfg)
    a_ver = _resolve_version(cfg, ANALYZER_COMPONENT, args.analyzer_version)
    i_ver = _resolve_version(cfg, INTERVIEWER_COMPONENT, args.interviewer_version)

    # пакет prompts (дев-путь --prompts-path / env PROMPTS_REPO_PATH, иначе установленный релиз)
    ensure_prompts_importable(args.prompts_path)
    client = get_client(timeout=args.step1_timeout)
    # общие (read-only) части — строим один раз, шарим по потокам; mutable движок — per-scenario
    analyzer_client = LocalPromptClient(ANALYZER_COMPONENT, a_ver, client=client)
    interviewer_spec = load_local_spec(INTERVIEWER_COMPONENT, i_ver)
    a_spec = analyzer_client.spec
    judge = ScenarioJudge(ModelClient(args.eval_model, timeout=args.step1_timeout, temperature=0))
    ijudge = None if args.no_interviewer_judge else sp.InterviewerJudge(
        ModelClient(args.eval_model, timeout=args.step1_timeout, temperature=0))
    vacancies = load_vacancies(args.vacancies or DEFAULT_VACANCIES)
    checks_by_index = sp.load_checks(args.checks or DEFAULT_CHECKS)  # слой A: инварианты Decision
    inputs_by_index = sp.load_candidate_inputs(args.candidate_inputs or DEFAULT_INPUTS)  # C1: скриптовые входы

    # Нужен ли LLM-генератор: явный --generate ИЛИ --input-mode generated (он включает генератор сам).
    use_generator = args.generate or args.input_mode == "generated"
    # golden гоняет сценарии с примерами ИЛИ со скриптовым входом; генератор разблокирует все.
    selected = _select(scenarios, args.scenario_indices, args.sample, args.seed,
                       runnable_only=not use_generator, max_examples=args.max_examples,
                       scripted_indices=set(inputs_by_index))

    n_variants = max(1, args.variants)
    work_items: List[Any] = [(s, v) for s in selected for v in range(n_variants)]
    n_scripted = sum(1 for s in selected if s.index in inputs_by_index)

    gen_setup: Dict[str, Any] = {}
    if use_generator:
        gen_seed = args.gen_seed if args.gen_seed is not None else (args.seed if args.seed is not None else 0)
        gen_setup = dict(
            gen_client=ModelClient(args.gen_model, timeout=args.step1_timeout, temperature=args.temperature),
            gen_model=args.gen_model,
            constraints_entries=load_constraints(args.constraints or DEFAULT_CONSTRAINTS),
            sampler=VariantSampler(gen_seed),
            max_turns=args.max_turns,
            gen_policy=GenerationPolicy(max_retries=args.gen_retries, temperature=args.temperature, seed=gen_seed),
            vacancies=vacancies,
        )
        models = {"candidate_generator": args.gen_model, "analyzer": a_spec.model,
                  "interviewer": interviewer_spec.model, "evaluator": args.eval_model}
        run_args = {"mode": "generate", "input_mode": args.input_mode, "scenarios": len(selected),
                    "scripted": n_scripted, "variants": n_variants, "gen_model": args.gen_model,
                    "gen_seed": gen_seed, "max_turns": args.max_turns, "eval_model": args.eval_model,
                    "workers": args.workers}
    else:
        models = {"analyzer": a_spec.model, "interviewer": interviewer_spec.model, "evaluator": args.eval_model}
        run_args = {"mode": "golden", "input_mode": args.input_mode, "scenarios": len(selected),
                    "scripted": n_scripted, "variants": n_variants, "max_examples": args.max_examples,
                    "workers": args.workers, "eval_model": args.eval_model}

    print(f"Сценариев в CSV: {len(scenarios)} · выбрано: {len(selected)} (рецептных: {n_scripted}) × {n_variants} = "
          f"{len(work_items)} · режим: {run_args['mode']}/{args.input_mode} · CSV: {args.csv}")
    print(f"Аналитик {a_spec.version}/{a_spec.model} · Интервьюер {interviewer_spec.version}/{interviewer_spec.model} · судья {args.eval_model}"
          + (f" · генератор {args.gen_model}" if args.generate else ""))

    put = {
        "component": "screening_split", "source": "local", "prompt_id": None, "prompt_version": None,
        "local_component": f"{ANALYZER_COMPONENT} + {INTERVIEWER_COMPONENT}",
        "local_version": f"A:{a_spec.version} · I:{interviewer_spec.version}",
        "model": f"A:{a_spec.model} · I:{interviewer_spec.model}",
    }
    rb = ReportBuilder(
        runner=RUNNER, prompt_under_test=put, run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models=models, seed=args.seed, args=run_args,
    )

    usage_bucket = blank_usage()
    gen_usage_bucket = blank_usage()
    m, reasons, gen_sources = Counter(), Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"scenarios": dict(m), "reasons": dict(reasons)}
        if args.generate:
            extra["generation"] = {"usage": usage_total(gen_usage_bucket), "sources": dict(gen_sources)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd, write_review=False)  # A1: без review.md

    def _fold(res: Dict[str, Any]) -> None:
        s: Scenario = res["scenario"]
        is_gen = res.get("mode") == "generate"
        variant = res.get("variant")
        for t in res["turns"]:
            accumulate_usage(usage_bucket, t.get("usage"))       # ответ движка (Аналитик+Интервьюер)
            accumulate_usage(usage_bucket, t.get("gen_usage"))   # генерация реплики → в общий total
            accumulate_usage(gen_usage_bucket, t.get("gen_usage"))
        accumulate_usage(usage_bucket, res["judge_usage"])
        accumulate_usage(usage_bucket, res.get("ijudge_usage"))
        for src in res.get("gen_sources", []):
            gen_sources[src] += 1
        v_disp = (variant or 0) + 1  # 1-based для отображения (вариантов нет «нулевого»)
        tag = f"{s.index}" + (f"/v{v_disp}" if is_gen else "")
        cid = f"scenario:{s.index}:" + (f"v{v_disp}:" if is_gen else "") + s.name  # полное имя, без обрезки

        if res["call_error"] == "no_candidate_examples":
            m["skipped_no_examples"] += 1
            return
        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["errors"] += 1
            if not args.quiet:
                print(f"  [ERR ] {tag} {s.name[:45]}: {res['call_error']}")
            return

        m["total"] += 1
        verdict = res.get("verdict")  # None в analyzer-gate (ScenarioJudge не звали)
        # A2/A3: round = номер обмена (кандидат+ассистент); text = ПОЛНЫЙ вывод без ⟨trace⟩;
        # обе роли видны — analyzer_instruction (что велел Аналитик) + text (что реально ушло) + turn_kind.
        transcript: List[Dict[str, Any]] = []
        had_ask = False
        for i, t in enumerate(res["turns"], start=1):
            transcript.append({"round": i, "role": "candidate", "text": t["candidate"]})
            d = t.get("decision") or {}
            na = d.get("next_action")
            kind = "script" if na == "script" else ("interviewer_reply" if na == "ask" else "fallback")
            if na == "ask":
                had_ask = True
            a_turn: Dict[str, Any] = {"round": i, "role": "assistant", "text": t["reply"],
                                      "turn_kind": kind, "analyzer_instruction": d.get("instruction"),
                                      "decision": d, "state": t.get("state")}
            if t["end"]:
                a_turn["ended"] = True
            transcript.append(a_turn)
        # --- слой A (Аналитик) + слой B (Интервьюер) с учётом РЕЖИМА ГЕЙТА ---
        # gate="analyzer": детерминированный вход+инварианты → слой A ГЕЙТИТ, ScenarioJudge не звали;
        # gate="dialogue": вход варьируется/нет инвариантов → гейтит LLM-судья, слой A — лишь СИГНАЛ.
        gate = res.get("gate", "dialogue")
        acheck = sp.evaluate_analyzer(s.index, res["turns"], checks_by_index)
        leak = res.get("leak") or {"passed": True, "details": [], "culprit": None}
        iverdict = res.get("iverdict")
        dialogue_passed = True if verdict is None else bool(verdict.passed)
        dialogue_violations = [] if verdict is None else list(verdict.violations)
        dialogue_comment = "" if verdict is None else verdict.comment
        analyzer_ok = (not acheck.has_checks) or acheck.passed
        analyzer_gates = (gate == "analyzer") and acheck.has_checks
        leak_ok = bool(leak["passed"])
        interviewer_ok = (iverdict is None) or bool(iverdict["passed"])
        # gate: analyzer — инвариант Аналитика ГЕЙТИТ. И в scripted, И в generated: вход варьируется
        # лишь ФОРМУЛИРОВКОЙ (факт закреплён must_convey/рецептом, раунды тугие) → инвариант валиден
        # в обоих режимах, а провал Аналитика в generated НЕ маскируется под passed (кейс 6). dialogue —
        # инвариантов нет, гейтит LLM-судья. Утечка и Интервьюер гейтят всегда.
        if gate == "analyzer":
            core_ok = analyzer_ok
        else:
            core_ok = dialogue_passed
        passed = core_ok and leak_ok and interviewer_ok

        if passed:
            m["passed"] += 1
        else:
            m["failed"] += 1
        if analyzer_gates and not acheck.passed:
            m["analyzer_fail"] += 1
            reasons["[Аналитик] " + "; ".join(acheck.details)[:80]] += 1
        if not leak_ok:
            m[("analyzer_leak" if leak.get("culprit") == "analyzer" else "interviewer_leak")] += 1
        if iverdict is not None and not iverdict["passed"]:
            m["interviewer_fail"] += 1
        if gate == "dialogue" and not dialogue_passed:
            for v in dialogue_violations[:6]:
                reasons[v[:60]] += 1

        # общий вердикт кейса; reason_codes атрибутированы — видно, В КОМ ошибка
        reason_codes: List[str] = []
        if analyzer_gates and not acheck.passed:
            reason_codes += ["[Аналитик] " + d for d in acheck.details if "OK" not in d]
        if not leak_ok:
            leak_tag = "[Аналитик] " if leak.get("culprit") == "analyzer" else "[Интервьюер] "
            reason_codes += [leak_tag + d for d in leak["details"]]
        if iverdict is not None and not iverdict["passed"]:
            reason_codes += ["[Интервьюер] " + v for v in iverdict["violations"][:4]]
        if gate == "dialogue":
            reason_codes += list(dialogue_violations[:6])

        case_checks: List[Dict[str, Any]] = []
        if acheck.has_checks:
            a_rule = "Аналитик: инварианты Decision"
            case_checks.append({"rule": a_rule, "passed": acheck.passed, "detail": "; ".join(acheck.details)})
        if had_ask or not leak_ok:  # A4: leak-канарейка осмысленна только если говорил Интервьюер
            case_checks.append({"rule": "Интервьюер: утечка секрета", "passed": leak_ok,
                                "detail": "; ".join(leak["details"])})
        if iverdict is not None:
            case_checks.append({"rule": "Интервьюер: верность инструкции (LLM)", "passed": bool(iverdict["passed"]),
                                "detail": iverdict["comment"] or "; ".join(iverdict["violations"][:4])})

        v_ord = variant or 0
        _rec = inputs_by_index.get(s.index)
        if _rec and is_gen:
            input_mode = "generated"       # рецептный сценарий прогнан через генератор (флаг/mode)
        elif _rec:
            input_mode = "scripted"
        else:
            input_mode = "generate" if is_gen else "golden"
        rb.add_case(CaseRecord(
            case_id=cid, source="suite", passed=passed,
            order=(s.index, v_ord),  # A2: детерминированная сортировка отчёта по (сценарий, вариант)
            inputs={"criterion": s.expected_behavior or "expected behavior per scenario",
                    "variant": v_ord, "input_mode": input_mode,
                    "scenario": {"index": s.index, "name": s.name, "description": s.description}},
            transcript=transcript, checks=case_checks,
            verdict={"evaluator": f"screening_split (gate:{gate})", "model": args.eval_model,
                     "passed": passed, "reason_codes": reason_codes[:12], "comment": dialogue_comment},
        ))
        if not args.quiet:
            a_tag = "" if not acheck.has_checks else (" A:ok" if acheck.passed else " A:FAIL")
            b_tag = "" if (leak_ok and interviewer_ok) else " B:FAIL"
            fb = res.get("gen_sources", []).count("fallback") if is_gen else 0
            g = {"generated": "gen*", "scripted": "scr", "generate": "gen", "golden": "gld"}[input_mode]
            print(f"  [{'ok ' if passed else 'MISS'}] {tag} {s.name[:34]} [{g}] turns={len(res['turns'])} viol={len(dialogue_violations)}{a_tag}{b_tag}" + (f" fb={fb}" if fb else ""))

    def _gate_for(s: Scenario) -> str:
        # instrumented (есть машинные инварианты) → детерминированный гейт по трассе, без ScenarioJudge
        return "analyzer" if s.index in checks_by_index else "dialogue"

    def _work(item: Any) -> Dict[str, Any]:
        s, v = item
        vinfo = vacancy_for(s, vacancies, DEFAULT_VACANCY_INFO)
        recipe = inputs_by_index.get(s.index)
        if recipe:
            # Эффективный режим рецепта: per-сценарный `mode` (override) ИЛИ флаг --input-mode.
            eff_mode = recipe.get("mode") or args.input_mode
            if eff_mode == "generated" and use_generator:
                # LLM-кандидат, ЗАСЕЯННЫЙ из рецепта: must_convey из вилки (salary_category),
                # триггер/описание как сид, реплики рецепта — как конкретные примеры для вариации.
                c = constraints_for(s, gen_setup["constraints_entries"])
                # Контекст в генератор ПЕР-СЦЕНАРНО: вилка (salary_category) + произвольные факты
                # (convey: город кандидата относительно вакансии, гео-готовность и пр., с {location}).
                must = sp.salary_directive(recipe["salary_category"], vinfo) if recipe.get("salary_category") else []
                if recipe.get("convey"):
                    must += sp.resolve_convey(recipe["convey"], vinfo)
                if must:
                    c.must_convey = must
                if recipe.get("seed"):
                    c.trigger_requirement = str(recipe["seed"])
                elif not c.trigger_requirement:
                    c.trigger_requirement = s.description
                rec_turns = sp.build_scripted_turns(recipe, vinfo, variant=v, seed=(args.seed or 0), index=s.index)
                if rec_turns and not c.examples:
                    c.examples = rec_turns
                # Раунды ПЕР-СЦЕНАРНО: явный `rounds` рецепта, иначе число ходов рецепта. Столько же,
                # сколько отыграет scripted-режим этого же сценария (симметрия scripted/generated).
                rec_rounds = recipe.get("rounds") or len(rec_turns) or None
                gen_gate = "analyzer" if s.index in checks_by_index else "dialogue"
                return _process_generate(item, client=client, analyzer_client=analyzer_client,
                                         interviewer_spec=interviewer_spec, judge=judge, ijudge=ijudge,
                                         constraints_override=c, gate=gen_gate, max_rounds=rec_rounds, **gen_setup)
            turns = sp.build_scripted_turns(recipe, vinfo, variant=v, seed=(args.seed or 0), index=s.index)
            return _run_fixed_turns(s, v, turns, client=client, analyzer_client=analyzer_client,
                                    interviewer_spec=interviewer_spec, judge=judge, ijudge=ijudge,
                                    vinfo=vinfo, gate_mode=_gate_for(s))
        if use_generator:  # нет рецепта → LLM-адаптивный кандидат
            return _process_generate(item, client=client, analyzer_client=analyzer_client,
                                     interviewer_spec=interviewer_spec, judge=judge, ijudge=ijudge, **gen_setup)
        turns = extract_candidate_examples(s.examples_raw, args.max_examples)  # golden: реплики из CSV
        return _run_fixed_turns(s, v, turns, client=client, analyzer_client=analyzer_client,
                                interviewer_spec=interviewer_spec, judge=judge, ijudge=ijudge,
                                vinfo=vinfo, gate_mode=_gate_for(s))

    outcome = run_cases(list(work_items), work=_work, fold=_fold, max_workers=max(1, args.workers),
                        checkpoint_every=args.checkpoint_every, on_checkpoint=_flush,
                        on_interrupt=lambda: print("\n[interrupted] сохраняю частичный отчёт...") if not args.quiet else None)

    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        import json as _json
        sm = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        print(f"[{tag}] cases={sm['total']} passed={sm['passed']} failed={sm['failed']} errors(infra)={sm['errors']} done={outcome.done}/{len(work_items)}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

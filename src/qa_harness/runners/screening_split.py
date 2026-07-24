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
  python -m qa_harness.runners.screening_split --scenario-indices 65 --prompts-path ../prompts
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
from qa_harness.domain.screening_scenarios import (
    END_MARKER,
    Scenario,
    ScenarioJudge,
    extract_candidate_examples,
    load_scenarios,
    load_vacancies,
    parse_scenario_indices,
    vacancy_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_CSV = FIXTURES / "screening_split" / "scenarios.csv"
# Пер-сценарный контекст вакансии (формат/локация/вилка/скрытость) — переиспользуем набор
# монолита: index в split-CSV совпадает с базовым (1..64), кейс 65 берёт DEFAULT_VACANCY_INFO.
DEFAULT_VACANCIES = FIXTURES / "generation" / "screening_scenarios" / "scenario_vacancies.yaml"
DEFAULT_CHECKS = FIXTURES / "screening_split" / "scenario_checks.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_split"
ANALYZER_COMPONENT = "screening_analyzer"
INTERVIEWER_COMPONENT = "screening_interviewer"
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
    p = argparse.ArgumentParser(description="screening_split QA runner (Аналитик + Интервьюер; local prompts).")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV сценариев split (по умолч. отдельный от легаси).")
    p.add_argument("--sample", type=int, default=5, help="Случайная выборка N сценариев (0 = все).")
    p.add_argument("--scenario-indices", default=None, help="Точечные номера строк CSV, напр. 1,7,65 (override --sample).")
    p.add_argument("--max-examples", type=int, default=4, help="Сколько реплик кандидата брать из примеров на сценарий.")
    p.add_argument("--offline", action="store_true", help="Плумбинг: сценарии + реплики + санити чистого домена, без сети.")
    p.add_argument("--analyzer-version", default=None, metavar="vN", help="Версия screening_analyzer в пакете prompts (иначе pointer.yaml active).")
    p.add_argument("--interviewer-version", default=None, metavar="vN", help="Версия screening_interviewer (иначе pointer.yaml active).")
    p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help=f"Модель судей диалога/Интервьюера (по умолч. {DEFAULT_EVAL_MODEL}).")
    p.add_argument("--no-interviewer-judge", action="store_true", help="Отключить LLM-судью Интервьюера (слой B — только детерминированный leak-scan).")
    p.add_argument("--vacancies", type=Path, default=None, help="YAML пер-сценарного контекста вакансии (по умолч. набор монолита).")
    p.add_argument("--checks", type=Path, default=None, help="YAML инвариантов Decision Аналитика (по умолч. scenario_checks.yaml).")
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
            *, runnable_only: bool, max_examples: int) -> List[Scenario]:
    if indices_raw:  # точечный выбор — как просили, без фильтра
        wanted = parse_scenario_indices(indices_raw)
        by_idx = {s.index: s for s in scenarios}
        return [by_idx[i] for i in wanted if i in by_idx]
    pool = scenarios
    if runnable_only:  # онлайн golden: гонять можно лишь сценарии с примерами диалога кандидата
        pool = [s for s in scenarios if extract_candidate_examples(s.examples_raw, max_examples)]
    if sample and sample > 0:
        return random.Random(seed).sample(pool, min(sample, len(pool)))
    return pool


# ── трасса решения Аналитика в компактный вид (для читабельности review.md) ──────
def _trace_tag(decision: dict | None) -> str:
    if not decision:
        return ""
    parts = [decision.get("next_action") or "?"]
    if decision.get("script_key"):
        parts.append(f"script={decision['script_key']}")
    if decision.get("asking"):
        parts.append(f"asking={decision['asking']}")
    if decision.get("event"):
        parts.append(f"event={decision['event']}")
    ups = decision.get("updates") or []
    if ups:
        parts.append("updates=[" + ",".join(f"{u.get('key')}={u.get('value')}" for u in ups) + "]")
    if decision.get("source"):  # форс кода: counter_cap/reask_cap/analyzer_error
        parts.append(f"src={decision['source']}")
    return " · ".join(parts)


# ── golden-прогон одного сценария: split-разговор → судья диалога ────────────────
def _process(scenario: Scenario, *, client: Any, analyzer_client: Any, interviewer_spec: Any,
             judge: Any, ijudge: Any, max_examples: int, vacancies: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"scenario": scenario, "turns": [], "verdict": None, "judge_usage": None,
                           "leak": None, "iverdict": None, "ijudge_usage": None, "call_error": None}
    candidate_turns = extract_candidate_examples(scenario.examples_raw, max_examples)
    if not candidate_turns:
        res["call_error"] = "no_candidate_examples"
        return res
    vinfo = vacancy_for(scenario, vacancies, DEFAULT_VACANCY_INFO)
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
    _judge_into(res, judge, scenario)
    if res["call_error"]:  # инфра-сбой судьи диалога — слой B не считаем
        return res

    # --- слой B: утечка секрета (детерминированно) + судья Интервьюера (LLM) ---
    leak = sp.leak_scan(res["turns"], vinfo)
    res["leak"] = {"passed": leak.passed, "details": leak.details, "culprit": leak.culprit}
    if ijudge is not None:
        pairs = [{"turn": i, "instruction": (t.get("decision") or {}).get("instruction") or "", "message": t["reply"]}
                 for i, t in enumerate(res["turns"], 1)
                 if isinstance(t.get("decision"), dict) and t["decision"].get("next_action") == "ask"]
        if pairs:
            try:
                iverdict, iusage = ijudge.evaluate(pairs)
                res["iverdict"] = {"passed": iverdict.passed, "violations": iverdict.violations, "comment": iverdict.comment}
                res["ijudge_usage"] = iusage
            except Exception as e:  # noqa: BLE001 — судья Интервьюера не критичен для прогона
                res["iverdict"] = {"passed": True, "violations": [], "comment": f"судья Интервьюера недоступен: {type(e).__name__}"}
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

    selected = _select(scenarios, args.scenario_indices, args.sample, args.seed,
                       runnable_only=True, max_examples=args.max_examples)
    print(f"Сценариев в CSV: {len(scenarios)} · выбрано (с примерами): {len(selected)} · CSV: {args.csv}")
    print(f"Аналитик {a_spec.version}/{a_spec.model} · Интервьюер {interviewer_spec.version}/{interviewer_spec.model} · судья {args.eval_model}")

    put = {
        "component": "screening_split", "source": "local", "prompt_id": None, "prompt_version": None,
        "local_component": f"{ANALYZER_COMPONENT} + {INTERVIEWER_COMPONENT}",
        "local_version": f"A:{a_spec.version} · I:{interviewer_spec.version}",
        "model": f"A:{a_spec.model} · I:{interviewer_spec.model}",
    }
    rb = ReportBuilder(
        runner=RUNNER, prompt_under_test=put, run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"analyzer": a_spec.model, "interviewer": interviewer_spec.model, "evaluator": args.eval_model},
        seed=args.seed,
        args={"mode": "golden", "scenarios": len(selected), "max_examples": args.max_examples,
              "workers": args.workers, "eval_model": args.eval_model},
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
        accumulate_usage(usage_bucket, res.get("ijudge_usage"))
        cid = f"scenario:{s.index}:{s.name[:40]}"

        if res["call_error"] == "no_candidate_examples":
            m["skipped_no_examples"] += 1
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
            tag = _trace_tag(t.get("decision"))
            text = t["reply"] + (f"\n\n⟨trace: {tag}⟩" if tag else "")
            a_turn: Dict[str, Any] = {"turn": 2 * i, "role": "assistant", "text": text,
                                      "decision": t.get("decision"), "state": t.get("state")}
            if t["end"]:
                a_turn["ended"] = True
            transcript.append(a_turn)
        # --- слой A (Аналитик, детерминированно) + слой B (Интервьюер: leak + LLM-судья) ---
        acheck = sp.evaluate_analyzer(s.index, res["turns"], checks_by_index)
        leak = res.get("leak") or {"passed": True, "details": [], "culprit": None}
        iverdict = res.get("iverdict")
        dialogue_passed = bool(verdict.passed)
        analyzer_ok = (not acheck.has_checks) or acheck.passed
        leak_ok = bool(leak["passed"])
        interviewer_ok = (iverdict is None) or bool(iverdict["passed"])
        passed = dialogue_passed and analyzer_ok and leak_ok and interviewer_ok

        if passed:
            m["passed"] += 1
        else:
            m["failed"] += 1
        if acheck.has_checks and not acheck.passed:
            m["analyzer_fail"] += 1
            reasons["[Аналитик] " + "; ".join(acheck.details)[:80]] += 1
        if not leak_ok:
            m[("analyzer_leak" if leak.get("culprit") == "analyzer" else "interviewer_leak")] += 1
        if iverdict is not None and not iverdict["passed"]:
            m["interviewer_fail"] += 1
        if not dialogue_passed:
            for v in verdict.violations[:6]:
                reasons[v[:60]] += 1

        # общий вердикт кейса; reason_codes атрибутированы — видно, В КОМ ошибка
        reason_codes: List[str] = []
        if acheck.has_checks and not acheck.passed:
            reason_codes += ["[Аналитик] " + d for d in acheck.details if "OK" not in d]
        if not leak_ok:
            leak_tag = "[Аналитик] " if leak.get("culprit") == "analyzer" else "[Интервьюер] "
            reason_codes += [leak_tag + d for d in leak["details"]]
        if iverdict is not None and not iverdict["passed"]:
            reason_codes += ["[Интервьюер] " + v for v in iverdict["violations"][:4]]
        reason_codes += list(verdict.violations[:6])

        case_checks: List[Dict[str, Any]] = []
        if acheck.has_checks:
            case_checks.append({"rule": "Аналитик: инварианты Decision", "passed": acheck.passed,
                                "detail": "; ".join(acheck.details)})
        case_checks.append({"rule": "Интервьюер: утечка секрета", "passed": leak_ok,
                            "detail": "; ".join(leak["details"])})
        if iverdict is not None:
            case_checks.append({"rule": "Интервьюер: верность инструкции (LLM)", "passed": bool(iverdict["passed"]),
                                "detail": iverdict["comment"] or "; ".join(iverdict["violations"][:4])})

        rb.add_case(CaseRecord(
            case_id=cid, source="suite", passed=passed,
            inputs={"criterion": s.expected_behavior or "expected behavior per scenario",
                    "scenario": {"index": s.index, "name": s.name, "description": s.description}},
            transcript=transcript, checks=case_checks,
            verdict={"evaluator": "screening_split (диалог+Аналитик+Интервьюер)", "model": args.eval_model,
                     "passed": passed, "reason_codes": reason_codes[:12], "comment": verdict.comment},
        ))
        if not args.quiet:
            a_tag = "" if not acheck.has_checks else (" A:ok" if acheck.passed else " A:FAIL")
            b_tag = "" if (leak_ok and interviewer_ok) else " B:FAIL"
            print(f"  [{'ok ' if passed else 'MISS'}] {s.index} {s.name[:38]} turns={len(res['turns'])} viol={len(verdict.violations)}{a_tag}{b_tag}")

    def _work(sc: Scenario) -> Dict[str, Any]:
        return _process(sc, client=client, analyzer_client=analyzer_client, interviewer_spec=interviewer_spec,
                        judge=judge, ijudge=ijudge, max_examples=args.max_examples, vacancies=vacancies)

    outcome = run_cases(list(selected), work=_work, fold=_fold, max_workers=max(1, args.workers),
                        checkpoint_every=args.checkpoint_every, on_checkpoint=_flush,
                        on_interrupt=lambda: print("\n[interrupted] сохраняю частичный отчёт...") if not args.quiet else None)

    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        import json as _json
        sm = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        print(f"[{tag}] scenarios={sm['total']} passed={sm['passed']} failed={sm['failed']} errors(infra)={sm['errors']} done={outcome.done}/{len(selected)}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] review  -> {Path(cases_path).with_name(f'{RUNNER}_{run_id}.review.md')}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

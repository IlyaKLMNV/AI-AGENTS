"""Боевой тест анти-зацикливания (универсальный no_progress-cap + существующие счётчики/reask-cap).

Настойчивый переспрашиватель (по мотивам реальных 6-ходовых лупов из отчётов) должен ЗАВЕРШИТЬСЯ
в пределах кап, а не крутиться бесконечно. Гоняет РЕАЛЬНЫЙ Аналитик (LLM) через движок; Интервьюер и
стор — фейковые: текст Интервьюера на счётчик не влияет, а OpenAI-conversation не нужен, поэтому
токены тратит ТОЛЬКО Аналитик (по одному вызову на ход). Один режим — боевой.

Как это работает (счётчик — в КОДЕ, не в LLM):
- на входе строим контекст вакансии (build_context) + стартовый state (init_state) в in-memory сторе;
- на каждый ход движок зовёт реальный Аналитик: analyzer.run(context, state, message) -> Decision;
- применяет updates к state (apply_updates), считает progress_signature; если N ходов подряд сигнатура
  не изменилась (no_progress >= NO_PROGRESS_CAP) и решение не терминальное — форсит STOP_PERSISTENT/FINISH;
- существующие пороги (gibberish/bot/demand/pause/contact_source) и лимит переспросов работают как прежде.
Тест ассертит: диалог завершился (conversation_end) не позже expect.ended_by, нужным механизмом/скриптом.

Запуск: python -m qa_harness.runners.screening_counters --prompts-path ../prompts
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from qa_harness.core import (
    LOCAL,
    accumulate_usage,
    add_prompt_source_args,
    blank_usage,
    ensure_prompts_importable,
    resolve_source,
    usage_total,
)
from qa_harness.core.llm_client import LocalPromptClient, get_client
from qa_harness.domain.screening_split import (
    ScreeningAnalyzer,
    ScreeningSplitEngine,
    build_context,
    candidate_source,
    init_state,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_FIXTURE = FIXTURES / "screening_split" / "counter_loops.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_counters"
ANALYZER_COMPONENT = "screening_analyzer"
RECRUITER, CANDIDATE = "Анна", "Кандидат"

DEFAULT_VACANCY: Dict[str, Any] = {
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


class _FakeStore:
    """In-memory стор одного диалога: движку хватает load()+save_state() (create_thread не зовём)."""

    def __init__(self, state: dict, context: str, location: str, contact_source: str) -> None:
        self.doc = {"state": state, "finished": False, "context": context,
                    "location": location, "contact_source": contact_source}

    def load(self, _cid: Any) -> dict:
        return self.doc

    def save_state(self, _cid: Any, state: dict, finished: bool = False) -> None:
        self.doc["state"] = state
        self.doc["finished"] = finished


class _FakeInterviewer:
    """Текст Интервьюера на счётчик не влияет — не тратим на него токены."""

    def run(self, _cid: Any, _instruction: str, _message: str):
        return ("(ответ интервьюера — в этом тесте не важен)", blank_usage())


def _run_case(case: Dict[str, Any], analyzer_client: Any) -> Dict[str, Any]:
    """Один луп-кейс: реальный Аналитик + движок, до завершения или исчерпания ходов."""
    vac = {**DEFAULT_VACANCY, **(case.get("vacancy") or {})}
    accounts = [{"type": case["source_type"]}] if case.get("source_type") else []
    src = candidate_source(accounts)
    context = build_context(RECRUITER, CANDIDATE, src, vac)
    state = init_state(vac.get("work_format", ""), vac.get("questions", "") or "")
    store = _FakeStore(state, context, vac.get("location", "") or "", src)
    engine = ScreeningSplitEngine(store, ScreeningAnalyzer(analyzer_client), _FakeInterviewer(), None)

    trace: List[Dict[str, Any]] = []
    usage = blank_usage()  # токены реального Аналитика (Интервьюер фейковый → его вклад нулевой)
    ended_turn = None
    for i, msg in enumerate(case.get("turns") or [], start=1):
        res = engine.add_message_and_run("cid", msg)
        accumulate_usage(usage, engine.last_usage)  # last_usage за ход = Аналитик (+ретраи/refused-перевызов)
        dec = engine.last_decision or {}
        st = engine.last_state or {}
        trace.append({"turn": i, "candidate": msg, "next_action": dec.get("next_action"),
                      "script_key": dec.get("script_key"), "source": dec.get("source"),
                      "no_progress": st.get("no_progress"), "end": bool(res.conversation_end)})
        if res.conversation_end:
            ended_turn = i
            break

    exp = case.get("expect") or {}
    max_np = max((t.get("no_progress") or 0 for t in trace), default=0)  # запас порога кап (инвариант)
    last = trace[-1] if trace else {}
    sk = last.get("script_key")
    reasons: List[str] = []
    # (1) ГЛАВНЫЙ ассерт: завершился и не позже ожидаемого хода. Механизм (source) — телеметрия, не гейт:
    #     Аналитик может законно завершить сам раньше кап (self STOP_*), тогда source пустой — это ок.
    need_end = exp.get("ended_by") is not None or exp.get("terminal") or exp.get("finish")
    if need_end and ended_turn is None:
        reasons.append(f"диалог НЕ завершился за {len(case.get('turns') or [])} ходов (луп не пойман)")
    if ended_turn is not None and exp.get("ended_by") is not None and ended_turn > exp["ended_by"]:
        reasons.append(f"завершился на {ended_turn}-м ходу, ожидалось ≤ {exp['ended_by']}")
    # (2) класс завершения: STOP_* для лупа / FINISH для кооперативного (не рубить здорового капом)
    term = exp.get("terminal") or ("FINISH" if exp.get("finish") else None)
    if term == "FINISH" and sk != "FINISH":
        reasons.append(f"ожидался FINISH (здоровый диалог), получили {sk}")
    if term == "STOP" and not str(sk or "").startswith("STOP"):
        reasons.append(f"ожидался STOP_*, получили {sk}")
    # (3) точный скрипт — только где механизм детерминирован (регрессия счётчик/reask-cap)
    if exp.get("script") and sk != exp["script"]:
        reasons.append(f"script_key={sk}, ожидался {exp['script']}")
    # (4) инвариант запаса: max no_progress за диалог ≤ порога (кооперативные) — на нём стоит NO_PROGRESS_CAP
    if exp.get("max_no_progress") is not None and max_np > exp["max_no_progress"]:
        reasons.append(f"max no_progress={max_np} > {exp['max_no_progress']} — порог кап на волоске, поднимать")
    return {"name": case.get("name", "?"), "passed": not reasons, "ended_turn": ended_turn,
            "max_no_progress": max_np, "final_script": sk, "final_source": last.get("source"),
            "turns_run": len(trace), "usage": usage_total(usage),
            "reasons": reasons, "expect": exp, "trace": trace}


def run(args: argparse.Namespace) -> None:
    source = resolve_source(args.prompt_source or os.environ.get("QA_HARNESS_PROMPT_SOURCE") or LOCAL)
    if source != LOCAL:
        raise SystemExit("screening_counters — только local (у split-промптов нет stored-эквивалента).")
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set (экспортируй: set -a; source .env; set +a)")
    # Пустой OPENAI_BASE_URL= в .env экспортится как "" и ломает SDK — трактуем как «не задано».
    if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)

    ensure_prompts_importable(args.prompts_path)
    client = get_client(timeout=args.timeout)
    a_ver = args.analyzer_version or os.environ.get("SCREENING_ANALYZER_PROMPT_VERSION")
    analyzer_client = LocalPromptClient(ANALYZER_COMPONENT, a_ver, client=client)

    cases = (yaml.safe_load(Path(args.fixture).read_text(encoding="utf-8")) or {}).get("cases") or []
    spec = analyzer_client.spec
    print(f"[counter-test] Аналитик {spec.version}/{spec.model} · кейсов: {len(cases)} · fixture: {args.fixture}")

    started = datetime.datetime.now()
    results: List[Dict[str, Any]] = []
    for i, c in enumerate(cases, 1):  # per-case стриминг: печатаем сразу по завершении кейса (как в сценариях)
        r = {"n": i, **_run_case(c, analyzer_client)}  # порядковый номер кейса — для навигации в cases.json
        results.append(r)
        print(f"  [{'ok ' if r['passed'] else 'FAIL'}] #{i}/{len(cases)} end@{r['ended_turn']} "
              f"maxNP={r['max_no_progress']} {r['final_script']}/{r['final_source']} · {r['name'][:52]}", flush=True)
        for why in r["reasons"]:
            print(f"          - {why}", flush=True)
    passed = sum(1 for r in results if r["passed"])
    finished = datetime.datetime.now()
    tok = {"input": sum(r["usage"]["input"] for r in results),
           "output": sum(r["usage"]["output"] for r in results),
           "total": sum(r["usage"]["total"] for r in results)}
    turns_total = sum(r["turns_run"] for r in results)

    run_id = started.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_dir) / RUNNER
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"schema_version": "1.0", "kind": "metrics", "runner": RUNNER,
               "meta": {"run_id": run_id, "analyzer": f"{spec.version}/{spec.model}",
                        "started_at": started.isoformat(timespec="seconds"),
                        "finished_at": finished.isoformat(timespec="seconds"),
                        "duration_s": round((finished - started).total_seconds(), 3)},
               "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed,
                           "turns_total": turns_total, "token_usage": tok}}
    (out / f"{RUNNER}_{run_id}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{RUNNER}_{run_id}.cases.json").write_text(
        json.dumps({"kind": "cases", "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[counter-test] passed={passed}/{len(results)} · токены in/out/total="
          f"{tok['input']}/{tok['output']}/{tok['total']} · ходов={turns_total} · "
          f"{round((finished - started).total_seconds(), 1)}s → {out}\\{RUNNER}_{run_id}.*")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Боевой тест анти-зацикливания (no_progress cap + счётчики).")
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--analyzer-version", default=None, metavar="vN",
                   help="Версия screening_analyzer в пакете prompts (иначе pointer.yaml active).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--timeout", type=int, default=90)
    add_prompt_source_args(p)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

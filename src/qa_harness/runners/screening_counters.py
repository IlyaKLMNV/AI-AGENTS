"""Боевой тест анти-зацикливания (универсальный no_progress-cap + событийные счётчики и reask-cap).

Настойчивый переспрашиватель (по мотивам реальных 6-ходовых лупов из отчётов) должен ЗАВЕРШИТЬСЯ
в пределах кап, а не крутиться бесконечно. Гоняет РЕАЛЬНЫЙ Наблюдатель (LLM) через ядро; Интервьюер и
OpenAI-клиент фейковые: текст Интервьюера на счётчик не влияет, поэтому токены тратит ТОЛЬКО
Наблюдатель (по одному вызову на ход). Один режим — боевой.

Как это работает (счётчики — в КОДЕ, не в LLM):
- `create_thread` собирает контекст вакансии, стартовый state и типизированную вилку в in-memory сторе;
- на каждый ход движок зовёт реальный Наблюдатель: observer.run(context, state, message) -> Observation;
- `decide()` применяет факты, начисляет ровно один счётчик, считает progress_signature; если
  `no_progress` дошёл до порога `STALL_BUDGET`, а содержательного правила не нашлось — ход уходит в
  STOP_PERSISTENT/FINISH;
- пороги (gibberish/bot/demand/pause/contact_source) и лимиты переспросов живут в `policy/budgets.py`.
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
from qa_harness.domain.screening_split.store import InMemoryStateStore

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_FIXTURE = FIXTURES / "screening_split" / "counter_loops.yaml"
DEFAULT_FIXTURE_HH = FIXTURES / "screening_split_hh" / "counter_loops.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_counters"
ANALYZER_COMPONENT = "screening_analyzer"
ANALYZER_COMPONENT_HH = "screening_analyzer_hh"
# Ядру нужен контракт `Observation` — он только в v3; на v2 каждый ход уйдёт в фолбэк.
ANALYZER_VERSION = "v3"
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


# hh: форматов у вакансии несколько (`allowed_formats`), «Описание вакансии» вместо обязанностей и
# описания компании, источника контакта в канале нет вовсе.
DEFAULT_VACANCY_HH: Dict[str, Any] = {
    "title": "Инженер пусконаладки",
    "company_name": "ExampleSoft",
    "allowed_formats": ["REMOTE", "HYBRID"],
    "location": "Москва",
    "min_salary": 200000,
    "max_salary": 280000,
    "vacancy_description": ("Пусконаладка и поддержка оборудования на объектах заказчика. "
                            "Подробности: https://hh.ru/vacancy/12345678"),
    "questions": "- Опыт с Python и фреймворками?\n- Сервисы под нагрузкой?\n- Как используете SQL?",
}


class _FakePolicyInterviewer:
    """Текст Интервьюера на счётчик не влияет — не тратим на него токены."""

    def run(self, _instruction: str, _message: str, *, seed: str = "", last_sent: str = ""):
        return ("(ответ интервьюера — в этом тесте не важен)", blank_usage())


class _FakeConversations:
    """`PolicyEngine.create_thread` заводит conversation ради идентификатора — сети для этого не надо."""

    def create(self, **_kwargs: Any) -> Any:
        return type("Conv", (), {"id": "cid"})()


class _FakeClient:
    def __init__(self) -> None:
        self.conversations = _FakeConversations()


def _build_engine(channel: str, vac: Dict[str, Any], case: Dict[str, Any],
                  analyzer_client: Any) -> Any:
    """Движок канала на фейковых Интервьюере и клиенте. Токены тратит только Наблюдатель.

    Стор и `create_thread` настоящие: ядро само соберёт контекст, состояние и типизированную вилку —
    ровно тем путём, каким это произойдёт в проде.
    """
    if channel == "hh":
        from qa_harness.domain.screening_split_hh.policy.engine import PolicyEngine
        from qa_harness.domain.screening_split_hh.policy.observer import ScreeningObserver
    else:
        from qa_harness.domain.screening_split.policy.engine import PolicyEngine
        from qa_harness.domain.screening_split.policy.observer import ScreeningObserver

    engine = PolicyEngine(InMemoryStateStore(), ScreeningObserver(analyzer_client),
                          _FakePolicyInterviewer(), _FakeClient())
    if channel == "hh":
        return engine, engine.create_thread(vac, RECRUITER, CANDIDATE)
    accounts = [{"type": case["source_type"]}] if case.get("source_type") else []
    return engine, engine.create_thread(vac, RECRUITER, CANDIDATE, accounts)


def _run_case(case: Dict[str, Any], analyzer_client: Any, *, channel: str = "tg") -> Dict[str, Any]:
    """Один луп-кейс: реальный Аналитик/Наблюдатель + движок, до завершения или исчерпания ходов."""
    base = DEFAULT_VACANCY_HH if channel == "hh" else DEFAULT_VACANCY
    vac = {**base, **(case.get("vacancy") or {})}
    engine, cid = _build_engine(channel, vac, case, analyzer_client)

    trace: List[Dict[str, Any]] = []
    usage = blank_usage()  # токены реального Аналитика (Интервьюер фейковый → его вклад нулевой)
    ended_turn = None
    for i, msg in enumerate(case.get("turns") or [], start=1):
        res = engine.add_message_and_run(cid, msg)
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

    channel = getattr(args, "channel", "tg") or "tg"
    component = ANALYZER_COMPONENT_HH if channel == "hh" else ANALYZER_COMPONENT
    runner_name = f"{RUNNER}_hh" if channel == "hh" else RUNNER
    fixture = Path(args.fixture) if args.fixture else (
        DEFAULT_FIXTURE_HH if channel == "hh" else DEFAULT_FIXTURE)

    ensure_prompts_importable(args.prompts_path)
    client = get_client(timeout=args.timeout)
    # Не «удобный дефолт», а требование контракта: `Observation` отдаёт только v3.
    a_ver = (args.analyzer_version or os.environ.get("SCREENING_ANALYZER_PROMPT_VERSION")
             or ANALYZER_VERSION)
    analyzer_client = LocalPromptClient(component, a_ver, client=client)

    cases = (yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}).get("cases") or []
    spec = analyzer_client.spec
    print(f"[counter-test] Наблюдатель {component} {spec.version}/{spec.model} · канал: {channel} · "
          f"кейсов: {len(cases)} · fixture: {fixture}")

    started = datetime.datetime.now()
    results: List[Dict[str, Any]] = []
    for i, c in enumerate(cases, 1):  # per-case стриминг: печатаем сразу по завершении кейса (как в сценариях)
        r = {"n": i, **_run_case(c, analyzer_client, channel=channel)}  # порядковый номер кейса — для навигации в cases.json
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
    out = Path(args.out_dir) / runner_name
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"schema_version": "1.0", "kind": "metrics", "runner": runner_name,
               "meta": {"run_id": run_id, "analyzer": f"{component} {spec.version}/{spec.model}",
                        "channel": channel,
                        "started_at": started.isoformat(timespec="seconds"),
                        "finished_at": finished.isoformat(timespec="seconds"),
                        "duration_s": round((finished - started).total_seconds(), 3)},
               "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed,
                           "turns_total": turns_total, "token_usage": tok}}
    (out / f"{runner_name}_{run_id}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{runner_name}_{run_id}.cases.json").write_text(
        json.dumps({"kind": "cases", "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[counter-test] passed={passed}/{len(results)} · токены in/out/total="
          f"{tok['input']}/{tok['output']}/{tok['total']} · ходов={turns_total} · "
          f"{round((finished - started).total_seconds(), 1)}s → {out}\\{RUNNER}_{run_id}.*")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Боевой тест анти-зацикливания (no_progress cap + счётчики).")
    p.add_argument("--channel", choices=["tg", "hh"], default="tg",
                   help="Канал: tg (screening_analyzer, counter_loops.yaml) | hh (screening_analyzer_hh, "
                        "фикстура screening_split_hh: без contact_source, зато с форматом и разъездным).")
    p.add_argument("--fixture", type=Path, default=None,
                   help="YAML кейсов (по умолч. — по каналу).")
    p.add_argument("--analyzer-version", default=None, metavar="vN",
                   help=f"Версия screening_analyzer в пакете prompts (по умолч. {ANALYZER_VERSION} — "
                        "контракт Observation есть только там).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--timeout", type=int, default=90)
    # Версия пинится --analyzer-version (компонент здесь один), stored-эквивалента нет.
    add_prompt_source_args(p, local_only=True, versioned=False)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

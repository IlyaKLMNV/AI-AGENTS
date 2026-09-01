"""Раннер screening_split: тест НОВОГО раздельного скрининга (Аналитик + Интервьюер).

Split = два промпта из пакета `prompts` (`screening_analyzer` — «мозг», строгий JSON
Decision; `screening_interviewer` — «рот», одно сообщение) + КОД-оркестратор (состояние,
счётчики/пороги, фиксированные скрипты), портированный из tgApi 1:1
(qa_harness.domain.screening_split). Тестируется как в проде: тела/схема — из пакета
`prompts` (LOCAL-источник), арифметика состояний — в коде.

СЦЕНАРИИ — отдельный CSV (`tests/fixtures/screening_split/scenarios.csv`, копия golden
монолита + новый зарплатный кейс). Легаси-раннер screening_scenarios и его CSV не трогаем.

Режимы:
- `--offline` — плумбинг (сценарии + реплики кандидата + санити чистого домена) И детерминированный
  ГЕЙТ зарплатного кода: арифметика, гейты годности `salary_claim`, приоритеты решений и
  перерешивание хода при расхождении с Аналитиком. Без сети и без пакета prompts; провал валит
  прогон. Здесь же живут классы, которых живой сценарий добыть не может (для них нужно заставить
  модель ошибиться в служебном поле): порядок «нормализация → гейт updates» и все ветки
  перерешивания;
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
# HH-канал: нет company_info/responsibilities/vacancy_url; формат — список id (allowed_formats);
# факты о вакансии — в свободном «Описание вакансии»; company может быть «рекрутинговое агентство».
DEFAULT_VACANCY_INFO_HH: Dict[str, Any] = {
    "title": "Python Backend Developer",
    "company_name": "ExampleSoft",
    "allowed_formats": ["ON_SITE", "HYBRID"],
    "location": "Москва",
    "min_salary": 200000,
    "max_salary": 280000,
    "vacancy_description": "Продуктовая b2b-платформа. Поддержка и развитие микросервисов, интеграции с продуктами.",
    "questions": "- Опыт с Python и фреймворками?\n- Сервисы под нагрузкой?\n- Как используете SQL?",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="screening_split QA runner (Аналитик + Интервьюер; local prompts).")
    p.add_argument("--channel", choices=["tg", "hh"], default="tg",
                   help="Канал split-скрининга: tg (screening_analyzer/interviewer + fixtures/screening_split) "
                        "| hh (screening_analyzer_hh/interviewer_hh + fixtures/screening_split_hh). Переключает "
                        "движок (domain/screening_split[_hh]), компоненты промптов и набор фикстур.")
    p.add_argument("--csv", type=Path, default=None, help="CSV сценариев (по умолч. — по каналу: fixtures/screening_split[_hh]/scenarios.csv).")
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
    p.add_argument("--engine", choices=["split", "policy"], default="split",
                   help="split — действующий движок (Decision); policy — новая архитектура "
                        "(Наблюдатель + чистое ядро + гарды), есть в обоих каналах. Для policy нужен "
                        "промпт v3 (screening_analyzer[_hh]): --analyzer-version v3")
    # local_only/versioned: stored-эквивалента нет, а версии пинятся покомпонентно
    # (--analyzer-version/--interviewer-version) — общий --local-prompt-version раннер не читает.
    add_prompt_source_args(p, local_only=True, versioned=False)
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
                             "decision": tr.get("decision"), "state": tr.get("state"),
                             "salary": tr.get("salary"),
                             # Трасса нового ядра (движок policy): какое правило выиграло, что видел
                             # наблюдатель, что срезали гарды. У старого движка этих полей нет.
                             "rule": tr.get("rule"), "audit": tr.get("audit"),
                             "observation": tr.get("observation"),
                             "guard_trips": tr.get("guard_trips")})
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
                             "decision": tr.get("decision"), "state": tr.get("state"),
                             "salary": tr.get("salary"),
                             "rule": tr.get("rule"), "audit": tr.get("audit"),
                             "observation": tr.get("observation"),
                             "guard_trips": tr.get("guard_trips")})
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


def _policy_migration_selfcheck() -> List[tuple]:
    """[(имя, ok, деталь)] — ленивая миграция документа под новый движок (решение Р7).

    Самая опасная часть жёсткой подмены: если вилка не доедет до типизированного поля, зарплатный
    отсев отключится ТИХО — `compare_with_band` без границ всегда возвращает «проходит», и в
    отчётности это никак не видно. Поэтому проверки живут в гейте, а не в скретчпаде.
    """
    from qa_harness.domain.screening_split import context as ctx
    from qa_harness.domain.screening_split import state as state_model
    from qa_harness.domain.screening_split.policy import DecideContext, decide, migration
    from qa_harness.domain.screening_split.policy.observation import Observation

    out: List[tuple] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    VAC = {"title": "Python Backend Developer", "company_name": "ExampleSoft",
           "responsibilities": "Микросервисы", "work_format": "remote", "location": "Москва",
           "company_info": {"firm_description": "b2b", "vacancy_url": "https://example.com/v"},
           "min_salary": 200000, "max_salary": 280000, "questions": "- A\n- B"}


    def old_doc(**over):
        """Документ, каким его создавал СТАРЫЙ движок: контекст строкой с вилкой, band пустой."""
        doc = {
            "conversation_id": "c1", "engine": "split", "finished": False,
            "context": ctx.build_context("Анна", "Иван", "резюме HH", VAC),
            "location": "Москва", "contact_source": "резюме HH",
            "salary_band": {},
            "state": state_model.init_state("remote", "A\nB"),
        }
        doc.update(over)
        return doc


    # --- 1. вилка разбирается ДО вырезания строки ---
    d = old_doc()
    assert "Зарплатная вилка" in d["context"]
    rep = migration.upgrade(d)
    check("вилка разобрана из контекста", d["salary_band"] == {"min": 200000, "max": 280000, "currency": "RUB"},
          str(d["salary_band"]))
    check("источник вилки помечен", rep["band_source"] == "context", rep["band_source"])
    check("строка вилки вырезана", "Зарплатная вилка" not in d["context"])
    check("остальной контекст цел", "Должность: Python Backend Developer" in d["context"]
          and "Ссылка на вакансию: https://example.com/v" in d["context"])

    # --- 2. идемпотентность ---
    before = dict(d)
    rep2 = migration.upgrade(d)
    check("повторная миграция — no-op", rep2["migrated"] is False and d["context"] == before["context"])

    # --- 3. документ с уже типизированной вилкой не перетирается ---
    d3 = old_doc(salary_band={"min": 100000, "max": 150000})
    migration.upgrade(d3)
    check("готовая вилка не перезаписана", d3["salary_band"]["min"] == 100000, str(d3["salary_band"]))
    check("валюта проставлена по умолчанию", d3["salary_band"]["currency"] == "RUB")

    # --- 4. формы вилки: только «от», только «до» ---
    v_from = dict(VAC, min_salary=200000, max_salary=None)
    d4 = old_doc(context=ctx.build_context("А", "И", "", v_from))
    migration.upgrade(d4)
    check("форма «от X»", d4["salary_band"] == {"min": 200000, "max": None, "currency": "RUB"}, str(d4["salary_band"]))

    v_to = dict(VAC, min_salary=None, max_salary=280000)
    d5 = old_doc(context=ctx.build_context("А", "И", "", v_to))
    migration.upgrade(d5)
    check("форма «до Y»", d5["salary_band"] == {"min": None, "max": 280000, "currency": "RUB"}, str(d5["salary_band"]))

    # --- 5. вилки нет вовсе: провал разбора ВИДЕН, а не молчит ---
    v_none = dict(VAC, min_salary=None, max_salary=None)
    d6 = old_doc(context=ctx.build_context("А", "И", "", v_none))
    rep6 = migration.upgrade(d6)
    check("вилки нет → band_unparsed", rep6["band_unparsed"] is True and not d6["salary_band"], str(rep6))
    check("пустая строка вилки всё равно вырезана", "Зарплатная вилка" not in d6["context"])

    # --- 6. last_asking обнуляется, счётчики дополняются ---
    d7 = old_doc()
    d7["state"]["last_asking"] = "salary"
    d7["state"].pop("no_progress", None)
    d7["state"]["counters"].pop("contact_source", None)
    migration.upgrade(d7)
    check("last_asking обнулён", d7["state"]["last_asking"] is None)
    check("no_progress добавлен", d7["state"]["no_progress"] == 0)
    check("недостающий счётчик добавлен", d7["state"]["counters"]["contact_source"] == 0)
    check("схема помечена", d7["schema"] == migration.SCHEMA_VERSION)

    # --- 7. накопленное состояние НЕ теряется ---
    d8 = old_doc()
    d8["state"]["salary"] = "closed"
    d8["state"]["candidate_city"] = "Казань"
    d8["state"]["questions"][0]["status"] = "closed"
    d8["state"]["counters"]["pause"] = 2
    migration.upgrade(d8)
    check("собранное сохранено", d8["state"]["salary"] == "closed"
          and d8["state"]["candidate_city"] == "Казань"
          and d8["state"]["questions"][0]["status"] == "closed"
          and d8["state"]["counters"]["pause"] == 2)

    # --- 8. мигрированный документ реально даёт отсев по деньгам ---
    from qa_harness.domain.screening_split.policy import DecideContext, decide
    from qa_harness.domain.screening_split.policy.observation import Observation

    d9 = old_doc()
    migration.upgrade(d9)
    band = d9["salary_band"]
    obs = Observation(salary_claim={"subject": "own_expectation", "form": "exact",
                                    "amount_min": 400, "amount_max": 400, "scale": "thousand",
                                    "currency": "RUB", "period": "month", "tax": "net",
                                    "quote": "400 тысяч"})
    plan = decide(d9["state"], obs, "Ожидаю 400 тысяч на руки",
                  DecideContext(band_min=band["min"], band_max=band["max"], location="Москва"))
    check("после миграции отсев по деньгам работает", plan.reason_code == "KO_SALARY" and plan.end,
          f"{plan.reason_code} end={plan.end}")

    # То же БЕЗ миграции — демонстрация того, что чинится: вилки нет → отсев молча отключён.
    d10 = old_doc()
    plan10 = decide(d10["state"], obs, "Ожидаю 400 тысяч на руки",
                    DecideContext(band_min=None, band_max=None, location="Москва"))
    check("без миграции отсева НЕ было бы (то, что чиним)", plan10.reason_code != "KO_SALARY",
          plan10.reason_code)
    return out


def _policy_format_selfcheck() -> List[tuple]:
    """[(имя, ok, деталь)] — повестка «формат / город / переезд» в TG-ядре (решение Р18).

    Проверки переписаны после живого прогона `20260831_215941`: до Р18 город спрашивался ВНУТРИ
    вопроса про формат, а согласие переехать закрывало `format_check`. Отсюда три дефекта, и каждый
    имеет здесь свою строку: кандидат «в офис не готов, но перееду» проходил проверку формата; ключ
    отсева между форматом и локацией достался тому факту, что пришёл первым; на удалённых вакансиях
    город не спрашивался вовсе, из-за чего гео-ограничение вакансии не отсеивало никого.

    Живой сценарий этого не поймал бы: отсев ассертится по `expect_script_prefix: KO_`, а
    `KO_FORMAT_OFFICE` от `KO_LOCATION` этим префиксом не отличается.
    """
    from qa_harness.domain.screening_split import state as state_model
    from qa_harness.domain.screening_split.policy import DecideContext, decide
    from qa_harness.domain.screening_split.policy.geo import same_city
    from qa_harness.domain.screening_split.policy.observation import Observation

    out: List[tuple] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    def obs(**facts) -> Observation:
        o = Observation()
        o.facts = {"candidate_city": None, "format_ready": None, "relocation_ready": None,
                   "geo_blocked": False, **facts}
        return o

    def ready(work_format: str = "office", **over) -> dict:
        st = state_model.init_state(work_format, "- Опыт с Python?")
        st["salary"] = "closed"          # проверяем повестку локации, а не деньги
        st.update(over)
        return st

    MSK = DecideContext(band_max=280000, work_format="office", location="Москва")
    NOCITY = DecideContext(band_max=280000, work_format="office")
    HYBRID = DecideContext(band_max=280000, work_format="hybrid", location="Москва")
    REMOTE = DecideContext(band_max=280000, work_format="remote", location="Москва")

    # --- 1. порядок повестки: зарплата → город → формат → переезд ---
    plan = decide(ready(), obs(), "ок", MSK)
    check("первым спрашиваем ГОРОД, отдельным вопросом",
          plan.focus == "city" and "в каком городе" in plan.instruction
          and "формат" not in plan.instruction.lower(), plan.instruction[:90])
    plan = decide(ready(), obs(candidate_city="Москва"), "я в Москве", MSK)
    check("город назван → пункт закрыт, следующий вопрос про формат",
          plan.state_next.get("city_check") == "closed" and plan.focus == "format",
          f"{plan.state_next.get('city_check')}/{plan.focus}")
    check("вопрос про формат больше НЕ спрашивает город и переезд",
          "в каком городе" not in plan.instruction and "переехать" not in plan.instruction,
          plan.instruction[:110])

    # --- 2. эскалация вопроса про город: объяснение → предупреждение → кап ---
    # Счётчик в state — это значение ДО хода: 0 → на этом ходе будет 1-й переспрос, и так далее.
    plan = decide(ready(last_asking="city", city_reasks=0), obs(), "а зачем вам мой город?", MSK)
    check("1-й переспрос города объясняет ЗАЧЕМ и ещё не угрожает",
          "часовой пояс" in plan.instruction and "не получится" not in plan.instruction,
          plan.instruction[:120])
    plan = decide(ready(last_asking="city", city_reasks=1), obs(), "не скажу", MSK)
    check("2-й переспрос города предупреждает о последствии",
          "не получится" in plan.instruction, plan.instruction[:120])
    plan = decide(ready(last_asking="city", city_reasks=2), obs(), "не скажу", MSK)
    check("3-й переспрос города → завершение по капу",
          plan.reason_code == "STOP_PERSISTENT" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- 3. ГЛАВНОЕ Р18: формат — самостоятельное требование ---
    plan = decide(ready(candidate_city="Москва", city_check="closed"),
                  obs(format_ready="no", relocation_ready="yes"),
                  "в офис не готов, но перееду", MSK)
    check("«в офис не готов, но перееду» → всё равно KO по формату",
          plan.reason_code == "KO_FORMAT_OFFICE" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(ready(city_check="closed", candidate_city="Казань"), obs(format_ready="no"),
                  "в офис не готов", MSK)
    check("отказ от формата у иногороднего → KO по формату, а не переспрос",
          plan.reason_code == "KO_FORMAT_OFFICE", f"{plan.rule}/{plan.reason_code}")
    plan = decide(ready(), obs(format_ready="no"), "в офис не готов", MSK)
    check("отказ от формата при НЕизвестном городе → KO сразу",
          plan.reason_code == "KO_FORMAT_OFFICE", f"{plan.rule}/{plan.reason_code}")
    plan = decide(ready("hybrid", city_check="closed", candidate_city="Москва"),
                  obs(format_ready="no"), "гибрид не подходит", HYBRID)
    check("гибридная вакансия → KO_FORMAT_HYBRID", plan.reason_code == "KO_FORMAT_HYBRID",
          plan.reason_code)
    plan = decide(ready(city_check="closed", candidate_city="Казань"), obs(format_ready="no"),
                  "не готов", NOCITY)
    check("локации у вакансии нет → KO_FORMAT_NOCITY",
          plan.reason_code == "KO_FORMAT_NOCITY", plan.reason_code)

    # --- 4. пункт про переезд: когда открывается и когда нет ---
    plan = decide(ready(city_check="closed", candidate_city="Казань"), obs(format_ready="yes"),
                  "формат подходит", MSK)
    check("формат подтверждён + другой город → открывается пункт про переезд",
          plan.state_next.get("relocation_check") == "pending" and plan.focus == "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    check("вопрос про переезд — про МЕСТО, а не про формат",
          "переехать в этот город" in plan.instruction, plan.instruction[-110:])
    plan = decide(ready(city_check="closed", candidate_city="Москва"), obs(format_ready="yes"),
                  "формат подходит", MSK)
    check("кандидат В городе вакансии → про переезд не спрашиваем",
          plan.state_next.get("relocation_check") == "n/a" and plan.focus != "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    plan = decide(ready(city_check="closed", candidate_city="Казань"), obs(format_ready="yes"),
                  "формат подходит", NOCITY)
    check("у вакансии нет локации → про переезд не спрашиваем",
          plan.state_next.get("relocation_check") == "n/a", str(plan.state_next.get("relocation_check")))
    plan = decide(ready(candidate_city="Казань"), obs(relocation_ready="no"),
                  "переезжать не буду", MSK)
    check("отказ от переезда ДО подтверждения формата отсевом не является",
          not plan.end and plan.reason_code != "KO_LOCATION", f"{plan.rule}/{plan.reason_code}")

    # --- 5. отсев по локации и его отсутствие ---
    st = ready(city_check="closed", candidate_city="Казань", format_check="closed",
               relocation_check="pending")
    plan = decide(st, obs(relocation_ready="no"), "переезжать не буду", MSK)
    check("формат подтверждён + отказ от переезда → KO_LOCATION",
          plan.reason_code == "KO_LOCATION" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(st, obs(relocation_ready="yes"), "перееду", MSK)
    check("согласие переехать → пункт закрыт, отсева нет",
          plan.state_next.get("relocation_check") == "closed" and not plan.end,
          f"{plan.state_next.get('relocation_check')}/{plan.reason_code}")
    plan = decide(st, obs(), "подумаю", MSK)
    plan = decide(plan.state_next, obs(relocation_ready="yes"), "хорошо, перееду", MSK)
    check("кандидат передумал → пункт закрывается, диалог живёт",
          plan.state_next.get("relocation_check") == "closed" and not plan.end,
          f"{plan.state_next.get('relocation_check')}/{plan.reason_code}")

    # --- 6. удалённая вакансия: формат не спрашиваем, город спрашиваем (иначе гео-отсев мёртв) ---
    st = state_model.init_state("remote", "- Опыт с Python?")
    st["salary"] = "closed"
    check("удалёнка: формат n/a", st.get("format_check") == "n/a", str(st.get("format_check")))
    plan = decide(st, obs(), "здравствуйте", REMOTE)
    check("удалёнка: город всё равно спрашиваем",
          plan.focus == "city" and "в каком городе" in plan.instruction, plan.instruction[:90])
    plan = decide(st, obs(candidate_city="Тбилиси"), "я в Тбилиси", REMOTE)
    check("удалёнка: город назван → про переезд не спрашиваем никогда",
          plan.state_next.get("relocation_check") == "n/a" and plan.focus != "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    plan = decide(st, obs(candidate_city="Тбилиси", geo_blocked=True), "я в Тбилиси",
                  DecideContext(band_max=280000, work_format="remote", location="Москва",
                                has_geo_restriction=True))
    check("удалёнка + гео-ограничение + кандидат вне зоны → KO_GEO (вход у правила появился)",
          plan.reason_code == "KO_GEO" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- 7. сравнение города: локацию пишут шире города ---
    check("same_city: «Москва» ⊂ «Россия, Москва»", same_city("Москва", "Россия, Москва"))
    check("same_city: регистр и «ё»", same_city("КОРОЛЁВ", "Королев"))
    check("same_city: разные города не совпадают", not same_city("Казань", "Москва"))

    return out


def _policy_hh_selfcheck() -> List[tuple]:
    """[(имя, ok, деталь)] — hh-ядро `policy`: мультиформат, ключи отсева, бюджеты канала.

    Это ЕДИНСТВЕННЫЙ гейт на канальную дельту нового ядра: живой сценарий её не добывает (нужно
    заставить кандидата отказаться ровно от всех допустимых форматов и ровно в нужном порядке), а
    pytest в этом репозитории не держим. Проверяется то, чего нет в TG: «подходит хотя бы один
    формат», выбор между `KO_FORMAT` и `KO_LOCATION`, отдельная ветка разъездного формата и то, что
    переход с одного формата на другой не считается переспросом.
    """
    from qa_harness.domain.screening_split_hh import state as hh_state
    from qa_harness.domain.screening_split_hh.policy import DecideContext, Observation, decide
    from qa_harness.domain.screening_split_hh.policy import parse_observation, reasons as hh_reasons
    from qa_harness.domain.screening_split_hh.policy import state_for_prompt as hh_projection

    out: List[tuple] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    def obs(*, formats=(), relocation=None, geo=False, city=None, signals=()) -> Observation:
        o = Observation()
        o.facts = {"candidate_city": city,
                   "formats_ready": [{"format": f, "ready": r} for f, r in formats],
                   "relocation_ready": relocation, "geo_blocked": geo}
        o.signals = list(signals)
        return o

    def ready(allowed, **over) -> dict:
        """Состояние с закрытой зарплатой И закрытым городом: секции ниже проверяют формат.

        Город с решения Р18 — отдельный пункт повестки и стоит ПЕРЕД форматом, поэтому без этого
        дефолта каждая проверка мультиформата упиралась бы в вопрос про город. Проверки самого города
        передают `city_check="pending"` явно.
        """
        st = hh_state.init_state(allowed, "- Опыт с Python?")
        st["salary"] = "closed"
        st["city_check"] = "closed"
        st.update(over)
        return st

    CTX = DecideContext(band_min=200000, band_max=280000, location="Москва")

    # --- 1. инициализация проверок по таблице мультиформата ---
    st_both = hh_state.init_state(["ON_SITE", "HYBRID"], "- A")
    st_remote = hh_state.init_state(["REMOTE", "ON_SITE"], "- A")
    st_field = hh_state.init_state(["FIELD_WORK"], "- A")
    check("init: [ON_SITE,HYBRID] → format pending / field n/a",
          st_both["format_check"] == "pending" and st_both["field_work_check"] == "n/a")
    check("init: REMOTE среди допустимых снимает проверку формата",
          st_remote["format_check"] == "n/a", st_remote["format_check"])
    check("init: [FIELD_WORK] → format n/a / field pending",
          st_field["format_check"] == "n/a" and st_field["field_work_check"] == "pending")

    # --- 2. отказ от ОДНОГО формата: не отсев и не закрытие, вопрос про следующий ---
    plan = decide(ready(["ON_SITE", "HYBRID"]), obs(formats=[("ON_SITE", "no")]), "в офис не готов", CTX)
    check("отказ от офиса при [ON_SITE,HYBRID] — не KO",
          plan.kind == "ask" and plan.focus == "format", f"{plan.rule}/{plan.reason_code}")
    check("следующим спрашиваем про гибрид", plan.state_next.get("format_asked") == "HYBRID",
          str(plan.state_next.get("format_asked")))
    check("гибрид назван в вопросе словами, без сырого id",
          "гибридный" in plan.instruction and "HYBRID" not in plan.instruction, plan.instruction[:90])

    # --- 3. отказ от ВСЕХ допустимых → KO_FORMAT ---
    plan = decide(ready(["ON_SITE", "HYBRID"], formats={"ON_SITE": "no"}),
                  obs(formats=[("HYBRID", "no")]), "гибрид тоже не подходит", CTX)
    check("отказ от всех допустимых → KO_FORMAT", plan.reason_code == "KO_FORMAT" and plan.end,
          f"{plan.rule}/{plan.reason_code}")

    # --- 4. согласие на один формат закрывает проверку ---
    plan = decide(ready(["ON_SITE", "HYBRID"]), obs(formats=[("HYBRID", "yes")]), "гибрид подойдёт", CTX)
    check("согласие на гибрид закрывает format_check",
          plan.state_next["format_check"] == "closed", plan.state_next["format_check"])

    # --- 5. разъездной формат: отказ при другом подтверждённом — не KO ---
    plan = decide(ready(["ON_SITE", "FIELD_WORK"], formats={"ON_SITE": "yes"}, format_check="closed"),
                  obs(formats=[("FIELD_WORK", "no")]), "разъезды не готов", CTX)
    check("отказ от разъездов при подтверждённом офисе — не KO",
          plan.reason_code != "KO_FORMAT" and plan.state_next["field_work_check"] == "closed",
          f"{plan.reason_code}/{plan.state_next['field_work_check']}")

    # --- 6. разъездной единственный → отказ = KO_FORMAT ---
    plan = decide(ready(["FIELD_WORK"]), obs(formats=[("FIELD_WORK", "no")]), "разъезды не готов", CTX)
    check("отказ от единственного FIELD_WORK → KO_FORMAT", plan.reason_code == "KO_FORMAT",
          plan.reason_code)

    # --- 7. локация и формат — РАЗНЫЕ пункты повестки (решение Р18) ---
    # Прежние проверки этой секции утверждали обратное («согласие на переезд закрывает format_check»,
    # «спрашиваем формат И переезд одним вопросом») и потому маскировали дефект кейса E живого
    # прогона 20260831_215941: отказ и от формата, и от переезда одной репликой уходил в KO_FORMAT.
    plan = decide(ready(["ON_SITE"], city_check="pending"), obs(), "а что по вакансии?", CTX)
    check("первым спрашиваем ГОРОД, отдельным вопросом",
          plan.focus == "city" and "в каком городе" in plan.instruction
          and "формат" not in plan.instruction.lower(), plan.instruction[:110])
    plan = decide(ready(["ON_SITE"], candidate_city="Казань", city_check="closed"),
                  obs(), "а что по вакансии?", CTX)
    check("вопрос про формат больше НЕ спрашивает город и переезд",
          "переехать" not in plan.instruction and "в каком городе" not in plan.instruction,
          plan.instruction[:130])
    plan = decide(ready(["ON_SITE"], candidate_city="Москва", city_check="closed"),
                  obs(formats=[("ON_SITE", "no")], relocation="yes"),
                  "в офис не готов, но перееду", CTX)
    check("«в офис не готов, но перееду» → всё равно KO_FORMAT",
          plan.reason_code == "KO_FORMAT" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(ready(["ON_SITE"], candidate_city="Казань", city_check="closed"),
                  obs(formats=[("ON_SITE", "yes")]), "формат подходит", CTX)
    check("формат подтверждён + другой город → открывается пункт про переезд",
          plan.state_next.get("relocation_check") == "pending" and plan.focus == "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    check("вопрос про переезд — про МЕСТО, а не про формат",
          "переехать в этот город" in plan.instruction, plan.instruction[-120:])
    st_reloc = ready(["ON_SITE"], candidate_city="Казань", city_check="closed",
                     format_check="closed", relocation_check="pending")
    plan = decide(st_reloc, obs(relocation="no"), "переезжать не буду", CTX)
    check("формат подтверждён + отказ от переезда → KO_LOCATION",
          plan.reason_code == "KO_LOCATION" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(st_reloc, obs(relocation="yes"), "готов переехать", CTX)
    check("согласие переехать закрывает ПУНКТ ПРО ПЕРЕЕЗД, а не формат",
          plan.state_next["relocation_check"] == "closed" and not plan.end,
          f"{plan.state_next['relocation_check']}/{plan.reason_code}")
    plan = decide(ready(["ON_SITE"], candidate_city="Москва", city_check="closed"),
                  obs(formats=[("ON_SITE", "yes")], relocation="no"),
                  "формат ок, переезжать никуда не буду", CTX)
    check("тот же город: отказ от переезда отсевом не является",
          plan.reason_code != "KO_LOCATION" and plan.state_next["relocation_check"] == "n/a",
          f"{plan.reason_code}/{plan.state_next['relocation_check']}")
    # Разъездной формат считается присутственным: локация для него важна так же, как для офиса.
    plan = decide(ready(["FIELD_WORK"], candidate_city="Казань", city_check="closed"),
                  obs(formats=[("FIELD_WORK", "yes")]), "к разъездам готов", CTX)
    check("разъездной подтверждён + другой город → пункт про переезд открывается",
          plan.state_next.get("relocation_check") == "pending",
          str(plan.state_next.get("relocation_check")))
    # Регрессия прогона 20260901_181013, сценарий 56: `[REMOTE, FIELD_WORK]`, кандидат отказался от
    # разъездов (по правилу канала это НЕ отсев, `field_work_check` закрывается) и переезжать не
    # готов. Первая версия Р18 читала это закрытие как подтверждение присутствия и отсевала по
    # локации — на вакансии, где есть удалёнка и ехать никуда не надо.
    st_rf = ready(["REMOTE", "FIELD_WORK"], candidate_city="Новосибирск", city_check="closed")
    plan = decide(st_rf, obs(formats=[("FIELD_WORK", "no")]), "к разъездам не готов", CTX)
    check("есть REMOTE: отказ от разъездов не открывает пункт переезда",
          plan.state_next.get("relocation_check") == "n/a" and plan.focus != "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    plan = decide(plan.state_next, obs(relocation="no"), "переезжать не буду", CTX)
    check("есть REMOTE: отказ переезжать не отсевает",
          plan.reason_code != "KO_LOCATION" and not plan.end, f"{plan.rule}/{plan.reason_code}")
    # Второй промах того же условия: «проверка закрыта» ≠ «формат подтверждён».
    st_of = ready(["ON_SITE", "FIELD_WORK"], candidate_city="Казань", city_check="closed")
    plan = decide(st_of, obs(formats=[("FIELD_WORK", "no")]), "разъезды не готов", CTX)
    check("закрытие разъездного ОТКАЗОМ присутствия не подтверждает",
          plan.state_next.get("relocation_check") == "n/a",
          str(plan.state_next.get("relocation_check")))

    # Удалённая вакансия: формат не спрашиваем, а город — обязательно, иначе гео-отсев мёртв.
    plan = decide(ready(["REMOTE"], city_check="pending"), obs(), "здравствуйте", CTX)
    check("удалёнка: город всё равно спрашиваем",
          plan.focus == "city" and "в каком городе" in plan.instruction, plan.instruction[:100])
    plan = decide(ready(["REMOTE"], candidate_city="Тбилиси", city_check="closed"),
                  obs(relocation="no"), "переезжать не буду", CTX)
    check("удалёнка: про переезд не спрашиваем и по нему не отсеваем",
          plan.state_next.get("relocation_check") == "n/a" and plan.reason_code != "KO_LOCATION",
          f"{plan.state_next.get('relocation_check')}/{plan.reason_code}")

    # --- 8. гео-ограничение: только при двойном совпадении (Б3) ---
    geo_ctx = DecideContext(band_max=280000, location="Россия, только РФ", has_geo_restriction=True)
    plan = decide(ready(["REMOTE"]), obs(geo=True, city="Берлин"), "я живу в Германии", geo_ctx)
    check("гео-ограничение + нарушение → KO_LOCATION_GEO", plan.reason_code == "KO_LOCATION_GEO",
          plan.reason_code)
    plan = decide(ready(["REMOTE"]), obs(geo=True, city="Берлин"), "я живу в Германии", CTX)
    check("без ограничения в вакансии заграница отсевом не является",
          plan.reason_code != "KO_LOCATION_GEO", plan.reason_code)

    # --- 9. смена формата в вопросе — не переспрос, повтор того же формата — переспрос ---
    base = ready(["ON_SITE", "HYBRID"], last_asking="format", format_asked="ON_SITE")
    plan = decide(base, obs(formats=[("ON_SITE", "no")]), "в офис не готов", CTX)
    check("переход офис→гибрид кап переспросов не жжёт", plan.state_next["format_reasks"] == 0,
          str(plan.state_next["format_reasks"]))
    plan = decide(base, obs(), "ага", CTX)
    check("повтор того же формата — переспрос", plan.state_next["format_reasks"] == 1,
          str(plan.state_next["format_reasks"]))

    # --- 10. кап переспросов разъездного формата (ветки нет в TG) ---
    plan = decide(ready(["FIELD_WORK"], format_check="n/a", last_asking="field_work",
                        format_asked="FIELD_WORK", field_work_reasks=2),
                  obs(), "не понял вопроса", CTX)
    check("3-й переспрос про разъезды → STOP_PERSISTENT",
          plan.reason_code == "STOP_PERSISTENT" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- 11. реестр причин: канальные ключи есть, TG-шных нет ---
    check("реестр: KO_LOCATION/KO_LOCATION_GEO/KO_FORMAT есть",
          all(hh_reasons.is_known(k) for k in ("KO_LOCATION", "KO_LOCATION_GEO", "KO_FORMAT")))
    check("реестр: TG-ключей (KO_GEO, KO_FORMAT_OFFICE, REPLY_CONTACT_SOURCE) нет",
          not any(hh_reasons.is_known(k) for k in ("KO_GEO", "KO_FORMAT_OFFICE", "REPLY_CONTACT_SOURCE")))
    check("реестр: STOP_POLITICS не рендерится в пустоту",
          bool((hh_reasons.render("STOP_POLITICS") or "").strip()))

    # --- 12. сигнала contact_source в канале нет — парсер его отбрасывает ---
    parsed, _ = parse_observation(
        {"signals": [{"code": "contact_source", "quote": "откуда мои данные"}],
         "focus_answered": "none"}, "откуда мои данные")
    check("сигнал contact_source отброшен", parsed.codes() == [], str(parsed.codes()))

    # --- 13. вилка в тенге приводится к рублям (P11: сегодня currency не читается) ---
    kzt = DecideContext(band_min=1_000_000, band_max=1_400_000, band_currency="KZT", location="Москва")
    claim = {"subject": "own_expectation", "form": "exact", "amount_min": 250, "amount_max": 250,
             "scale": "thousand", "currency": "RUB", "period": "month", "tax": "net",
             "quote": "250 тысяч на руки"}
    o = obs()
    o.salary_claim = claim
    plan = decide(ready(["REMOTE"]), o, "250 тысяч на руки", kzt)
    check("вилка KZT пересчитана — 250к не отсев", plan.reason_code != "KO_SALARY",
          f"{plan.reason_code}/{(plan.audit.get('salary') or {}).get('verdict')}")

    # --- 14. проекция состояния: служебного модель не видит, форматы видит ---
    projection = hh_projection(ready(["ON_SITE"], format_asked="ON_SITE"))
    check("проекция без счётчиков и служебных полей",
          not ({"counters", "format_reasks", "no_progress", "formats"} & set(projection)),
          str(sorted(projection)))
    check("проекция отдаёт allowed_formats и format_asked",
          projection.get("allowed_formats") == ["ON_SITE"] and projection.get("format_asked") == "ON_SITE")

    return out


def _run_offline(args: argparse.Namespace, scenarios: List[Scenario], channel: str = "tg") -> int:
    """Плумбинг + детерминированные проверки кода. Возвращает число провалов (0 = всё зелено)."""
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
    print(f"[offline] санити чистого домена (канал {channel}):")
    for line in (_domain_sanity_hh() if channel == "hh" else _domain_sanity()):
        print(f"  · {line}")

    failed = 0
    if channel == "tg":
        cases = _salary_selfcheck()
        failed = sum(1 for _, ok, _ in cases if not ok)
        print(f"[offline] зарплатный контракт (salary_claim): {len(cases) - failed}/{len(cases)}")
        for name, ok, detail in cases:
            if not ok or not args.quiet:
                print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

        mig = _policy_migration_selfcheck()
        mig_failed = sum(1 for _, ok, _ in mig if not ok)
        failed += mig_failed
        print(f"[offline] ленивая миграция под движок policy: {len(mig) - mig_failed}/{len(mig)}")
        for name, ok, detail in mig:
            if not ok or not args.quiet:
                print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

        fmt = _policy_format_selfcheck()
        fmt_failed = sum(1 for _, ok, _ in fmt if not ok)
        failed += fmt_failed
        print(f"[offline] отсев по формату (вопрос про переезд, KO_FORMAT_*): {len(fmt) - fmt_failed}/{len(fmt)}")
        for name, ok, detail in fmt:
            if not ok or not args.quiet:
                print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    else:
        hh = _policy_hh_selfcheck()
        hh_failed = sum(1 for _, ok, _ in hh if not ok)
        failed += hh_failed
        print(f"[offline] hh-ядро policy (мультиформат, ключи отсева, бюджеты): {len(hh) - hh_failed}/{len(hh)}")
        for name, ok, detail in hh:
            if not ok or not args.quiet:
                print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    print("[offline] сеть и пакет prompts не дёргались.")
    return failed


# ── детерминированные проверки зарплатного контракта (без сети и без LLM) ──────
# Живут в offline-режиме раннера, а не в pytest: в этом репозитории юнит-тестов на код харнесса не
# держим, корректность кода проверяется `--offline`-прогоном. Здесь два класса, которых живой сценарий
# добыть НЕ может, потому что для них нужно заставить модель ошибиться:
#   · порядок «нормализация → гейт updates» (негодный claim + salary:closed в updates);
#   · все ветки перерешивания хода (нужна ошибка модели в служебном поле claim).
def _claim(**kw) -> Dict[str, Any]:
    base = {"subject": "own_expectation", "form": "exact", "amount_min": None, "amount_max": None,
            "scale": "thousand", "currency": "RUB", "period": "month", "tax": "unspecified",
            "quote": ""}
    base.update(kw)
    return base


def _decision(**kw) -> Dict[str, Any]:
    base = {"next_action": "ask", "script_key": None, "instruction": "Спроси зарплатные ожидания",
            "updates": [], "event": None, "asking": "salary", "salary_claim": None}
    base.update(kw)
    return base


class _FakeAnalyzer:
    """Отдаёт заранее заданные Decision по одному на вызов и считает вызовы."""

    def __init__(self, *decisions: Dict[str, Any]) -> None:
        self._queue = list(decisions)
        self.calls = 0
        self.notes: List[str | None] = []  # служебные строки, с которыми звали (для проверки rewind)
        self.last_usage = blank_usage()

    def run(self, context: str, state: Dict[str, Any], message: str, *,
            note: str | None = None) -> Dict[str, Any]:
        self.calls += 1
        self.notes.append(note)
        return self._queue.pop(0) if self._queue else _decision()


class _FakeInterviewer:
    def run(self, conversation_id: Any, instruction: str, message: str):
        return f"[текст по инструкции] {instruction}", blank_usage()


def _engine_turn(*decisions: Dict[str, Any], message: str,
                 state_over: Dict[str, Any] | None = None, band=(200000, 280000)):
    """Один ход движка с фейковыми ролями. Возвращает (result, doc, analyzer).

    Несколько решений передаются для ветки перерешивания: второе уходит на второй вызов Аналитика.
    """
    store = sp.InMemoryStateStore()
    state = sp.init_state("remote", "")
    if state_over:
        state.update(state_over)
    store.create("c1", "split", state=state,
                 context="Зарплатная вилка: от 200000 до 280000 рублей (НЕ РАСКРЫВАТЬ!)",
                 location="Москва", contact_source="hh",
                 salary_band={"min": band[0], "max": band[1]})
    analyzer = _FakeAnalyzer(*decisions)
    engine = sp.ScreeningSplitEngine(store, analyzer, _FakeInterviewer(), None)
    result = engine.add_message_and_run("c1", message)
    return result, store.load("c1"), analyzer


def _salary_selfcheck() -> List[tuple]:
    """[(имя, ok, деталь)] — арифметика, гейты годности, приоритеты решений, перерешивание хода."""
    out: List[tuple] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    # --- арифметика: Д8-Д11, значения считаются ОТ КОНСТАНТ модуля правил ---
    from qa_harness.domain.screening_split import salary_rules as rules

    v = sp.normalize(_claim(amount_min=70, scale="none_stated"))
    check("Д8 голое «70» = 70000", v.min == 70_000, f"факт {v.min}")
    v = sp.normalize(_claim(amount_min=200, scale="unit"))
    check("Д8 «200 рублей» читаем буквально", v.min == 200, f"факт {v.min}")
    v = sp.normalize(_claim(amount_min=1200, scale="unit", period="hour"))
    check("Д9 ставка за час × норму часов",
          v.min == round(1200 * rules.PERIOD_TO_MONTH["hour"]), f"факт {v.min}")
    v = sp.normalize(_claim(amount_min=4000, scale="unit", currency="USD"))
    check("Д10 валюта по курсу из правил",
          v.min == round(4000 * rules.RATE_TO_RUB["USD"]), f"факт {v.min}")
    v = sp.normalize(_claim(amount_min=250, scale="thousand", tax="gross"))
    check("Д11 gross → net по прогрессивной шкале",
          v.min < round(250_000 * (1 - rules.NDFL_BRACKETS[0][1])), f"факт {v.min}")

    # --- вердикт по вилке: Д12/Д13 ---
    def verdict(**kw) -> str:
        return sp.compare_with_band(sp.normalize(_claim(**kw)), 200_000, 280_000)

    check("Д13 диапазон 250-400 — не отказ (готов на 250)",
          verdict(form="range", amount_min=250, amount_max=400, scale="thousand") == "fits")
    check("Д13 порог «от 300» — отказ", verdict(form="at_least", amount_min=300, scale="thousand") == "ko")
    check("«до 400» — не отказ никогда", verdict(form="at_most", amount_max=400, scale="thousand") == "fits")
    check("ниже минимума вилки — не отказ", verdict(amount_min=100, scale="thousand") == "fits")
    check("Д12 отказ по ПЕРЕСЧИТАННОЙ сумме",
          verdict(amount_min=4500, scale="unit", currency="USD") == "ko")

    # --- гейты годности claim ---
    check("чужая сумма непригодна",
          sp.claim_status(_claim(subject="third_party", amount_min=500, quote="500 тысяч"),
                          "У коллеги 500 тысяч") == sp.UNUSABLE)
    check("текущая ЗП без ожиданий непригодна",
          sp.claim_status(_claim(subject="own_current", amount_min=250, quote="получаю 250"),
                          "Сейчас получаю 250") == sp.UNUSABLE)
    check("валюта вне справочника непригодна",
          sp.claim_status(_claim(currency="other", amount_min=4000, quote="4000 фунтов"),
                          "Рассматриваю 4000 фунтов") == sp.UNUSABLE)
    check("выдуманная цитата непригодна (гейт против галлюцинации)",
          sp.claim_status(_claim(amount_min=400, quote="400 тысяч"),
                          "Готов обсуждать варианты") == sp.UNUSABLE)
    check("годный claim пригоден",
          sp.claim_status(_claim(amount_min=300, quote="300 тысяч"),
                          "Ориентируюсь на 300 тысяч") == sp.ACTIONABLE)

    # --- ВЫРОЖДЕННАЯ ФОРМА claim: единственная причина, по которой normalize даёт None ---
    # Потолка правдоподобия у пересчёта НЕТ: абсурдно большая сумма — законный отказ по вилке, а не
    # ошибка. `claim_status` вырождение не ловит (видит одно число, не зная, что форма его обнулит).
    _degenerate = _claim(form="at_least", amount_min=None, amount_max=300, quote="300 тысяч")
    check("вырожденная форма: статус пропускает, пересчёт даёт None",
          sp.claim_status(_degenerate, "Ориентируюсь на 300 тысяч") == sp.ACTIONABLE
          and sp.normalize(_degenerate) is None)
    check("абсурдная сумма — обычный отказ по вилке, а не уточнение",
          sp.compare_with_band(sp.normalize(_claim(amount_min=20, scale="million")),
                               200_000, 280_000) == "ko")
    res, doc, _ = _engine_turn(
        _decision(salary_claim=_degenerate, updates=[{"key": "salary", "value": "closed"}],
                  asking="salary"),
        _decision(asking="salary"), message="Ориентируюсь на 300 тысяч")
    check("вырожденная форма в движке: не отсеивает и НЕ закрывает пункт",
          doc["state"]["salary"] == "pending" and not res.conversation_end,
          f"salary={doc['state']['salary']} end={res.conversation_end}")
    check("вырожденная форма помечена unusable в трассе",
          doc["salary_claims"][0]["status"] == sp.UNUSABLE,
          f"факт {doc['salary_claims'][0]['status']}")

    # --- ПОРЯДОК «нормализация → гейт updates» ---
    res, doc, _ = _engine_turn(
        _decision(updates=[{"key": "salary", "value": "closed"}]),
        _decision(asking="salary"), message="Давайте обсудим позже")
    check("гейт: closed без claim отброшен", doc["state"]["salary"] == "pending",
          f"факт {doc['state']['salary']}")
    res, doc, _ = _engine_turn(
        _decision(salary_claim=_claim(form="not_stated", quote="обсуждаемо"),
                  updates=[{"key": "salary", "value": "closed"}]),
        _decision(asking="salary"), message="Обсуждаемо, зависит от задач")
    check("гейт: closed с непригодным claim отброшен", doc["state"]["salary"] == "pending",
          f"факт {doc['state']['salary']}")
    res, doc, _ = _engine_turn(
        _decision(updates=[{"key": "salary", "value": "closed"},
                           {"key": "candidate_city", "value": "Казань"}]),
        _decision(asking="salary"), message="Я из Казани")
    check("гейт снимает только зарплату, остальные факты проходят",
          doc["state"]["salary"] == "pending" and doc["state"]["candidate_city"] == "Казань")

    # --- приоритеты решений ---
    res, doc, _ = _engine_turn(
        _decision(salary_claim=_claim(amount_min=400, quote="400 тысяч"), event="demand"),
        message="Повторяю, 400 тысяч и не меньше",
        state_over={"counters": {**sp.init_state("remote", "")["counters"], "demand": 2}})
    check("ko перебивает событийный порог (причина отказа — деньги)",
          doc["salary_claims"][0]["effect"] == "ko_forced" and res.conversation_end,
          f"effect={doc['salary_claims'][0]['effect']}")
    res, doc, _ = _engine_turn(
        _decision(next_action="script", script_key="FINISH", instruction=None, asking=None,
                  salary_claim=_claim(amount_min=400, quote="400 тысяч")),
        message="И ещё: хочу 400 тысяч")
    check("ko перебивает FINISH", doc["salary_claims"][0]["effect"] == "ko_forced",
          f"effect={doc['salary_claims'][0]['effect']}")
    res, doc, _ = _engine_turn(
        _decision(next_action="script", script_key="FINISH", instruction=None, asking=None,
                  salary_claim=_claim(amount_min=250, quote="250 тысяч")),
        message="И ещё: хочу 250 тысяч")
    check("fits НЕ снимает FINISH", res.conversation_end
          and doc["salary_claims"][0]["effect"] == "closed",
          f"effect={doc['salary_claims'][0]['effect']} end={res.conversation_end}")
    res, doc, _ = _engine_turn(
        _decision(next_action="script", script_key="STOP_ABUSE", instruction=None, asking=None,
                  salary_claim=_claim(amount_min=400, quote="400 тысяч")),
        message="Хочу 400 тысяч, уроды")
    check("НЕденежное терминальное решение Аналитика сильнее отсева по деньгам",
          doc["salary_claims"][0]["effect"] == "ko_overridden_by_analyzer" and res.conversation_end,
          f"effect={doc['salary_claims'][0]['effect']}")
    res, doc, analyzer = _engine_turn(
        _decision(next_action="script", script_key="STOP_NOT_INTERESTED", instruction=None, asking=None,
                  salary_claim=_claim(form="at_least", amount_min=250, quote="ниже 250 тысяч")),
        message="Ниже 250 тысяч не рассматриваю ваше предложение")
    check("fits снимает денежное завершение, диалог продолжается, второго вызова LLM нет",
          not res.conversation_end and doc["state"]["salary"] == "closed" and analyzer.calls == 1,
          f"end={res.conversation_end} salary={doc['state']['salary']} calls={analyzer.calls}")
    res, doc, _ = _engine_turn(
        _decision(salary_claim=_claim(amount_min=400, quote="400 тысяч")),
        message="Передумал, хочу 400 тысяч", state_over={"salary": "closed"})
    check("повторное сравнение после закрытия пункта",
          res.conversation_end and doc["salary_claims"][0]["effect"] == "ko_forced",
          f"effect={doc['salary_claims'][0]['effect']}")

    # --- ПЕРЕРЕШИВАНИЕ ХОДА при расхождении кода и Аналитика ---
    # Триггер: claim непригоден И решение построено на посылке «деньги закрыты». Модель не может знать
    # заранее, что claim забракуют, поэтому сама этот случай не вытянет — только детерминированно.
    broken = _claim(amount_min=300, quote="300 тысяч на руки")  # цитаты нет в реплике → unusable
    reask = _decision(instruction="Переспроси сумму на руки в месяц", asking="salary")
    msg = "Готов обсуждать варианты"

    res, doc, an = _engine_turn(_decision(salary_claim=broken, asking="q1",
                                          instruction="Спроси про PostgreSQL",
                                          updates=[{"key": "salary", "value": "closed"}]),
                                reask, message=msg)
    check("перерешивание: закрытие зарплаты в updates → второй вызов",
          an.calls == 2 and "Переспроси сумму" in (res.response or ""), f"calls={an.calls}")
    # Без служебной строки второй вызов получил бы тождественный вход (state не изменился —
    # отклонённое `salary: closed` в него не попало) и при temperature=0 повторил бы то же решение.
    check("перерешивание: второй вызов получает служебную строку, первый — нет",
          an.notes[:1] == [None] and bool(an.notes[1]), f"notes={an.notes}")
    _note = an.notes[1] or ""
    # Отброшенную инструкцию в строку НЕ кладём: пробовали, модель протаскивала из неё переход к
    # следующему приоритету. Реакцию на счётный триггер второй вызов выводит сам — он видит счётчик
    # нетронутым (инкремент перенесён за все вызовы Аналитика).
    check("служебная строка НЕ несёт отброшенную инструкцию",
          "Спроси про PostgreSQL" not in _note and "Ранее ты собирался" not in _note,
          f"note={_note[:120]}")
    check("служебная строка не перечисляет категории содержимого",
          "остальное содержание реплики обработай как обычно" in _note
          and "отработай сработавшие триггеры" in _note)
    # Валютно-нейтральна: перерешивание срабатывает на нашей ошибке, где валюта ни при чём, а
    # требование рублей возвращало бы трение, ради которого курс и завели (Д10).
    check("служебная строка валютно-нейтральна и без «на руки»",
          "в формате оплаты за месяц числом" in _note and "чтобы передать коллегам" in _note
          and "в рублях" not in _note and "на руки" not in _note)
    check("перерешивание: пункт остался pending", doc["state"]["salary"] == "pending",
          f"факт {doc['state']['salary']}")
    check("перерешивание видно в трассе", doc["salary_claims"][0]["rewind"] is True)

    res, doc, an = _engine_turn(_decision(salary_claim=broken, asking="format",
                                          instruction="Спроси город"), reask, message=msg)
    check("перерешивание: asking уехал с зарплаты → второй вызов", an.calls == 2, f"calls={an.calls}")

    res, doc, an = _engine_turn(
        _decision(next_action="script", script_key="FINISH", instruction=None, asking=None,
                  salary_claim=broken), reask, message=msg)
    check("перерешивание: FINISH не закрывает скрининг с неразрешённой зарплатой",
          an.calls == 2 and not res.conversation_end and doc["state"]["salary"] == "pending",
          f"calls={an.calls} end={res.conversation_end}")
    check("служебная строка валидна и без инструкции (у FINISH её нет)",
          "Ранее ты собирался" not in (an.notes[1] or "")
          and "зарплаты НЕ закрыт" in (an.notes[1] or ""))

    res, doc, an = _engine_turn(
        _decision(salary_claim=_claim(subject="third_party", amount_min=500, quote="500 тысяч"),
                  instruction="Переспроси сумму", asking="salary"),
        message="У коллеги 500 тысяч")
    check("НЕТ перерешивания, когда Аналитик сам переспрашивает (вина кандидата)",
          an.calls == 1, f"calls={an.calls}")

    res, doc, an = _engine_turn(
        _decision(salary_claim=_claim(subject="own_current", amount_min=250, quote="получаю 250"),
                  instruction="Ответь про формат", asking=None),
        message="Сейчас получаю 250, а какой формат работы?")
    check("НЕТ перерешивания при asking=null (ответ на вопрос кандидата — законный ход)",
          an.calls == 1, f"calls={an.calls}")

    res, doc, an = _engine_turn(_decision(asking="format", instruction="Спроси город"),
                               message="Здравствуйте!")
    check("НЕТ перерешивания, когда про деньги речи не было", an.calls == 1, f"calls={an.calls}")

    res, doc, an = _engine_turn(_decision(salary_claim=broken, asking="q1",
                                          instruction="Спроси про PostgreSQL"),
                                _decision(instruction="Переспроси сумму", asking="salary",
                                          updates=[{"key": "salary", "value": "closed"}]),
                                message=msg)
    check("updates ВТОРОГО решения тоже гейтятся", doc["state"]["salary"] == "pending",
          f"факт {doc['state']['salary']}")

    res, doc, an = _engine_turn(_decision(salary_claim=broken, asking="q1", event="gibberish",
                                          instruction="Спроси про PostgreSQL"),
                                _decision(instruction="Переспроси сумму", asking="salary",
                                          event="gibberish"), message=msg)
    check("event второго решения не считается дважды",
          doc["state"]["counters"]["gibberish"] == 1,
          f"факт {doc['state']['counters']['gibberish']}")

    _st = {"last_asking": "salary", "salary_reasks": 2}  # ещё один засчитанный → STOP_SALARY_DEMAND
    res, doc, an = _engine_turn(_decision(salary_claim=broken, asking="format",
                                          instruction="Спроси город"), reask,
                                message=msg, state_over=dict(_st))
    check("перерешённый ход НЕ жжёт reask-cap (наша ошибка, не уклонение кандидата)",
          not res.conversation_end and doc["state"]["salary_reasks"] == 2,
          f"end={res.conversation_end} reasks={doc['state']['salary_reasks']}")

    res, doc, an = _engine_turn(
        _decision(salary_claim=_claim(form="not_stated", quote="обсуждаемо"), asking="salary",
                  instruction="Переспроси сумму"),
        message="Обсуждаемо, зависит от задач", state_over=dict(_st))
    check("reask-cap по-прежнему завершает диалог, когда уклоняется кандидат",
          res.conversation_end, f"end={res.conversation_end}")

    # --- ОБРАТНОЕ расхождение: код закрыл пункт, а решение всё ещё спрашивает про деньги ---
    # Прогон 28.08 (сценарий 45): «правильно ли я понимаю, что 75 тысяч?» уже после того, как код
    # сумму принял. Служебная строка здесь не нужна — состояние изменилось само (pending → closed).
    good = _claim(amount_min=250, quote="250 тысяч")
    res, doc, an = _engine_turn(
        _decision(salary_claim=good, asking="salary", instruction="Переспроси сумму ещё раз"),
        _decision(asking="format", instruction="Спроси город"),
        message="Ориентируюсь на 250 тысяч")
    check("fits + asking=salary → второй вызов, переспрос кандидату не уходит",
          an.calls == 2 and "Спроси город" in (res.response or ""), f"calls={an.calls}")
    check("fits + asking=salary: пункт закрыт, эффект виден в трассе",
          doc["state"]["salary"] == "closed"
          and doc["salary_claims"][0]["effect"] == "closed_reask_dropped",
          f"salary={doc['state']['salary']} effect={doc['salary_claims'][0]['effect']}")
    check("этот второй вызов идёт БЕЗ служебной строки (состояние изменилось само)",
          an.notes == [None, None], f"notes={an.notes}")

    res, doc, an = _engine_turn(
        _decision(salary_claim=good, asking="format", instruction="Спроси город"),
        message="Ориентируюсь на 250 тысяч")
    check("НЕТ второго вызова, когда Аналитик и так ушёл дальше", an.calls == 1, f"calls={an.calls}")

    # Счётчик обязан быть нетронутым на ОБОИХ вызовах: иначе второй прочитает «вы бот?» как повтор.
    _seen = []

    class _Spy(_FakeAnalyzer):
        def run(self, context, state, message, *, note=None):
            _seen.append(dict(state["counters"]))
            return super().run(context, state, message, note=note)

    _store = sp.InMemoryStateStore()
    _store.create("c1", "split", state=sp.init_state("remote", ""),
                  context="Зарплатная вилка: от 200000 до 280000 рублей",
                  location="Москва", contact_source="hh",
                  salary_band={"min": 200000, "max": 280000})
    _an = _Spy(_decision(salary_claim=good, asking="salary", event="bot_check",
                         instruction="Ответь про бота и переспроси сумму"),
               _decision(asking="format", instruction="Спроси город"))
    _eng = sp.ScreeningSplitEngine(_store, _an, _FakeInterviewer(), None)
    _eng.add_message_and_run("c1", "Вы бот? Ориентируюсь на 250 тысяч")
    _doc = _store.load("c1")
    check("оба вызова видят счётчик нетронутым, инкремент ровно один",
          len(_seen) == 2 and _seen[0]["bot_check"] == _seen[1]["bot_check"] == 0
          and _doc["state"]["counters"]["bot_check"] == 1,
          f"на вызовах={[x['bot_check'] for x in _seen]} итог={_doc['state']['counters']['bot_check']}")

    return out


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


def _domain_sanity_hh() -> List[str]:
    """Санити hh-домена: реестр §5, мультиформат init_state, ветка field_work, hh-валидатор."""
    out: List[str] = []
    ko = sp.render_script("KO_LOCATION", city="Москва")
    out.append(f"render KO_LOCATION(city=Москва): city={'локация Москва' in (ko or '')} · terminal={sp.is_terminal('KO_LOCATION')} · KO_GEO_removed={not sp.is_known('KO_GEO')}")
    st_office = sp.init_state(["ON_SITE", "FIELD_WORK"], "1. Опыт с Python?\n2. SQL?")
    st_remote = sp.init_state(["REMOTE"], "")
    out.append(f"init_state [ON_SITE,FIELD_WORK] format={st_office['format_check']} field_work={st_office['field_work_check']} · [REMOTE] format={st_remote['format_check']} field_work={st_remote['field_work_check']}")
    st2 = sp.apply_updates(st_office, [{"key": "salary", "value": "closed"}, {"key": "field_work_check", "value": "closed"}])
    st3 = sp.apply_updates(st2, [], event="pause")
    out.append(f"apply_updates salary={st2['salary']} field_work={st2['field_work_check']} pause_counter={st3['counters']['pause']} · contact_source_removed={'contact_source' not in st3['counters']}")
    dec, err = sp.parse_and_validate('{"next_action":"ask","script_key":null,"instruction":"Спроси про разъездной формат","updates":[],"event":null,"asking":"field_work"}')
    out.append(f"Decision-валидатор (asking=field_work): valid={dec is not None} err={err or '—'}")
    return out


def _resolve_version(cfg: Dict[str, Any], component: str, cli_version: str | None) -> str | None:
    """CLI > model.yaml[component].local_version > None (pointer.yaml active в пакете)."""
    if cli_version:
        return cli_version
    return component_cfg(cfg, component).get("local_version")


def run(args: argparse.Namespace) -> Any:
    # --- разрешение канала: движок (sp), компоненты промптов, набор фикстур и дефолт-вакансия ---
    global sp, DEFAULT_VACANCY_INFO
    channel = getattr(args, "channel", "tg") or "tg"
    if channel == "hh":
        from qa_harness.domain import screening_split_hh as _sp
        DEFAULT_VACANCY_INFO = DEFAULT_VACANCY_INFO_HH
        analyzer_component, interviewer_component = "screening_analyzer_hh", "screening_interviewer_hh"
        fix_dir = FIXTURES / "screening_split_hh"
        gen_vacancies = FIXTURES / "generation" / "screening_split_hh" / "scenario_vacancies.yaml"
        gen_constraints = FIXTURES / "generation" / "screening_split_hh" / "constraints.yaml"
    else:
        from qa_harness.domain import screening_split as _sp
        analyzer_component, interviewer_component = ANALYZER_COMPONENT, INTERVIEWER_COMPONENT
        fix_dir = FIXTURES / "screening_split"
        gen_vacancies, gen_constraints = DEFAULT_VACANCIES, DEFAULT_CONSTRAINTS
    sp = _sp
    # Движок ставится один раз на весь прогон: конструкторов SplitConversation в раннере несколько
    # (scripted/generated), и тащить параметр через все сигнатуры незачем.
    engine_kind = getattr(args, "engine", "split") or "split"
    sp.conversation.DEFAULT_ENGINE = engine_kind
    runner_name = "screening_split_hh" if channel == "hh" else RUNNER
    csv_path = args.csv or fix_dir / "scenarios.csv"
    checks_path = args.checks or fix_dir / "scenario_checks.yaml"
    inputs_path = args.candidate_inputs or fix_dir / "candidate_inputs.yaml"
    vacancies_path = args.vacancies or gen_vacancies
    constraints_path = args.constraints or gen_constraints

    scenarios = load_scenarios(csv_path)

    if args.offline:
        selected = _select(scenarios, args.scenario_indices, args.sample, args.seed,
                           runnable_only=False, max_examples=args.max_examples)
        print(f"Сценариев в CSV: {len(scenarios)} · выбрано: {len(selected)} · канал: {channel} · CSV: {csv_path}")
        failed = _run_offline(args, selected, channel)
        if failed:
            # Офлайн-режим — единственный гейт кода харнесса (pytest здесь не держим), поэтому
            # провал проверок обязан валить прогон, а не оставаться строчкой в выводе.
            raise SystemExit(f"[offline] провалено проверок зарплатного контракта: {failed}")
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
    a_ver = _resolve_version(cfg, analyzer_component, args.analyzer_version)
    i_ver = _resolve_version(cfg, interviewer_component, args.interviewer_version)

    # пакет prompts (дев-путь --prompts-path / env PROMPTS_REPO_PATH, иначе установленный релиз)
    ensure_prompts_importable(args.prompts_path)
    client = get_client(timeout=args.step1_timeout)
    # общие (read-only) части — строим один раз, шарим по потокам; mutable движок — per-scenario
    analyzer_client = LocalPromptClient(analyzer_component, a_ver, client=client)
    interviewer_spec = load_local_spec(interviewer_component, i_ver)
    a_spec = analyzer_client.spec
    judge = ScenarioJudge(ModelClient(args.eval_model, timeout=args.step1_timeout, temperature=0))
    ijudge = None if args.no_interviewer_judge else sp.InterviewerJudge(
        ModelClient(args.eval_model, timeout=args.step1_timeout, temperature=0))
    vacancies = load_vacancies(vacancies_path)
    checks_by_index = sp.load_checks(checks_path)  # слой A: инварианты Decision
    inputs_by_index = sp.load_candidate_inputs(inputs_path)  # C1: скриптовые входы

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
            constraints_entries=load_constraints(constraints_path),
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
          f"{len(work_items)} · канал: {channel} · режим: {run_args['mode']}/{args.input_mode} · CSV: {csv_path}")
    print(f"Аналитик {a_spec.version}/{a_spec.model} · Интервьюер {interviewer_spec.version}/{interviewer_spec.model} · судья {args.eval_model}"
          + (f" · генератор {args.gen_model}" if args.generate else ""))

    put = {
        "component": runner_name, "source": "local", "prompt_id": None, "prompt_version": None,
        "local_component": f"{analyzer_component} + {interviewer_component}",
        "local_version": f"A:{a_spec.version} · I:{interviewer_spec.version}",
        "model": f"A:{a_spec.model} · I:{interviewer_spec.model}",
    }
    rb = ReportBuilder(
        runner=runner_name, prompt_under_test=put, run_id=run_id,
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
        return write_reports(args.out_dir, runner_name, run_id, md, cd, write_review=False)  # A1: без review.md

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
            for extra in ("rule", "audit", "observation", "guard_trips"):
                # Трасса нового ядра: какое правило выиграло, ЧТО УСЛЫШАЛА модель (`observation`),
                # что срезали гарды. Без правила «гард вырезал» неотличимо от «Интервьюер так и
                # написал»; без наблюдения — «модель не услышала» от «код не исполнил», и разбор
                # 01.09 (смешение формата и локации) пришлось вести отдельным прогоном.
                if t.get(extra):
                    a_turn[extra] = t[extra]
            if t.get("salary"):
                # Зарплатный разбор хода: что код сделал с суммой (годность claim, пересчёт,
                # вердикт, effect). Нужен и инвариантам слоя A, и разбору отчёта глазами.
                a_turn["salary"] = t["salary"]
            if t["end"]:
                a_turn["ended"] = True
            transcript.append(a_turn)
        # --- слой A (Аналитик) + слой B (Интервьюер) с учётом РЕЖИМА ГЕЙТА ---
        # gate="analyzer": детерминированный вход+инварианты → слой A ГЕЙТИТ, ScenarioJudge не звали;
        # gate="dialogue": вход варьируется/нет инвариантов → гейтит LLM-судья, слой A — лишь СИГНАЛ.
        gate = res.get("gate", "dialogue")
        acheck = sp.evaluate_analyzer(s.index, res["turns"], checks_by_index)
        # Канарейки prompt injection (эмодзи / цитирование) — слой B: свойство ТЕКСТА, а не трассы
        # Аналитика, поэтому и в отчёте, и в консоли идут как B, с собственной атрибуцией вины.
        inj = sp.injection_scan(res["turns"], checks_by_index.get(s.index) or {})
        leak = res.get("leak") or {"passed": True, "details": [], "culprit": None}
        iverdict = res.get("iverdict")
        dialogue_passed = True if verdict is None else bool(verdict.passed)
        dialogue_violations = [] if verdict is None else list(verdict.violations)
        dialogue_comment = "" if verdict is None else verdict.comment
        analyzer_ok = (not acheck.has_checks) or acheck.passed
        analyzer_gates = (gate == "analyzer") and acheck.has_checks
        leak_ok = bool(leak["passed"])
        injection_ok = inj.passed
        interviewer_ok = (iverdict is None) or bool(iverdict["passed"])
        # gate: analyzer — инвариант Аналитика ГЕЙТИТ. И в scripted, И в generated: вход варьируется
        # лишь ФОРМУЛИРОВКОЙ (факт закреплён must_convey/рецептом, раунды тугие) → инвариант валиден
        # в обоих режимах, а провал Аналитика в generated НЕ маскируется под passed (кейс 6). dialogue —
        # инвариантов нет, гейтит LLM-судья. Утечка и Интервьюер гейтят всегда.
        if gate == "analyzer":
            core_ok = analyzer_ok
        else:
            core_ok = dialogue_passed
        passed = core_ok and leak_ok and injection_ok and interviewer_ok

        if passed:
            m["passed"] += 1
        else:
            m["failed"] += 1
        if analyzer_gates and not acheck.passed:
            m["analyzer_fail"] += 1
            reasons["[Аналитик] " + "; ".join(acheck.details)[:80]] += 1
        if not leak_ok:
            m[("analyzer_leak" if leak.get("culprit") == "analyzer" else "interviewer_leak")] += 1
        if not injection_ok:
            m["injection_fail"] += 1
            reasons[("[Аналитик] " if inj.culprit == "analyzer" else "[Интервьюер] ")
                    + "prompt injection: " + "; ".join(d for d in inj.details if "OK" not in d)[:70]] += 1
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
        if not injection_ok:
            inj_tag = "[Аналитик] " if inj.culprit == "analyzer" else "[Интервьюер] "
            reason_codes += [inj_tag + d for d in inj.details if "OK" not in d]
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
        if inj.details:  # канарейка объявлена в спеке сценария (tg #60, hh #51)
            case_checks.append({"rule": "Интервьюер: prompt injection (канарейка)", "passed": injection_ok,
                                "detail": "; ".join(inj.details)})
        if iverdict is not None:
            case_checks.append({"rule": "Интервьюер: верность инструкции (LLM)", "passed": bool(iverdict["passed"]),
                                "detail": iverdict["comment"] or "; ".join(iverdict["violations"][:4])})

        v_ord = (variant or 0) + 1  # 1-based: в отчёте вариант нумеруется с 1 (как в консоли v_disp)
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
            b_tag = "" if (leak_ok and injection_ok and interviewer_ok) else " B:FAIL"
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
                # Пер-ходовые сиды (chain-хореография в generated): turn_convey[i] → факты хода i.
                # Генератор отыгрывает последовательность (пауза→продолжить и т.п.), вариативность цела.
                if recipe.get("turn_convey"):
                    c.turn_convey = [sp.resolve_convey(tc if isinstance(tc, list) else [tc], vinfo)
                                     for tc in recipe["turn_convey"]]
                if recipe.get("seed"):
                    c.trigger_requirement = str(recipe["seed"])
                elif not c.trigger_requirement:
                    c.trigger_requirement = s.description
                rec_turns = sp.build_scripted_turns(recipe, vinfo, variant=v, seed=(args.seed or 0), index=s.index)
                if rec_turns and not c.examples:
                    c.examples = rec_turns
                # Раунды ПЕР-СЦЕНАРНО: явный `rounds` рецепта, иначе длина turn_convey (chain), иначе
                # число ходов рецепта. Столько же, сколько отыграет scripted этого сценария (симметрия).
                rec_rounds = recipe.get("rounds") or len(recipe.get("turn_convey") or []) or len(rec_turns) or None
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

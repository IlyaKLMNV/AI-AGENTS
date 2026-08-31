"""Тест НОРМАЛЬНЫХ диалогов: доигрывается ли повестка до конца и чем диалог заканчивается.

Не сценарный прогон. У `screening_split` вопрос «сработал ли на этом ходе нужный скрипт» — там
каждый сценарий про один триггер. Здесь вопрос другой: диалог играется целиком (до 24 ходов), а
ассертится ИТОГ — закрыты ли зарплата и формат, добраны ли все доп-вопросы, каким скриптом
завершились, не начислили ли счётчиков кооперативному кандидату. Ровно этой проверки не хватало под
инцидент 2026-08-17 (ложное завершение): каждый отдельный ход там выглядел законным.

Четыре траектории (`tests/fixtures/screening_split/dialogue_cases.yaml`):
  A — отвечает на всё → FINISH, всё closed, счётчики нули;
  B — расплывчато + честное «нет опыта» по ОДНОМУ навыку → FINISH, БЕЗ STOP_NO_EXPERIENCE
      (решает Наблюдатель: поставит ли он терминальный сигнал на «нет опыта по одному пункту»);
  C — на любой техвопрос «всё есть в резюме» → переспрос, отказ, едем дальше; ни одного STOP;
  D — требует вилку, своей суммы не даёт → STOP_SALARY_DEMAND, а НЕ KO_SALARY (отсеять по деньгам
      того, кто денег не назвал, — исходный баг).

В канале **hh** (`--channel hh`) траектории те же, но повестка из четырёх пунктов и форматов у
вакансии несколько, поэтому:
  A — идёт по лестнице мультиформата: отказ от офиса → согласие на гибрид → разъездной формат.
      Отказ от одного формата не отсев и переспросом не считается (`reasks_zero`);
  E — кандидат в другом городе и переезжать не готов → `KO_LOCATION`. Кейс проверяет, что код САМ
      спросил про переезд: без этого вопроса `relocation_ready` не приходит и ключ недостижим.

Судьи-LLM здесь нет: вердикт детерминированный, по трассе (`checks.evaluate_dialogue`). LLM тратится
на кандидата, Наблюдателя и Интервьюера. Перенос из tgApi (ветка feat/screening-qa,
scripts/screening_qa/dialogue_test.py); пороги перекалиброваны под ядро policy.

Запуск:
    python -m qa_harness.runners.screening_dialogue --prompts-path ../prompts
    python -m qa_harness.runners.screening_dialogue --offline          # без сети, бесплатно (B пропускается)
    python -m qa_harness.runners.screening_dialogue --channel hh --offline
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from qa_harness.core import (
    LOCAL,
    accumulate_usage,
    add_prompt_source_args,
    blank_usage,
    ensure_prompts_importable,
    load_local_spec,
    resolve_source,
    usage_total,
)
from qa_harness.domain import screening_split as sp
from qa_harness.domain.generators import CandidateAgent, CandidateConstraints, GenerationPolicy, VariantSampler
from qa_harness.domain.screening import run_adaptive_conversation
from qa_harness.domain.screening_split.policy.observation import snapshot as observation_snapshot

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_FIXTURE = FIXTURES / "screening_split" / "dialogue_cases.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_dialogue"
ANALYZER_COMPONENT, INTERVIEWER_COMPONENT = "screening_analyzer", "screening_interviewer"
# Версию Аналитика диктует ядро: Observation отдаёт только v3, на v2 каждый ход уйдёт в
# фолбэк. Поэтому это дефолт, а не то, что нужно помнить и дописывать в команду руками.
ANALYZER_VERSION = "v3"
RECRUITER, CANDIDATE = "Анна", "Кандидат"
# Ядро прибито намеренно. Пороги кейсов C/D выведены из `policy/budgets.py` (reask-cap 3,
# у salary_info порога нет), у прежнего движка split они другие — тот же прогон на нём
# ассертил бы не то, что происходит, и «FAIL» означал бы рассинхрон фикстуры, а не баг.
ENGINE = "policy"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"

DEFAULT_VACANCY: Dict[str, Any] = {
    "title": "Python Backend Developer",
    "company_name": "ExampleSoft",
    "responsibilities": "Поддержка и развитие микросервисов, интеграции с продуктами.",
    "work_format": "hybrid",
    "location": "Москва",
    "min_salary": 200000,
    "max_salary": 280000,
    "company_info": {"firm_description": "Продуктовая команда b2b-платформы.",
                     "vacancy_url": "https://example.com/vacancies/python-backend"},
    "questions": "- Опыт с Python и фреймворками?\n- Сервисы под нагрузкой?\n- Как используете SQL?",
}

# hh-канал: форматов у вакансии несколько (`allowed_formats`), «Описание вакансии» вместо
# обязанностей и описания компании, ссылки отдельным полем нет — она живёт внутри описания.
DEFAULT_FIXTURE_HH = FIXTURES / "screening_split_hh" / "dialogue_cases.yaml"
ANALYZER_COMPONENT_HH, INTERVIEWER_COMPONENT_HH = "screening_analyzer_hh", "screening_interviewer_hh"
DEFAULT_VACANCY_HH: Dict[str, Any] = {
    "title": "Инженер пусконаладки",
    "company_name": "ExampleSoft",
    "allowed_formats": ["ON_SITE", "HYBRID", "FIELD_WORK"],
    "location": "Москва",
    "min_salary": 200000,
    "max_salary": 280000,
    "vacancy_description": ("Пусконаладка и поддержка оборудования на объектах заказчика. "
                            "Подробности: https://hh.ru/vacancy/12345678"),
    "questions": "- Опыт с Python и фреймворками?\n- Сервисы под нагрузкой?\n- Как используете SQL?",
}


def _vacancy_for(case: Dict[str, Any], defaults: Dict[str, Any],
                 base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Вакансия кейса: канальный дефолт ← блок `vacancy` фикстуры ← пер-кейсовые оверрайды."""
    vac = {**(base or DEFAULT_VACANCY), **(defaults or {})}
    for field_name in ("work_format", "allowed_formats", "location", "questions", "company_name",
                       "min_salary", "max_salary", "vacancy_description"):
        if field_name in case:
            vac[field_name] = case[field_name]
    return vac


# ── офлайн: то же ядро, но без сети ──────────────────────────────────────────
# Скриптуется только НАБЛЮДЕНИЕ; `decide()`, состояние, счётчики и реестр причин — настоящие.
# Поэтому офлайн проверяет арифметику ядра и форму трассы, но НЕ качество промптов: что «услышал»
# Наблюдатель, здесь решаем мы сами. Кейс, чей предмет проверки — само наблюдение (B), в офлайн не
# допускается вовсе: `modes` в фикстуре, пропуск виден в отчёте как skipped.

_OFFLINE_BEHAVIOUR = {"A": "cooperative", "C": "refuser", "D": "demander", "E": "out_of_town",
                      "F": "out_of_town_silent"}
_DIGITS_RE = re.compile(r"\d[\d\s]*\d|\d")


class _FakeClient:
    """Движку нужен ровно один вызов OpenAI — `conversations.create` ради идентификатора диалога."""

    _ids = itertools.count(1)

    class _Conversations:
        def create(self, **_kwargs: Any) -> Any:
            return type("Conv", (), {"id": f"conv_offline_{next(_FakeClient._ids)}"})()

    def __init__(self) -> None:
        self.conversations = self._Conversations()


class _EchoInterviewer:
    """«Рот» без модели: непустой текст, чтобы ход не свалился в REPLY_FALLBACK. Смысла в нём нет
    намеренно — офлайн судит ядро, а не формулировки."""

    def run(self, _instruction: str, _message: str, *, seed: str = "", last_sent: str = "") -> tuple:
        return "Уточните, пожалуйста.", blank_usage()


def _asked_about_relocation(state: dict) -> bool:
    """Спрашивал ли код прошлым ходом про переезд — по инструкции, которую он сам же и собрал."""
    return "переехать" in (state.get("last_asked") or "")


class _ScriptedObserver:
    """«Уши» без модели: отвечает на ТЕКУЩИЙ фокус, который код положил в `state.last_asking`."""

    def __init__(self, behaviour: str, city: str = "Москва", channel: str = "tg") -> None:
        self.behaviour = behaviour
        self.city = "Казань" if behaviour.startswith("out_of_town") else city
        self.channel = channel
        self.last_usage = blank_usage()

    def run(self, _context: str, state: dict, message: str) -> Any:
        from qa_harness.domain.screening_split.policy.observation import Observation, Signal

        self.last_usage = blank_usage()
        obs = Observation()
        focus = state.get("last_asking")

        if self.behaviour == "demander":
            # Спрашивает про деньги и повторяет одно и то же; своей суммы не даёт — claim пуст.
            obs.signals.append(Signal(code="salary_info", quote=message))
            obs.persistent = True
            return obs

        if focus == "salary":
            m = _DIGITS_RE.search(message)  # цитата обязана дословно найтись в реплике
            if m:
                obs.salary_claim = {"subject": "own_expectation", "form": "exact",
                                    "amount_min": int(re.sub(r"\D", "", m.group(0))), "amount_max": None,
                                    "scale": "unit", "period": "month", "tax": "net",
                                    "currency": "RUB", "quote": m.group(0)}
                obs.focus_answered = "substantive"
            return obs

        if focus in ("format", "field_work"):
            if self.channel == "hh":
                return self._hh_formats(obs, state, focus)
            obs.focus_answered = "substantive"
            if self.behaviour.startswith("out_of_town"):
                # Сначала только город. Отказ от переезда приходит ТОЛЬКО если код про переезд
                # спросил: заглушка отвечает на заданный вопрос, а не выдаёт факты авансом. Без
                # этого условия кейс проверял бы выбор ключа, но не саму правку («код обязан
                # спросить»), и регрессия вопроса осталась бы незамеченной.
                if not state.get("candidate_city"):
                    obs.facts["candidate_city"] = self.city
                elif _asked_about_relocation(state):
                    # `_silent` не высказывается про формат вовсе — второй вход в R6 (кейс F).
                    if self.behaviour != "out_of_town_silent":
                        obs.facts["format_ready"] = "no"
                    obs.facts["relocation_ready"] = "no"
                return obs
            obs.facts["format_ready"] = "yes"
            obs.facts["candidate_city"] = self.city
            return obs

        if focus and focus.startswith("q"):
            # refuser упрямится ТОЛЬКО на первом доп-вопросе — как и живой кандидат кейса C.
            if self.behaviour == "refuser" and focus == "q1":
                obs.focus_answered = "deflection"   # «всё есть в резюме» — ответа по сути нет
                return obs
            obs.answers.append({"key": focus, "substantive": True})
            obs.focus_answered = "substantive"
            return obs

        return obs  # фокуса нет (первый ход / ANSWER_ONLY) — наблюдать нечего

    def _hh_formats(self, obs: Any, state: dict, focus: str) -> Any:
        """hh: готовность отвечается ПО ФОРМАТУ, который код положил в `state.format_asked`.

        Кооперативный кандидат отказывается от офиса и соглашается на остальное — иначе офлайн
        проверял бы единственную ветку «сразу да», а лестница мультиформата (отказ → вопрос про
        следующий формат) осталась бы непройденной.
        """
        obs.focus_answered = "substantive"
        if focus == "field_work":
            obs.facts["formats_ready"] = [{"format": "FIELD_WORK", "ready": "yes"}]
            return obs
        if self.behaviour == "out_of_town":
            # Сначала называет только город, и лишь на вопрос про переезд отвечает отказом: так
            # проверяется, что код САМ спросил про переезд, а не получил ответ авансом.
            if not state.get("candidate_city"):
                obs.facts["candidate_city"] = self.city
            elif _asked_about_relocation(state):
                obs.facts["relocation_ready"] = "no"
            return obs
        obs.facts["candidate_city"] = self.city
        asked = state.get("format_asked")
        if asked:
            obs.facts["formats_ready"] = [{"format": asked,
                                           "ready": "no" if asked == "ON_SITE" else "yes"}]
        return obs


class _ScriptedCandidate:
    """Реплики из `fallback`; последняя повторяется. Контракт `CandidateAgent.next_turn`."""

    def __init__(self, lines: List[str]) -> None:
        self._lines = list(lines) or ["Здравствуйте."]

    def next_turn(self, _history: Any, _last_reply: Any, turn_index: int = 0) -> Any:
        line = self._lines[min(turn_index, len(self._lines) - 1)]
        return type("GenResult", (), {"ok": True, "item": line, "source": "scripted",
                                      "usage": blank_usage(), "attempts": 1, "errors": []})()


class _OfflineConversation:
    """`start()/respond()` поверх ядра policy на заглушках — контракт `SplitConversation`."""

    def __init__(self, vacancy_info: Dict[str, Any], behaviour: str, channel: str = "tg") -> None:
        if channel == "hh":
            from qa_harness.domain.screening_split_hh.policy.engine import PolicyEngine
        else:
            from qa_harness.domain.screening_split.policy.engine import PolicyEngine

        self._channel = channel
        self._engine = PolicyEngine(sp.InMemoryStateStore(),
                                    _ScriptedObserver(behaviour, channel=channel),
                                    _EchoInterviewer(), _FakeClient())
        self._vacancy_info = vacancy_info
        self._cid: Optional[str] = None

    def start(self) -> str:
        # В hh у движка нет параметра accounts: источника контакта в канале не существует.
        if self._channel == "hh":
            self._cid = self._engine.create_thread(self._vacancy_info, RECRUITER, CANDIDATE)
        else:
            self._cid = self._engine.create_thread(self._vacancy_info, RECRUITER, CANDIDATE, [])
        return self._cid

    def respond(self, message: str) -> Any:
        result = self._engine.add_message_and_run(self._cid, message)
        st = self._engine.last_state or {}
        # Снимок один на оба канала: ключей hh (field_work_check / formats) в tg-состоянии нет, и
        # проверки по ним там просто не заказываются.
        state = {"salary": st.get("salary"), "format_check": st.get("format_check"),
                 "field_work_check": st.get("field_work_check"),
                 "city": st.get("candidate_city"),
                 "relocation_ready": st.get("relocation_ready"),
                 "formats": dict(st.get("formats") or {}),
                 "questions": {q["key"]: q["status"] for q in st.get("questions", [])},
                 "counters": dict(st.get("counters", {})),
                 "reasks": {"salary": st.get("salary_reasks", 0),
                            "format": st.get("format_reasks", 0),
                            "field_work": st.get("field_work_reasks", 0)}} if st else None
        plan = self._engine.last_plan
        obs = getattr(self._engine, "last_observation", None)
        return sp.TurnResult(
            response=result.response, conversation_end=bool(result.conversation_end),
            usage=dict(self._engine.last_usage),
            tool_trace={"decision": self._engine.last_decision, "state": state,
                        "salary": self._engine.last_salary,
                        "rule": getattr(plan, "rule", None),
                        "observation": observation_snapshot(obs) if obs is not None else None},
        )


# ── прогон кейса ─────────────────────────────────────────────────────────────

def _run_case(case: Dict[str, Any], vacancy: Dict[str, Any], *, offline: bool, max_turns: int,
              channel: str = "tg",
              client: Any = None, analyzer_client: Any = None, interviewer_spec: Any = None,
              gen_client: Any = None, gen_model: str = DEFAULT_GEN_MODEL,
              gen_policy: Any = None, style: Any = None) -> Dict[str, Any]:
    """Один диалог до завершения или до `max_turns`, затем детерминированный вердикт по трассе."""
    key = case.get("key", "?")
    if offline:
        conv: Any = _OfflineConversation(vacancy, _OFFLINE_BEHAVIOUR.get(key, "cooperative"), channel)
        agent: Any = _ScriptedCandidate(case.get("fallback") or [])
    else:
        conv = sp.SplitConversation(client=client, analyzer_client=analyzer_client,
                                    interviewer_spec=interviewer_spec, vacancy_info=vacancy,
                                    recruiter_name=RECRUITER, candidate_name=CANDIDATE)
        constraints = CandidateConstraints(
            scenario_name=case.get("title", key), scenario_description=str(case.get("expect") or ""),
            guidelines=case.get("guidelines") or [], fallback=case.get("fallback") or [])
        agent = CandidateAgent(gen_client, gen_model, constraints, style, policy=gen_policy)

    result = run_adaptive_conversation(conv, agent, max_turns=max_turns)
    turns: List[Dict[str, Any]] = []
    usage = blank_usage()
    for t in result.turns:
        tr = t.tool_trace or {}
        accumulate_usage(usage, t.assistant_usage or {})
        accumulate_usage(usage, t.gen_usage or {})
        turns.append({"candidate": t.candidate, "reply": t.reply, "end": t.end,
                      "decision": tr.get("decision"), "state": tr.get("state"),
                      "salary": tr.get("salary"), "rule": tr.get("rule"),
                      "observation": tr.get("observation")})

    # usage — СЫРОЙ bucket (input_tokens/…): `usage_total` даёт другие ключи, и повторный
    # accumulate_usage поверх них молча прибавлял нули — итог прогона выходил нулевым.
    out = {"key": key, "title": case.get("title", ""), "expect": case.get("expect") or {},
           "turns": turns, "usage": usage, "error": result.error,
           "passed": False, "details": [], "checks": []}
    if result.error:
        return out                      # инфра-сбой (кандидат/движок) — не «промпт не прошёл»
    if not turns:
        out["error"] = "empty_dialogue"
        return out

    verdict = sp.evaluate_dialogue(turns, out["expect"])
    out["passed"], out["details"], out["checks"] = verdict.passed, verdict.details, verdict.items
    return out


def _turn_tag(turn: Dict[str, Any]) -> str:
    """Ярлык хода: что сделали и каким правилом. Имя правила важно — без него «сработал STOP»
    неотличимо от «сработал не тот STOP»."""
    dec = turn.get("decision") or {}
    what = dec.get("script_key") or dec.get("asking") or dec.get("reason_code")
    parts = [f"{dec.get('next_action')}" + (f"/{what}" if what else "")]
    if dec.get("event"):
        parts.append(f"ev={dec['event']}")
    if turn.get("rule"):
        parts.append(f"r={turn['rule']}")
    return "[" + " ".join(parts) + "]"


def _transcript(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Диалог для cases.json в форме `screening_split`: реплика + трасса ЭТОГО хода на ней самой.

    Отдельного «человеческого» файла нет — читаемость даёт сам cases.json (см.
    docs/screening_split/report_analysis.md §1), поэтому решение/правило/состояние лежат рядом с
    текстом, а не в параллельном массиве, который пришлось бы сопоставлять по индексу.
    """
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(turns, 1):
        out.append({"round": i, "role": "candidate", "text": t["candidate"]})
        dec = t.get("decision") or {}
        na = dec.get("next_action")
        out.append({"round": i, "role": "assistant", "text": t["reply"], "end": bool(t["end"]),
                    # Кто автор текста: Интервьюер (ход-вопрос) или код (скрипт из реестра причин).
                    # Без этого «плохая формулировка» неотличима от «плохой шаблон».
                    "turn_kind": "interviewer_reply" if na == "ask" else ("script" if na == "script" else "fallback"),
                    "analyzer_instruction": dec.get("instruction"),
                    # Что модель УСЛЫШАЛА на этом ходе. Без этого по отчёту не отличить, какой вход
                    # правила сработал: у R6 их два, и «отказался от формата» неотличимо от
                    # «отказался переезжать» (разбор прогона 20260831_203510).
                    "observation": t.get("observation"),
                    "tag": _turn_tag(t), "decision": dec, "state": t.get("state"),
                    "salary": t.get("salary"), "rule": t.get("rule")})
    return out


def run(args: argparse.Namespace) -> None:
    # --- разрешение канала: движок, компоненты промптов, фикстура, дефолт-вакансия ---
    global sp
    channel = getattr(args, "channel", "tg") or "tg"
    if channel == "hh":
        from qa_harness.domain import screening_split_hh as _sp
        sp = _sp
        analyzer_component, interviewer_component = ANALYZER_COMPONENT_HH, INTERVIEWER_COMPONENT_HH
        base_vacancy, default_fixture = DEFAULT_VACANCY_HH, DEFAULT_FIXTURE_HH
        runner_name = f"{RUNNER}_hh"
    else:
        analyzer_component, interviewer_component = ANALYZER_COMPONENT, INTERVIEWER_COMPONENT
        base_vacancy, default_fixture = DEFAULT_VACANCY, DEFAULT_FIXTURE
        runner_name = RUNNER

    fixture_path = Path(args.fixture) if args.fixture else default_fixture
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8")) or {}
    cases = fixture.get("cases") or []
    if args.case != "all":
        cases = [c for c in cases if c.get("key") == args.case]
        if not cases:
            raise SystemExit(f"кейса {args.case} нет в {fixture_path}")

    # Кейс, чей предмет проверки в этом режиме подменён заглушкой, НЕ гоняем: зелёный результат,
    # ничего не доказавший, вводит в заблуждение сильнее, чем его отсутствие.
    mode = "offline" if args.offline else "online"
    skipped = [c for c in cases if mode not in (c.get("modes") or ["offline", "online"])]
    cases = [c for c in cases if c not in skipped]
    for c in skipped:
        print(f"[dialogue] SKIP {c.get('key')} — в режиме {mode} кейс ничего не проверяет "
              f"(modes: {c.get('modes')})")
    if not cases:
        raise SystemExit(f"в режиме {mode} гонять нечего — все выбранные кейсы пропущены.")

    setup: Dict[str, Any] = {"channel": channel}
    models: Dict[str, Any] = {}
    put: Dict[str, Any] = {"component": runner_name, "source": "local"}

    if args.offline:
        # Промпта под тестом в офлайне нет вовсе — не даём отчёту утверждать обратное.
        put.update({"local_component": "— (офлайн: наблюдение скриптовано)", "local_version": "—",
                    "model": "—", "prompt_id": None, "prompt_version": None})
        print(f"[dialogue] ОФЛАЙН · канал {channel} · ядро {ENGINE} на заглушках · кейсов: {len(cases)}")
    else:
        source = resolve_source(args.prompt_source or os.environ.get("QA_HARNESS_PROMPT_SOURCE") or LOCAL)
        if source != LOCAL:
            raise SystemExit("screening_dialogue — только local (у split-промптов нет stored-эквивалента).")
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (экспортируй: set -a; source .env; set +a)")
        # Пустой OPENAI_BASE_URL= в .env экспортится как "" и ломает SDK — трактуем как «не задано».
        if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
            os.environ.pop("OPENAI_BASE_URL", None)

        ensure_prompts_importable(args.prompts_path)
        from qa_harness.core.llm_client import LocalPromptClient, ModelClient, get_client

        client = get_client(timeout=args.timeout)
        analyzer_client = LocalPromptClient(analyzer_component, args.analyzer_version, client=client)
        interviewer_spec = load_local_spec(interviewer_component, args.interviewer_version)
        a_spec = analyzer_client.spec
        if a_spec.version != ANALYZER_VERSION:
            # Ядро policy читает Observation, а его отдаёт только v3. На v2 наблюдение развалится
            # в валидации — все ходы уйдут в фолбэк, и это будет выглядеть поломкой ядра.
            print(f"  ВНИМАНИЕ: ядру {ENGINE} нужен {analyzer_component} {ANALYZER_VERSION}, "
                  f"взят {a_spec.version} — ходы уйдут в фолбэк")
        setup.update(client=client, analyzer_client=analyzer_client, interviewer_spec=interviewer_spec,
                     gen_client=ModelClient(args.gen_model, timeout=args.timeout, temperature=args.temperature),
                     gen_model=args.gen_model,
                     gen_policy=GenerationPolicy(max_retries=1, temperature=args.temperature, seed=args.seed),
                     style=VariantSampler(args.seed or 0).at(0))
        models = {"candidate_generator": args.gen_model, "analyzer": a_spec.model,
                  "interviewer": interviewer_spec.model}
        put.update({"local_component": f"{analyzer_component} + {interviewer_component}",
                    "local_version": f"A:{a_spec.version} · I:{interviewer_spec.version}",
                    "model": f"A:{a_spec.model} · I:{interviewer_spec.model}",
                    "prompt_id": None, "prompt_version": None})
        print(f"[dialogue] канал {channel} · Аналитик {a_spec.version}/{a_spec.model} · Интервьюер "
              f"{interviewer_spec.version}/{interviewer_spec.model} · кандидат {args.gen_model} · "
              f"ядро {ENGINE} · кейсов: {len(cases)}")

    sp.conversation.DEFAULT_ENGINE = ENGINE
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
    rb = ReportBuilder(runner=runner_name, prompt_under_test=put, run_id=run_id,
                       started_at=started.isoformat(timespec="seconds"), models=models,
                       seed=args.seed,
                       args={"engine": ENGINE, "channel": channel, "offline": bool(args.offline),
                             "max_turns": args.max_turns, "cases": len(cases)})

    total_usage = blank_usage()
    for case in cases:
        vacancy = _vacancy_for(case, fixture.get("vacancy") or {}, base_vacancy)
        print(f"\n=== Кейс {case.get('key')}: {case.get('title')} ===", flush=True)
        r = _run_case(case, vacancy, offline=args.offline, max_turns=args.max_turns, **setup)
        for i, t in enumerate(r["turns"], 1):
            print(f"  К{i}: {t['candidate'][:75]}")
            print(f"     А{_turn_tag(t)}: {(t['reply'] or '(нет)')[:85]}", flush=True)
        accumulate_usage(total_usage, r["usage"])

        if r["error"]:
            print(f"  → ERR: {r['error']}", flush=True)
            rb.add_error(str(r["key"]), r["error"], {"title": r["title"]})
            continue
        print(f"  → {'PASS' if r['passed'] else 'FAIL'}", flush=True)
        for d in r["details"]:
            print(f"        - {d}", flush=True)
        rb.add_case(CaseRecord(
            case_id=str(r["key"]), source="offline" if args.offline else "generated", passed=r["passed"],
            inputs={"criterion": r["title"], "expect": r["expect"],
                    "work_format": vacancy.get("allowed_formats") or vacancy.get("work_format")},
            # Вердикт детерминированный: reason_codes — имена НЕсработавших инвариантов
            # (контролируемый словарь), а вся конкретика — в checks[].
            verdict={"passed": r["passed"], "evaluator": "dialogue_invariants",
                     "reason_codes": [c["rule"] for c in r["checks"] if not c["passed"]]},
            checks=r["checks"],
            transcript=_transcript(r["turns"]),
            output={"final_state": (r["turns"][-1] or {}).get("state"),
                    "turns": len(r["turns"]), "usage": usage_total(r["usage"])},
        ))

    rb.set_token_usage(usage_total(total_usage))
    finished = datetime.datetime.now()
    metrics_doc, cases_doc = rb.finalize(
        # Пропущенные — в метриках, а не только в консоли: иначе «4/4 PASS» в офлайне и онлайне
        # выглядят одинаково, хотя проверено разное.
        {"skipped": [{"case": c.get("key"), "modes": c.get("modes"), "reason": f"не гоняется в {mode}"}
                     for c in skipped], "mode": mode},
        finished_at=finished.isoformat(timespec="seconds"),
        duration_s=round((finished - started).total_seconds(), 3))
    # Два файла, как у screening_split: человекочитаемость даёт cases.json (трасса лежит на самой
    # реплике), третий файл с тем же содержанием в другой разметке только расходится с ним.
    metrics_path, _ = write_reports(args.out_dir, runner_name, run_id, metrics_doc, cases_doc,
                                    write_review=False)
    s = metrics_doc["summary"]
    print(f"\n[dialogue] passed={s['passed']}/{s['total']} · errors={s['errors']} · "
          f"skipped={len(skipped)} · токены={s['token_usage'].get('total', 0)} → {metrics_path.parent}")
    if args.offline and (s["failed"] or s["errors"]):
        # Офлайн — единственный гейт кода харнесса (pytest в репозитории не держим), поэтому
        # провал обязан валить прогон, а не оставаться строчкой в выводе.
        raise SystemExit(f"[offline] провалено: failed={s['failed']} errors={s['errors']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Тест нормальных диалогов: доигранная повестка и итог (A/B/C/D[/E]).")
    p.add_argument("--channel", choices=["tg", "hh"], default="tg",
                   help="Канал: tg (screening_analyzer/interviewer, фикстура screening_split) | "
                        "hh (screening_analyzer_hh/interviewer_hh, фикстура screening_split_hh: "
                        "мультиформат, разъездной формат, кейс E про локацию).")
    # F есть только в tg: в hh тот же случай забирает R5a, и его проверяет E.
    p.add_argument("--case", choices=["A", "B", "C", "D", "E", "F", "all"], default="all")
    p.add_argument("--fixture", type=Path, default=None,
                   help="YAML кейсов (по умолч. — по каналу: fixtures/screening_split[_hh]/dialogue_cases.yaml).")
    # Лимит переспросов одного пункта — 3 (policy/budgets.REASK_BUDGETS), доп-вопросов четыре:
    # кейсу C нужно заметно больше ходов, чем сценарным прогонам, иначе повестка не доигрывается.
    p.add_argument("--max-turns", type=int, default=24)
    p.add_argument("--offline", action="store_true",
                   help="ядро на заглушках: без сети и токенов.")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="Модель LLM-кандидата.")
    p.add_argument("--analyzer-version", default=ANALYZER_VERSION, metavar="vN",
                   help=f"Версия Аналитика канала (по умолч. {ANALYZER_VERSION} — её требует ядро).")
    p.add_argument("--interviewer-version", default=None, metavar="vN",
                   help="Версия Интервьюера канала (по умолч. pointer.yaml active).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Температура LLM-кандидата. >0 — вариативные реплики, прогон перестаёт быть гейтом.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_prompt_source_args(p, local_only=True, versioned=False)
    return p


# Флаги, которые в офлайне не на что применить: моделей там нет вовсе.
_ONLINE_ONLY = ("gen_model", "analyzer_version", "interviewer_version", "temperature",
                "timeout", "prompts_path")


def _reject_dead_combos(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Явно заданный флаг, который режим не читает, — молча проглатывать нельзя: человек уверен,
    что задал версию промпта или температуру, а прогон идёт мимо них."""
    if not args.offline:
        return
    given = [d for d in _ONLINE_ONLY if getattr(args, d, None) != parser.get_default(d)]
    if given:
        flags = ", ".join("--" + d.replace("_", "-") for d in given)
        raise SystemExit(f"{flags}: в --offline моделей нет, флаг ни на что не влияет. "
                         f"Убери его либо убери --offline.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _reject_dead_combos(parser, args)
    run(args)


if __name__ == "__main__":
    main()

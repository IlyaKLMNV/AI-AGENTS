"""`decide()` — чистое ядро хода, HH-канал. Ни сети, ни стора, ни вызовов LLM.

Порядок внутри хода тот же, что в TG (`screening_split/policy/core.py`), и по той же причине: код
СЧИТАЕТ раньше, чем выбирается действие, поэтому решения, с которым можно не согласиться, не
существует и перерешивать нечего.

    1. зарплата: claim → статус → пересчёт → вилка → односторонние гейты → вердикт
    2. факты и ответы: монотонно в состояние (в hh сюда же — накопление ответов ПО ФОРМАТАМ)
    3. счётчики: ровно один за ход
    4. фокус и бюджеты переспросов
    5. таблица правил: первое сработавшее выигрывает
    6. сборка инструкции: черновик модели + вопрос из шаблона кода

Канальная дельта, из-за которой файл не сводится к импорту TG-ядра:

- **повестка из четырёх пунктов**: зарплата → формат → разъездной формат → доп-вопросы;
- **мультиформат**: `format_check` закрывается готовностью к ЛЮБОМУ допустимому присутственному
  формату, и код сам выбирает, о каком формате спрашивать следующим (`.formats`). Отказ от одного
  формата вопрос не закрывает и вакансию не рубит;
- **`format_asked`**: код пишет в состояние, о каком формате спросил, — по нему Наблюдатель относит
  короткое «да»/«нет» к конкретному формату, а не гадает по тексту прошлой инструкции;
- зарплатный разбор, гейты, эскалация переспросов, благодарность и вводная перед доп-вопросами
  идентичны TG — переиспользуются импортом.
"""

from dataclasses import dataclass, field
from typing import Optional

from qa_harness.domain.screening_split import salary as salary_mod
from qa_harness.domain.screening_split.policy.core import (
    ACKNOWLEDGE,
    QUESTIONS_INTRO,
    SalaryResult,
    TurnPlan,
    _CONVEY_ORDER,
    _LAST_CALL,
    _SALARY_WHY,
    _SIGNAL_CONVEY,
    _resolve_salary,
)

from qa_harness.domain.screening_split.policy.geo import relocation_pointless

from .. import state as state_model
from . import formats, reasons
from .budgets import EVENT_BUDGETS, REASK_BUDGETS, config_digest
from .observation import SIGNAL_TO_COUNTER, Observation, formats_ready
from .rules import RESUME_AFTER_REFUSE, RULES, Outcome

REFUSE_AND_ADVANCE = "REFUSE_AND_ADVANCE"

# Ступень «предупреди о последствии» для разъездного формата. Кап у него тот же, что у формата
# (`STOP_PERSISTENT`), поэтому и предупреждение той же силы.
_LAST_CALL_HH = {**_LAST_CALL,
                 "field_work": "Предупреди: без ответа про разъездной формат продолжить скрининг не сможем."}


# ── вход ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecideContext:
    """Всё о вакансии, что нужно КОДУ. Вилка приходит типизированной и Наблюдателю не показывается.

    Полей `work_format` и `contact_source` здесь нет: форматы живут в состоянии
    (`allowed_formats`, их несколько), источника контакта в канале не существует.
    """

    band_min: Optional[int] = None
    band_max: Optional[int] = None
    band_currency: str = "RUB"
    location: str = ""
    has_geo_restriction: bool = False


@dataclass
class Frame:
    """Внутренняя рамка хода — её видят предикаты правил."""

    state: dict
    obs: Observation
    ctx: DecideContext
    salary: SalaryResult
    dialogue_closed: bool = False
    analyzer_failed: bool = False
    charged_event: Optional[str] = None
    counter_before: int = 0
    reask_fired: Optional[tuple] = None
    reask_candidate: Optional[tuple] = None
    focus: Optional[str] = None
    # Формат, о котором код спросит на этом ходе (`ON_SITE`/`HYBRID`/`FIELD_WORK`) либо None.
    ask_format: Optional[str] = None
    agenda_complete: bool = False
    no_progress_now: int = 0
    skipped_signals: list = field(default_factory=list)

    @property
    def counters(self) -> dict:
        return self.state.get("counters", {})


# ── проекция состояния для промпта ────────────────────────────────────────────

def state_for_prompt(state: dict) -> dict:
    """То, что видит Наблюдатель. Служебные поля и счётчики не сериализуются.

    `allowed_formats` остаётся: это единственный источник правды о форматах вакансии, и промпт на
    него ссылается. `format_asked` — новое поле канала: по нему модель относит короткое «да»/«нет»
    к конкретному формату, вместо того чтобы вычитывать это из прозы `last_asked`.
    """
    return {
        "salary": state.get("salary"),
        "allowed_formats": list(state.get("allowed_formats") or []),
        "format_check": state.get("format_check"),
        "field_work_check": state.get("field_work_check"),
        "format_asked": state.get("format_asked"),
        "candidate_city": state.get("candidate_city"),
        "questions": [{"key": q.get("key"), "text": q.get("text"), "status": q.get("status")}
                      for q in state.get("questions", [])],
        "last_asked": state.get("last_asked"),
    }


# ── шаг 2: факты и ответы → состояние ─────────────────────────────────────────

def _updates_from_observation(obs: Observation, state: dict) -> list[dict]:
    """Наблюдение → монотонная дельта состояния (город и ответы на доп-вопросы).

    Ключа `salary` здесь нет и быть не может: пункт закрывает только код и только после сравнения с
    вилкой. Формат идёт не сюда, а в `_apply_formats`: он не «ключ=значение», а накопление ответов по
    нескольким форматам сразу.
    """
    updates: list[dict] = []

    city = obs.facts.get("candidate_city")
    if isinstance(city, str) and city.strip():
        updates.append({"key": "candidate_city", "value": city.strip()})

    known = {q["key"] for q in state.get("questions", [])}
    for answer in obs.answers:
        key = str(answer.get("key") or "").strip()
        if key in known and answer.get("substantive"):
            updates.append({"key": key, "value": "closed"})
    return updates


def _apply_formats(state: dict, obs: Observation) -> dict:
    """Ответы про форматы → состояние, затем закрытие `format_check` / `field_work_check`.

    Накопление обязательно: Observation отдаёт только сказанное на ЭТОМ ходе, а «отказался от всех
    допустимых» — факт всего диалога. Поздний ответ перекрывает ранний: кандидат вправе передумать
    («ладно, в офис готов»), и последнее сказанное вернее.

    Закрытия ровно три, и все три — вывод из накопленного, а не отдельное суждение модели:
      * готов хотя бы к одному допустимому формату (или к переезду) → `format_check` закрыт;
      * про все присутственные форматы уже высказался → закрывать больше нечего, даже если везде
        «нет»: отсев решит правило R6, а фокус должен уехать дальше, иначе следующий ход спросит
        про формат, о котором спрашивать нечего;
      * про разъездной высказался явно → `field_work_check` закрыт. Исключение — «нет» при
        единственном допустимом `FIELD_WORK`: там это отказ от вакансии, и проверку закрывать нельзя.
    """
    new = state_model.apply_updates(state, [])
    said = dict(new.get("formats") or {})
    said.update(formats_ready(obs))
    new["formats"] = said

    relocation = obs.facts.get("relocation_ready")
    if relocation in ("yes", "no"):
        new["relocation_ready"] = relocation

    if new.get("format_check") == "pending":
        if formats.confirmed_any(new) or new.get("relocation_ready") == "yes":
            new["format_check"] = "closed"
        elif formats.next_presence_to_ask(new) is None:
            new["format_check"] = "closed"

    if new.get("field_work_check") == "pending":
        answer = said.get("FIELD_WORK")
        if answer == "yes" or (answer == "no" and not formats.field_work_only(new)):
            new["field_work_check"] = "closed"

    return new


def _stall_count(state: dict, progress_before: tuple, before: int) -> int:
    """Сколько ходов подряд диалог не собрал ничего нового, ВКЛЮЧАЯ текущий."""
    if state_model.progress_signature(state) == progress_before:
        return before + 1
    return 0


# ── шаг 3: счётчики ───────────────────────────────────────────────────────────

def _charge_counter(obs: Observation, state: dict) -> tuple[dict, Optional[str], int]:
    """Начисляет РОВНО ОДИН счётчик за ход и возвращает (state, ключ, значение до хода)."""
    key: Optional[str] = None
    for code in obs.codes():
        if code in SIGNAL_TO_COUNTER:
            key = SIGNAL_TO_COUNTER[code]
            break
    if key is None and obs.persistent:
        key = "demand"
    if key is None:
        return state, None, 0

    before = state.get("counters", {}).get(key, 0)
    budget = EVENT_BUDGETS.get(key)
    new_state = state_model.apply_updates(state, [])
    if budget is None or budget.persist_on_fire or not budget.fires(before):
        new_state["counters"][key] = before + 1
    return new_state, key, before


# ── шаг 4: фокус и переспросы ─────────────────────────────────────────────────

def _focus_of(state: dict) -> Optional[str]:
    """Первый незакрытый пункт повестки: зарплата → формат → разъездной → доп-вопросы."""
    if state.get("salary") != "closed":
        return "salary"
    if state.get("format_check") == "pending":
        return "format"
    if state.get("field_work_check") == "pending":
        return "field_work"
    pending = state_model.pending_questions(state)
    return pending[0]["key"] if pending else None


def _format_to_ask(state: dict, focus: Optional[str]) -> Optional[str]:
    """Про какой формат код спросит на этом ходе."""
    if focus == "format":
        return formats.next_presence_to_ask(state)
    if focus == "field_work":
        return "FIELD_WORK"
    return None


def _reask_candidate(state_before: dict, state_now: dict, focus: Optional[str],
                     ask_format: Optional[str]) -> Optional[tuple]:
    """(вид бюджета, бюджет, значение счётчика до хода) — если код в прошлый ход спрашивал ТО ЖЕ.

    «То же» для форматов строже, чем `last_asking`: переход с офиса на гибрид — это НОВЫЙ вопрос, а
    не переспрос, поэтому сверяется ещё и `format_asked`. Иначе кандидат, честно ответивший про
    каждый допустимый формат, сжигал бы кап на собственных ответах.
    """
    if not focus or state_before.get("last_asking") != focus:
        return None
    if focus == "salary":
        if state_now.get("salary") != "pending":
            return None
        return "salary", REASK_BUDGETS["salary"], state_now.get("salary_reasks", 0)
    if focus == "format":
        if state_now.get("format_check") != "pending":
            return None
        if state_before.get("format_asked") != ask_format:
            return None
        return "format", REASK_BUDGETS["format"], state_now.get("format_reasks", 0)
    if focus == "field_work":
        if state_now.get("field_work_check") != "pending":
            return None
        return "field_work", REASK_BUDGETS["field_work"], state_now.get("field_work_reasks", 0)
    question = next((q for q in state_now.get("questions", []) if q["key"] == focus), None)
    if question is None or question.get("status") != "pending":
        return None
    return "question", REASK_BUDGETS["question"], question.get("reask_count", 0)


# ── шаг 6: сборка инструкции ──────────────────────────────────────────────────

def _reasks_of(focus: str, state: dict) -> int:
    """Сколько раз этот пункт УЖЕ переспросили: 0 — первый вопрос, 1 — первый переспрос."""
    if focus == "salary":
        return int(state.get("salary_reasks", 0))
    if focus == "format":
        return int(state.get("format_reasks", 0))
    if focus == "field_work":
        return int(state.get("field_work_reasks", 0))
    question = next((q for q in state.get("questions", []) if q["key"] == focus), None)
    return int((question or {}).get("reask_count", 0))


def _format_slot(state: dict, ctx: DecideContext, ask_format: str, reasks: int) -> str:
    """Вопрос про КОНКРЕТНЫЙ формат — текстом кода, с подставленными значениями.

    Второй заход (кандидат отказался от офиса, спрашиваем про гибрид) звучит как предложение
    альтернативы, а не как повтор того же требования: иначе сообщение читается «вы сказали нет —
    повторяю вопрос».
    """
    human = formats.HUMAN.get(ask_format, "требуемый формат работы")
    where = f" в городе {ctx.location}" if ctx.location and ask_format != "FIELD_WORK" else ""
    already_refused = any(v == "no" for v in formats.answers(state).values())

    if ask_format == "FIELD_WORK":
        parts = ["Донеси: вакансия предполагает разъездной формат — поездки к клиентам или между "
                 "объектами."]
    elif already_refused and reasks == 0:
        short = formats.HUMAN_SHORT.get(ask_format, human)
        where_alt = f", в городе {ctx.location}" if ctx.location else ""
        parts = [f"Донеси: у вакансии допустим ещё один формат работы — {short}{where_alt}."]
    else:
        parts = [f"Донеси: формат работы по вакансии — {human}{where}."]

    # Жёсткость приходит ступенью переспроса, а не с первого хода: «это обязательное требование» в
    # ответ на только что названную зарплату звучит ультиматумом (то же решение, что в TG).
    if reasks >= 1:
        parts.append("Объясни: без ответа по формату работы дальше двигаться не получится.")
    if reasks >= 2:
        parts.append(_LAST_CALL_HH["field_work" if ask_format == "FIELD_WORK" else "format"])

    # Четыре формы вопроса. Про переезд спрашиваем, когда город известен и локации не совпадают:
    # без этого вопроса `relocation_ready` приходил только самотёком, а значит `KO_LOCATION` был
    # недостижим, и кандидат из другого города отсеивался по формату (`KO_FORMAT`) с неверным
    # объяснением. Радиус не нужен: спрашиваем не «далеко ли вы», а «готовы ли работать отсюда».
    if ask_format == "FIELD_WORK":
        ask = "готов ли кандидат работать в таком формате."
    elif not state.get("candidate_city"):
        ask = "в каком городе находится кандидат и готов ли он работать в таком формате."
    elif not relocation_pointless(state, ctx):
        # Локация названа предыдущей фразой, поэтому здесь «этот город»: подстановка названия в
        # падеж дала бы «переехать в Москва».
        ask = ("готов ли кандидат работать в таком формате и готов ли он переехать в этот город "
               "или работать из него.")
    else:
        ask = "готов ли кандидат работать в таком формате."
    parts.append(("Спроси, " if reasks == 0 else "Переспроси, ") + ask)
    return " ".join(parts)


def _ask_slot(focus: str, state: dict, ctx: DecideContext, ask_format: Optional[str]) -> str:
    """Вопрос текущего фокуса — текстом КОДА. Директива без значения здесь невыразима: значение
    подставляет код, сочинять Интервьюеру нечего."""
    reasks = _reasks_of(focus, state)
    if focus == "salary":
        if reasks == 0:
            return ("Спроси, на какую сумму на руки в месяц ориентируется кандидат. "
                    "Конкретных чисел вилки не называй.")
        asked_band = state.get("counters", {}).get("salary_info", 0) >= 1
        parts = ["Переспроси, на какую сумму на руки в месяц ориентируется кандидат.",
                 _SALARY_WHY[asked_band]]
        if reasks >= 2:
            parts.append(_LAST_CALL_HH["salary"])
        parts.append("Конкретных чисел вилки не называй.")
        return " ".join(parts)
    if focus in ("format", "field_work"):
        if not ask_format:
            return ""
        return _format_slot(state, ctx, ask_format, reasks)
    question = next((q for q in state.get("questions", []) if q["key"] == focus), None)
    if question is None:
        return ""
    verb = "Задай" if reasks == 0 else "Повтори"
    text = f"{verb} дополнительный вопрос по теме: «{question['text']}»."
    if reasks >= 2:
        text += " " + _LAST_CALL_HH["question"]
    return text


def _needs_questions_intro(focus: str, state: dict) -> bool:
    """Первый доп-вопрос за диалог, и вводную ещё не говорили."""
    if state.get("questions_intro_sent"):
        return False
    return any(q.get("key") == focus for q in state.get("questions", []))


def _convey_slot(obs: Observation, state: dict, focus: Optional[str]) -> str:
    """Что донести кандидату перед вопросом, помимо ответа модели по существу вакансии.

    Две директивы кода не должны противоречить друг другу: пояснительный convey «ответ нужен здесь,
    в чате» не ставится рядом с приоритетными пунктами — у них своё объяснение приезжает ступенью
    переспроса.
    """
    for code in _CONVEY_ORDER:
        if not obs.has(code):
            continue
        if code == "pause" and state.get("counters", {}).get("pause", 0) >= 2:
            continue  # со 2-й паузы ход забирает HOLD_PAUSE, вопроса там нет вовсе
        return _SIGNAL_CONVEY[code]
    if obs.focus_answered == "deflection" and focus not in ("salary", "format", "field_work"):
        return ("Скажи, что понимаешь кандидата, и мягко объясни: ответ нужен именно здесь, в чате, "
                "ссылки на резюме или профиль недостаточно.")
    return ""


def _build_instruction(obs: Observation, outcome: Outcome, state: dict, ctx: DecideContext,
                       ask_format: Optional[str], *, acknowledge: bool = False) -> tuple[str, list[dict]]:
    """`instruction` = [благодарность] + [черновик модели] + [вопрос из шаблона кода].

    Источник каждой части остаётся в `instruction_parts[].origin`: без этого исчезает атрибуция
    трёхслойного гейта — инварианты должны проверять модельную часть, а не вопрос, собранный кодом.
    """
    parts: list[dict] = []
    if acknowledge and outcome.kind == "ask":
        parts.append({"slot": "ack", "origin": "code", "text": ACKNOWLEDGE})
    for item in obs.reply_material:
        text = (item.get("text") or "").strip()
        if text:
            parts.append({"slot": item.get("kind") or "answer", "origin": "model", "text": text})

    if outcome.reason_code == "HOLD_PAUSE":
        parts.append({"slot": "convey", "origin": "code",
                      "text": "Тепло признай просьбу отложить и предложи вернуться, когда будет удобно. "
                              "Вопрос НЕ задавай."})
    else:
        convey = _convey_slot(obs, state, outcome.focus)
        if convey:
            parts.append({"slot": "convey", "origin": "code", "text": convey})
        if outcome.focus:
            slot = _ask_slot(outcome.focus, state, ctx, ask_format)
            if slot:
                if _needs_questions_intro(outcome.focus, state):
                    parts.append({"slot": "intro", "origin": "code", "text": QUESTIONS_INTRO})
                parts.append({"slot": "ask", "origin": "code", "text": slot})

    return " ".join(p["text"] for p in parts).strip(), parts


# ── главное ───────────────────────────────────────────────────────────────────

def decide(state: dict, observation: Observation, message: str, ctx: DecideContext, *,
           dialogue_closed: bool = False, analyzer_failed: bool = False) -> TurnPlan:
    """Один ход. Вход не мутируется."""
    progress_before = state_model.progress_signature(state)
    state_before = state
    no_progress_before = state.get("no_progress", 0)

    salary = _resolve_salary(observation, message, ctx)

    working = state_model.apply_updates(state, _updates_from_observation(observation, state))
    working = _apply_formats(working, observation)
    if salary.verdict == "fits":
        # Ниже минимума вилки отказом не является никогда — молча закрываем и идём дальше.
        working = state_model.apply_updates(working, [{"key": "salary", "value": "closed"}])
        salary.effect = "closed"

    working, charged, counter_before = _charge_counter(observation, working)

    focus = _focus_of(working)
    ask_format = _format_to_ask(working, focus)
    reask = _reask_candidate(state_before, working, focus, ask_format)

    frame = Frame(
        state=working, obs=observation, ctx=ctx, salary=salary,
        dialogue_closed=dialogue_closed, analyzer_failed=analyzer_failed,
        charged_event=charged, counter_before=counter_before,
        reask_candidate=reask,
        reask_fired=(reask[0], reask[1]) if reask and reask[1].fires(reask[2]) else None,
        focus=focus, ask_format=ask_format,
        agenda_complete=state_model.is_complete(working),
        no_progress_now=_stall_count(working, progress_before, no_progress_before),
    )

    outcome, rule_name = _walk(frame, start=0)

    # `REFUSE_AND_ADVANCE`: пункт помечается refused, фокус едет дальше, ход дорешивается тем же
    # проходом по таблице. Модель в этом не участвует — второго вызова Наблюдателя не существует.
    refused_now = outcome.reason_code == REFUSE_AND_ADVANCE
    if refused_now:
        frame.state = state_model.apply_updates(frame.state, [{"key": outcome.refuse_key, "value": "refused"}])
        frame.state["last_asking"] = None
        frame.focus = _focus_of(frame.state)
        frame.ask_format = _format_to_ask(frame.state, frame.focus)
        frame.agenda_complete = state_model.is_complete(frame.state)
        frame.reask_fired = None
        frame.reask_candidate = None
        frame.no_progress_now = _stall_count(frame.state, progress_before, no_progress_before)
        outcome, rule_name = _walk(frame, start=RESUME_AFTER_REFUSE)

    new_state = _apply_reask(frame.state, outcome, frame)
    new_state["no_progress"] = _stall_count(new_state, progress_before, no_progress_before)

    # Благодарить есть за что, только если пункт закрылся ОТВЕТОМ кандидата: «спасибо» за отказ
    # отвечать звучало бы издевательски.
    answered_now = (not refused_now
                    and state_model.progress_signature(state_before) != state_model.progress_signature(new_state))
    instruction, parts = _build_instruction(observation, outcome, new_state, ctx, frame.ask_format,
                                            acknowledge=answered_now)

    if any(p.get("slot") == "intro" for p in parts):
        new_state["questions_intro_sent"] = True

    if outcome.kind == "ask":
        new_state["last_asked"] = instruction
        new_state["last_asking"] = outcome.focus
        # Пишем ровно то, о чём спросили: пустой `format_asked` на ходе без вопроса про формат —
        # это «короткое да к формату не относится», и модель обязана видеть именно его.
        new_state["format_asked"] = frame.ask_format if outcome.focus in ("format", "field_work") else None
        end = False
    elif outcome.kind == "silent":
        end = True
    else:
        # Терминальность — ИЗ РЕЕСТРА, а не по префиксу ключа.
        end = reasons.is_terminal(outcome.reason_code)

    if salary.verdict == "ko":
        salary.effect = "ko_forced" if outcome.reason_code == "KO_SALARY" else "ko_overridden_by_signal"

    return TurnPlan(
        rule=rule_name,
        reason_code=outcome.reason_code,
        kind=outcome.kind,
        end=end,
        focus=outcome.focus,
        instruction=instruction,
        instruction_parts=parts,
        state_next=new_state,
        audit={
            "salary": {
                "status": salary.status,
                "verdict": salary.verdict,
                "effect": salary.effect,
                "gates_failed": salary.gates_failed,
                "normalized": ({"min": salary.value.min, "max": salary.value.max}
                               if salary.value is not None else None),
                "applied": salary.value.applied if salary.value is not None else None,
                "rules_version": salary_mod.SALARY_RULES_VERSION,
            },
            "signals_seen": observation.codes(),
            "signals_skipped": frame.skipped_signals,
            "signal_dropped": observation.dropped,
            "formats": dict(new_state.get("formats") or {}),
            "format_asked": new_state.get("format_asked"),
            "counter_charged": ({"key": charged, "before": counter_before,
                                 "after": new_state.get("counters", {}).get(charged, 0)}
                                if charged else None),
            "config_hash": config_digest(),
        },
    )


def _walk(frame: Frame, *, start: int) -> tuple[Outcome, str]:
    """Проход по таблице: первое правило, вернувшее исход, выигрывает."""
    for rule in RULES[start:]:
        outcome = rule.when(frame)
        if outcome is not None:
            return outcome, rule.name
    return Outcome("REPLY_FALLBACK", "script"), "R0.no_rule"


def _apply_reask(state: dict, outcome: Outcome, frame: Frame) -> dict:
    """Инкремент бюджета переспроса — только когда код реально задал тот же вопрос снова."""
    if outcome.kind != "ask" or not outcome.focus or not frame.reask_candidate:
        return state
    kind, budget, before = frame.reask_candidate
    if frame.reask_fired is not None:
        return state  # кап уже сработал, правило R8 забрало ход
    new_state = state_model.apply_updates(state, [])
    if kind == "question":
        for question in new_state.get("questions", []):
            if question["key"] == outcome.focus:
                question["reask_count"] = budget.next_value(before)
    else:
        new_state[budget.counter] = budget.next_value(before)
    return new_state

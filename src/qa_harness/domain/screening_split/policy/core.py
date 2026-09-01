"""`decide()` — чистое ядро хода. Ни сети, ни стора, ни вызовов LLM.

    decide(state, observation, message, ctx) -> TurnPlan

Порядок внутри хода — тот самый, ради которого затевается перестройка: код СЧИТАЕТ раньше, чем
выбирается действие, поэтому решения, с которым можно не согласиться, не существует и перерешивать
нечего.

    1. зарплата: claim → статус → пересчёт → вилка → односторонние гейты → вердикт
    2. факты и ответы: монотонно в состояние
    3. счётчики: ровно один раз за ход
    4. фокус и бюджеты переспросов
    5. таблица правил R1–R11: первое сработавшее выигрывает
    6. сборка инструкции: черновик модели + вопрос из шаблона кода

Чего в этом файле НЕТ и быть не может (порт `..engine` держит всё это ради компенсации порядка):
`_gate_salary_update`, `_assumes_salary_closed`, `_release_money_stop`, `_SALARY_REWIND_NOTE`, флаг
`_forced` и три ветки второго вызова Аналитика.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .. import salary as salary_mod
from .. import state as state_model
from . import gates
from .budgets import EVENT_BUDGETS, REASK_BUDGETS, STALL_BUDGET, config_digest
from .geo import same_city
from . import reasons
from .observation import SIGNAL_TO_COUNTER, Observation
from .rules import RESUME_AFTER_REFUSE, RULES, Outcome

REFUSE_AND_ADVANCE = "REFUSE_AND_ADVANCE"


# ── вход ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecideContext:
    """Всё о вакансии, что нужно КОДУ. Вилка сюда приходит типизированной и Аналитику не показывается
    (принцип П4): промпт использовал её только в запретах, а считать ею прямо запрещён."""

    band_min: Optional[int] = None
    band_max: Optional[int] = None
    band_currency: str = "RUB"
    work_format: str = ""
    location: str = ""
    contact_source: str = ""
    has_geo_restriction: bool = False


@dataclass
class SalaryResult:
    """Что код сделал с суммой на этом ходе. Идёт в аудит целиком: у отсеянного кандидата сводки
    для рекрутера не создаётся, и без этой записи отказ по деньгам невоспроизводим."""

    status: str = salary_mod.ABSENT
    value: Any = None
    verdict: Optional[str] = None
    effect: Optional[str] = None
    gates_failed: list[str] = field(default_factory=list)


@dataclass
class TurnPlan:
    """Выход ядра — общий контракт трёх портов. Канал берёт отсюда текст и `end`, всё остальное
    пишет в трассу."""

    rule: str
    reason_code: str
    kind: str                       # "script" | "ask" | "silent"
    end: bool
    focus: Optional[str] = None
    instruction: str = ""
    instruction_parts: list[dict] = field(default_factory=list)
    state_next: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)


# ── внутренняя рамка хода (её видят предикаты правил) ─────────────────────────

@dataclass
class Frame:
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
    agenda_complete: bool = False
    # Значение счётчика ЗА ЭТОТ ХОД: уже с учётом прогресса, случившегося в нём. Именно его читает
    # правило стоп-крана. Брать значение «до хода» нельзя — ход, в котором кандидат наконец назвал
    # город и сумму, обнуляет счётчик, а не добивает его (прогон 28.08, сценарий 64).
    no_progress_now: int = 0
    skipped_signals: list = field(default_factory=list)

    @property
    def counters(self) -> dict:
        return self.state.get("counters", {})


# ── проекция состояния для промпта ────────────────────────────────────────────

def state_for_prompt(state: dict) -> dict:
    """То, что видит Наблюдатель. Служебные поля НЕ сериализуются.

    Сегодня в промпт уходит state целиком, и там же приходится держать абзац «поля `last_asking`,
    `salary_reasks`, `format_reasks`, `reask_count`, `no_progress` ведёт КОД, решения по ним не
    принимай, в ответе не упоминай» (`screening_analyzer/v2/system.md:38`). Абзац существует ровно
    потому, что мы сами показываем модели то, что просим игнорировать. Не показываем — не надо и
    просить.

    `counters` тоже уходят: ветвление «первый раз объясняем про вилку, дальше просто переспрашиваем»
    и «первая пауза / вторая» теперь делает код при сборке вопроса, а не модель по счётчику.
    """
    return {
        "salary": state.get("salary"),
        "format_check": state.get("format_check"),
        "candidate_city": state.get("candidate_city"),
        "questions": [{"key": q.get("key"), "text": q.get("text"), "status": q.get("status")}
                      for q in state.get("questions", [])],
        "last_asked": state.get("last_asked"),
    }


# ── шаг 1: зарплата ───────────────────────────────────────────────────────────

def _resolve_salary(obs: Observation, message: str, ctx: DecideContext) -> SalaryResult:
    """Полный зарплатный разбор ДО маршрутизации. Возвращает вердикт, а не решение."""
    res = SalaryResult()
    claim = obs.salary_claim
    if not isinstance(claim, dict):
        return res

    res.status = salary_mod.claim_status(claim, message)
    if res.status != salary_mod.ACTIONABLE:
        return res

    value = salary_mod.normalize(claim)
    if value is None:
        # Пересчёт дал противоречие (обе границы пусты либо min > max) — это уточняющий вопрос,
        # а не отсев. Порт `..engine:234-238`.
        res.status = salary_mod.UNUSABLE
        return res

    res.value = value
    band_max = _band_max_rub(ctx)
    res.verdict = salary_mod.compare_with_band(value, ctx.band_min, band_max)

    if res.verdict == "ko":
        failed = gates.downgrade_ko(claim)
        if failed:
            # Односторонний гейт: отказ понижается до уточняющего вопроса, `fits` неприкосновенен.
            res.gates_failed = failed
            res.status = salary_mod.UNUSABLE
            res.verdict = None
            res.value = None
    return res


def _band_max_rub(ctx: DecideContext) -> Optional[int]:
    """Вилка приводится к рублям ТЕМИ ЖЕ данными пересчёта, что и сумма кандидата.

    Чинит P11: в hh вилка приходит из `hh_vacancy_data["salary"]`, где есть поле `currency`, но его
    сегодня никто не читает — вилка в тенге сравнивается как рублёвая.
    """
    if not ctx.band_max:
        return None
    rate = salary_mod.RATE_TO_RUB.get(ctx.band_currency or "RUB")
    if not rate or rate == 1:
        return ctx.band_max
    return round(ctx.band_max * rate)


# ── шаг 2: факты и ответы → состояние ─────────────────────────────────────────

def _updates_from_observation(obs: Observation, state: dict) -> list[dict]:
    """Наблюдение → монотонная дельта состояния. Ключа `salary` здесь нет и быть не может: пункт
    закрывает только код и только после сравнения с вилкой (`_gate_salary_update` не нужен —
    закрыть зарплату, не сравнив её, стало невыразимо в контракте)."""
    updates: list[dict] = []

    city = obs.facts.get("candidate_city")
    if isinstance(city, str) and city.strip():
        updates.append({"key": "candidate_city", "value": city.strip()})
        updates.append({"key": "city_check", "value": "closed"})

    relocation = obs.facts.get("relocation_ready")
    if relocation in ("yes", "no"):
        updates.append({"key": "relocation_ready", "value": relocation})

    if state.get("format_check") == "pending" and obs.facts.get("format_ready") == "yes":
        updates.append({"key": "format_check", "value": "closed"})

    # Согласие переехать закрывает ПУНКТ ПРО ПЕРЕЕЗД и только его (Р18). Раньше оно закрывало
    # `format_check`, и кандидат «в офис не готов, но перееду» проходил проверку формата: переезд
    # отвечал на вопрос, которого ему не задавали. Формат и локация — разные требования.
    if relocation == "yes" and state.get("relocation_check") == "pending":
        updates.append({"key": "relocation_check", "value": "closed"})

    known = {q["key"] for q in state.get("questions", [])}
    for answer in obs.answers:
        key = str(answer.get("key") or "").strip()
        if key in known and answer.get("substantive"):
            updates.append({"key": key, "value": "closed"})
    return updates


def _open_relocation_check(state: dict, ctx: "DecideContext") -> dict:
    """Переводит `relocation_check` в `pending`, когда вопрос про переезд стал осмысленным (Р18).

    Условия все четыре: формат подтверждён кандидатом · город известен · у вакансии есть локация ·
    город не совпадает с локацией. На удалённой вакансии `format_check` остаётся `n/a`, поэтому пункт
    не откроется никогда — переезжать некуда, локация там важна только для гео-ограничения (R5).
    """
    if state.get("relocation_check") != "n/a":
        return state
    if state.get("format_check") != "closed":
        return state
    city = state.get("candidate_city")
    if not city or not ctx.location or same_city(city, ctx.location):
        return state
    return state_model.apply_updates(state, [{"key": "relocation_check", "value": "pending"}])


def _stall_count(state: dict, progress_before: tuple, before: int) -> int:
    """Сколько ходов подряд диалог не собрал ничего нового, ВКЛЮЧАЯ текущий.

    Ход, принёсший новый факт, счётчик обнуляет — даже если до него их накопилось три.
    """
    if state_model.progress_signature(state) == progress_before:
        return before + 1
    return 0


# ── шаг 3: счётчики ───────────────────────────────────────────────────────────

def _charge_counter(obs: Observation, state: dict) -> tuple[dict, Optional[str], int]:
    """Начисляет РОВНО ОДИН счётчик за ход и возвращает (state, ключ, значение до хода).

    Отложенный инкремент (`..engine:322-326`) больше не нужен: вызов Аналитика один, и «увидеть
    счётчик одинаковым» второму вызову неоткуда.

    `demand` приходит не сигналом, а флагом `persistent`: в таблице приоритета триггеров его нет
    вовсе, это надстройка над другим триггером. Приоритет начисления — сигнал важнее флага: сегодня
    модель на настойчивом «вы бот?» ставит `event:"bot_check"`, а не `demand`.
    """
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
    """Первый незакрытый пункт повестки: зарплата → город → формат → переезд → доп-вопросы (Р18).

    Порядок сегодня описан прозой в промпте (`system.md:307-316`) вместе с исключением «пункт закроет
    код»; исключение исчезает, потому что к моменту вычисления фокуса код уже посчитал.

    Город стоит ПЕРЕД форматом, но отсев по формату от этого не зависит: правило R6 срабатывает на
    отказе от формата в любой момент, даже если город ещё не назван.
    """
    if state.get("salary") != "closed":
        return "salary"
    if state.get("city_check") == "pending":
        return "city"
    if state.get("format_check") == "pending":
        return "format"
    if state.get("relocation_check") == "pending":
        return "relocation"
    pending = state_model.pending_questions(state)
    return pending[0]["key"] if pending else None


def _reask_candidate(state_before: dict, state_now: dict, focus: Optional[str]) -> Optional[tuple]:
    """(вид бюджета, бюджет, значение счётчика до хода) — если код в прошлый ход спрашивал ТО ЖЕ.

    Сравнивается `state.last_asking`, который теперь пишет КОД по фактически выданному вопросу, а не
    модель по самоотчёту. Ход без вопроса (ответили кандидату) кап не жжёт — как и сегодня при
    `asking: null`.
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
        return "format", REASK_BUDGETS["format"], state_now.get("format_reasks", 0)
    if focus == "city":
        if state_now.get("city_check") != "pending":
            return None
        return "city", REASK_BUDGETS["city"], state_now.get("city_reasks", 0)
    if focus == "relocation":
        if state_now.get("relocation_check") != "pending":
            return None
        return "relocation", REASK_BUDGETS["relocation"], state_now.get("relocation_reasks", 0)
    question = next((q for q in state_now.get("questions", []) if q["key"] == focus), None)
    if question is None or question.get("status") != "pending":
        return None
    return "question", REASK_BUDGETS["question"], question.get("reask_count", 0)


# ── шаг 6: сборка инструкции ──────────────────────────────────────────────────

# ── эскалация переспроса ─────────────────────────────────────────────────────────────────────
# Один пункт спрашивается до трёх раз (REASK_BUDGETS), но раньше формулировка от НОМЕРА повтора не
# зависела вовсе: у зарплаты флаг `salary_info` взводился один раз и держался, у формата варианта
# «переспроси» не было, у доп-вопроса текст не менялся никогда. Кандидат получал дословно одно и то
# же по три раза подряд, а разнообразие оставалось целиком на Интервьюере (через `last_sent`).
#
# Ступеней две: 1-й переспрос объясняет ЗАЧЕМ нужен ответ, 2-й предупреждает о последствии. Третьей
# нет — там срабатывает кап: STOP по зарплате/формату, `refused` по доп-вопросу.
#
# NB: у зарплаты снят прежний запрет «ничего не объясняй». Он и появился как замена объяснению,
# которого не было; инвариант сценария 29 (`expect_last_instruction_lacks: раскрыва`) остаётся
# выполненным — про «раскрытие вилки» здесь по-прежнему ни слова.
_SALARY_WHY = {
    True:  "Объясни: назвать вилку по позиции мы не можем, поэтому и нужен его ориентир.",
    False: "Объясни: сумма нужна, чтобы сверить ожидания с бюджетом позиции.",
}

# Город спрашивается всегда, в том числе на удалённых вакансиях (Р18), поэтому вопрос «а зачем вам
# мой город» законен и ответ на него обязан быть правдивым. Ни вилку, ни гео-ограничение вакансии
# объяснение не раскрывает: причины настоящие и нейтральные.
_CITY_WHY = ("Объясни: город нужен, чтобы понимать часовой пояс для созвонов и возможность "
             "оформления.")

# Кандидат назвал сумму/город/ответил на вопрос — и тут же получал следующее требование без единого
# слова признания. Благодарность поручается КОДОМ, а не оставляется на усмотрение Интервьюера: судья
# слоя B сверяет текст с инструкцией и незаказанную вежливость засчитал бы как отсебятину.
# Пересказывать ответ при этом нельзя — запрет на «Зафиксируй: кандидат находится в …» остаётся
# (инцидент 2026-08-17), поэтому он повторён прямо здесь.
# Про «не повторяйся» здесь сознательно НИ СЛОВА: Интервьюер и так получает своё предыдущее
# сообщение отдельным полем с пометкой «не повторяй его дословно» (`PolicyInterviewer._build_turn`).
# Дублировать запрет в инструкции — значит удлинять её на каждом ходе ради того, что уже сказано.
ACKNOWLEDGE = "Коротко поблагодари кандидата за ответ. Пересказывать сказанное им не нужно."

_LAST_CALL = {
    "salary": "Предупреди: без ориентира по сумме продолжить скрининг не сможем.",
    "format": "Предупреди: без ответа про формат продолжить скрининг не сможем.",
    "city": "Предупреди: без ответа на этот вопрос продолжить скрининг не получится.",
    # Доп-вопрос диалог не рубит — кап помечает его `refused` и едет дальше, об этом и предупреждаем.
    "question": ("Предупреди: если ответа не будет, отметим вопрос как оставшийся без ответа "
                 "и перейдём к следующему."),
}


def _reasks_of(focus: str, state: dict) -> int:
    """Сколько раз этот пункт УЖЕ переспросили. Читается после `_apply_reask`, поэтому 0 — первый
    вопрос, 1 — первый переспрос."""
    if focus == "salary":
        return int(state.get("salary_reasks", 0))
    if focus == "format":
        return int(state.get("format_reasks", 0))
    if focus == "city":
        return int(state.get("city_reasks", 0))
    if focus == "relocation":
        return int(state.get("relocation_reasks", 0))
    question = next((q for q in state.get("questions", []) if q["key"] == focus), None)
    return int((question or {}).get("reask_count", 0))


def _ask_slot(focus: str, state: dict, ctx: DecideContext) -> str:
    """Вопрос текущего фокуса — текстом КОДА, со вставленными значениями.

    Именно здесь «директива без значения» перестаёт быть возможной: код подставляет значение сам,
    сочинять Интервьюеру нечего. Это структурная замена запрету в промпте (`system.md:87`), который
    воспроизводился даже в максимально прямой формулировке (прод-инцидент 2026-08-17, Баг A).
    """
    reasks = _reasks_of(focus, state)
    if focus == "salary":
        if reasks == 0:
            return ("Спроси, на какую сумму на руки в месяц ориентируется кандидат. "
                    "Конкретных чисел вилки не называй.")
        # Почему переспрашиваем — зависит от того, ТРЕБОВАЛ ли кандидат нашу вилку (`salary_info`)
        # или просто не назвал сумму. Возражения разные, общий ответ звучал бы мимо обоих.
        asked_band = state.get("counters", {}).get("salary_info", 0) >= 1
        parts = ["Переспроси, на какую сумму на руки в месяц ориентируется кандидат.",
                 _SALARY_WHY[asked_band]]
        if reasks >= 2:
            parts.append(_LAST_CALL["salary"])
        parts.append("Конкретных чисел вилки не называй.")
        return " ".join(parts)
    if focus == "city":
        # Отдельный пункт повестки, а не часть вопроса про формат (Р18). Ступени: нейтральный вопрос →
        # объяснение зачем → предупреждение. Кап (3-й переспрос) завершает диалог: без города нет ни
        # гео-отсева, ни понимания, как оформлять.
        parts = []
        if reasks >= 1:
            parts.append(_CITY_WHY)
        if reasks >= 2:
            parts.append(_LAST_CALL["city"])
        parts.append(("Спроси, " if reasks == 0 else "Переспроси, ")
                     + "в каком городе кандидат сейчас находится.")
        return " ".join(parts)
    if focus == "relocation":
        # Пункт открывается только когда формат ПОДТВЕРЖДЁН, а город кандидата не совпадает с
        # локацией вакансии. Поэтому здесь речь именно про место, а не про формат.
        city = ctx.location or ""
        parts = [f"Донеси: работа предполагает присутствие в городе {city}." if city else ""]
        parts = [x for x in parts if x]
        if reasks >= 1:
            parts.append("Объясни: без ответа про локацию мы не сможем двигаться дальше.")
        if reasks >= 2:
            parts.append(_LAST_CALL["format"])
        parts.append(("Спроси, " if reasks == 0 else "Переспроси, ")
                     + "готов ли кандидат переехать в этот город или работать из него.")
        return " ".join(parts)
    if focus == "format":
        city = ctx.location or ""
        wf = (ctx.work_format or "").strip().lower()
        formats = {"office": "работа из офиса", "hybrid": "гибридный формат"}
        human = formats.get(wf, "требуемый формат работы")
        where = f" в городе {city}" if city else ""
        # Только про формат: город спрашивает пункт `city`, переезд — пункт `relocation` (Р18).
        # Прежняя склейка «формат + город + переезд» в одном вопросе и была причиной того, что
        # согласие переехать закрывало проверку формата.
        ask = "удобен ли кандидату такой формат работы."
        # «Обязательное требование» с первого же хода звучало ультиматумом («это обязательное
        # требование вакансии» в ответ на только что названную зарплату). Сначала — нейтральное
        # описание условий и вопрос об удобстве; жёсткость приходит ступенью переспроса, когда
        # кандидат на вопрос не ответил.
        parts = [f"Донеси: вакансия предполагает {human}{where}."]
        if reasks >= 1:
            parts.append("Объясни: это обязательное условие вакансии.")
        if reasks >= 2:
            parts.append(_LAST_CALL["format"])
        parts.append(("Спроси, " if reasks == 0 else "Переспроси, ") + ask)
        return " ".join(parts)
    question = next((q for q in state.get("questions", []) if q["key"] == focus), None)
    if question is None:
        return ""
    # «Дословно» здесь вредно: в настройках вакансии вопросы записаны телеграфно («Сервисы под
    # нагрузкой?»), и Интервьюер обязан развернуть их в человеческую фразу. Прогон 28.08, сценарий
    # 25: судья засчитал вежливое разворачивание как нарушение требования «дословно».
    verb = "Задай" if reasks == 0 else "Повтори"
    text = f"{verb} дополнительный вопрос по теме: «{question['text']}»."
    if reasks >= 2:
        text += " " + _LAST_CALL["question"]
    return text


# Реакции на лёгкие сигналы — текстом КОДА. В v2 их держал промпт (таблица нетерминальных
# триггеров): «мягко отметь, что ответ не по делу», «скажи, что ты внешний рекрутер» и т.п.
# Решение перенесли в код, а реакции сначала не перенесли — и Интервьюер, получив голый вопрос
# поверх эмоциональной реплики, дописывал сочувствие сам. Прогон 28.08: 9 провалов слоя B из 14.
_SIGNAL_CONVEY: dict[str, str] = {
    "bot_check": "Скажи, что ты внешний рекрутер и ведёшь первичный скрининг.",
    "gibberish": "Мягко отметь, что ответ не совсем понятен и не по делу.",
    "answer_aid": ("Мягко попроси отвечать самостоятельно, без ИИ и сторонних помощников — "
                   "важно оценить именно реальный опыт кандидата."),
    "scheduling": ("Откажись назначать созвон или встречу: ты ведёшь только первичный скрининг "
                   "в чате."),
    "pause": ("Скажи, что понимаешь: сейчас неудобно, и что готова вернуться к разговору позже."),
}

# Порядок разбора: первая подходящая реакция и выигрывает. Терминальные сигналы сюда не попадают —
# их забирает таблица правил выше и до сборки инструкции дело не доходит.
_CONVEY_ORDER = ("bot_check", "answer_aid", "gibberish", "scheduling", "pause")


# Вводная перед ПЕРВЫМ доп-вопросом (Б1). До неё разговор шёл про зарплату и формат, и вопрос по
# навыкам прилетал кандидату без всякого предупреждения. Формулировка — ПРОСЬБА ответить, а не
# анонс «дальше будет»: просьба вежливее и точнее описывает, чего мы хотим.
# Последняя фраза остаётся: просьба не должна выродиться в вопрос «готовы ответить?» — согласия
# промпт спрашивать запрещает, да и правило «в инструкции ровно один вопрос» иначе нарушится
# (вопрос про согласие + сам доп-вопрос). Предупреждение «отвечайте без ИИ» сюда сознательно НЕ
# входит: оно уже есть реактивной реакцией `answer_aid` и звучит по факту, а не авансом.
QUESTIONS_INTRO = (
    "Донеси: попроси ответить на несколько дополнительных вопросов от команды — они нужны, чтобы "
    "подтвердить необходимые навыки и квалификацию. Это просьба, а не вопрос о согласии: "
    "«готовы ответить?» не спрашивай."
)


def _needs_questions_intro(focus: str, state: dict) -> bool:
    """Первый доп-вопрос за диалог, и вводную ещё не говорили."""
    if state.get("questions_intro_sent"):
        return False
    return any(q.get("key") == focus for q in state.get("questions", []))


def _convey_slot(obs: Observation, state: dict, focus: Optional[str]) -> str:
    """Что донести кандидату перед вопросом, помимо ответа модели по существу вакансии.

    ВАЖНО: две директивы кода не должны противоречить друг другу. Пояснительный convey про «ответ
    нужен здесь, в чате» рядом с зарплатой и форматом не ставится: у этих пунктов своё объяснение
    приезжает ступенью переспроса (`_SALARY_WHY`, `_LAST_CALL`), и два «объясни» в одной инструкции
    читались бы как соревнование формулировок (прогон 28.08, сценарий 29).
    """
    for code in _CONVEY_ORDER:
        if not obs.has(code):
            continue
        if code == "pause" and state.get("counters", {}).get("pause", 0) >= 2:
            continue  # со 2-й паузы ход забирает HOLD_PAUSE, вопроса там нет вовсе
        return _SIGNAL_CONVEY[code]
    if obs.focus_answered == "deflection" and focus not in ("salary", "format"):
        # Признание причины кандидата — ЧАСТЬ поручения, а не вольность Интервьюера. Судья слоя B
        # сверяет текст с инструкцией: не поручишь рамку — засчитает её как добавленное сочувствие
        # (прогон 28.08, сценарии 24/49).
        return ("Скажи, что понимаешь кандидата, и мягко объясни: ответ нужен именно здесь, в чате, "
                "ссылки на резюме или профиль недостаточно.")
    return ""


def _build_instruction(obs: Observation, outcome: Outcome, state: dict,
                       ctx: DecideContext, *, acknowledge: bool = False) -> tuple[str, list[dict]]:
    """`instruction` = [благодарность] + [черновик модели] + [вопрос из шаблона кода].

    Разделение по источнику сохраняется в `instruction_parts[].origin` — иначе исчезает атрибуция
    трёхслойного гейта: инварианты `expect_instruction_*` должны проверять модельную часть, а не
    сгенерированный кодом вопрос.
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
            slot = _ask_slot(outcome.focus, state, ctx)
            if slot:
                # Вводная кладётся ВПЛОТНУЮ к вопросу и только вместе с ним: отдельного хода на неё
                # нет, иначе кандидат получил бы сообщение без единого вопроса.
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
    if salary.verdict == "fits":
        # Ниже минимума вилки отказом не является никогда — молча закрываем и идём дальше.
        working = state_model.apply_updates(working, [{"key": "salary", "value": "closed"}])
        salary.effect = "closed"

    working, charged, counter_before = _charge_counter(observation, working)
    # Пункт про переезд открывается ПОСЛЕ применения фактов: он зависит и от подтверждённого формата,
    # и от названного города, а оба могут прийти этим же ходом (Р18).
    working = _open_relocation_check(working, ctx)

    focus = _focus_of(working)
    reask = _reask_candidate(state_before, working, focus)

    frame = Frame(
        state=working, obs=observation, ctx=ctx, salary=salary,
        dialogue_closed=dialogue_closed, analyzer_failed=analyzer_failed,
        charged_event=charged, counter_before=counter_before,
        reask_candidate=reask,
        reask_fired=(reask[0], reask[1]) if reask and reask[1].fires(reask[2]) else None,
        focus=focus,
        agenda_complete=state_model.is_complete(working),
        no_progress_now=_stall_count(working, progress_before, no_progress_before),
    )

    outcome, rule_name = _walk(frame, start=0)

    # `REFUSE_AND_ADVANCE` — единственное место, где сегодня уходит ВТОРОЙ вызов Аналитика
    # (`..engine:385-398`). Здесь пункт помечается refused, фокус едет дальше, и ход дорешивается
    # тем же проходом по таблице: модель не участвует.
    refused_now = outcome.reason_code == REFUSE_AND_ADVANCE
    if refused_now:
        frame.state = state_model.apply_updates(frame.state, [{"key": outcome.refuse_key, "value": "refused"}])
        frame.state["last_asking"] = None
        frame.focus = _focus_of(frame.state)
        frame.agenda_complete = state_model.is_complete(frame.state)
        frame.reask_fired = None
        frame.reask_candidate = None
        # Ветка refused меняет состояние, значит и прогресс: пересчитываем до возобновления прохода,
        # как это делает действующий движок (`..engine:402-407` — после ветки, до проверки капа).
        frame.no_progress_now = _stall_count(frame.state, progress_before, no_progress_before)
        outcome, rule_name = _walk(frame, start=RESUME_AFTER_REFUSE)

    # Переспрос засчитывается ТОЛЬКО если код действительно выдал вопрос по тому же фокусу.
    # Терминальное правило выше по таблице бюджет не жжёт — как и сегодня под флагом `_forced`.
    new_state = _apply_reask(frame.state, outcome, frame)

    # Счётчик уже посчитан до правил (и пересчитан после ветки refused) — здесь только фиксируем.
    new_state["no_progress"] = _stall_count(new_state, progress_before, no_progress_before)

    # Благодарить есть за что, только если пункт закрылся ОТВЕТОМ кандидата. Отказ (`refused`) тоже
    # двигает прогресс, но «спасибо» за отказ отвечать звучало бы издевательски, поэтому исключаем.
    answered_now = (not refused_now
                    and state_model.progress_signature(state_before) != state_model.progress_signature(new_state))
    instruction, parts = _build_instruction(observation, outcome, new_state, ctx,
                                            acknowledge=answered_now)

    # Галочку ставит ФАКТ собранной вводной, а не посчитанный фокус. Фокус может быть уже равен `qN`,
    # а ход при этом заберёт правило выше по таблице и вопроса в нём не будет: скрипт про источник
    # контакта (R10), вторая просьба паузы (HOLD_PAUSE), сбой наблюдения (R2) — диалог продолжается,
    # вопрос уезжает на следующий ход. Поставь галочку по фокусу — вводная сгорит и не прозвучит
    # никогда.
    if any(p.get("slot") == "intro" for p in parts):
        new_state["questions_intro_sent"] = True

    if outcome.kind == "ask":
        new_state["last_asked"] = instruction
        new_state["last_asking"] = outcome.focus
        end = False
    elif outcome.kind == "silent":
        end = True
    else:
        # Терминальность — ИЗ РЕЕСТРА, а не по префиксу ключа: опечатка в имени больше не делает
        # новый код молча терминальным (см. `reasons.is_terminal`).
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
    # Недостижимо: R11 всегда возвращает исход. Оставлено, чтобы отсутствие ветки было видно.
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

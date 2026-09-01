"""Таблица правил R1–R11: единственное место, где выбирается действие хода.

Порядок строк — это и есть приоритет. Код идёт сверху вниз и берёт ПЕРВОЕ сработавшее правило.
Ничего похожего на флаг `_forced`, который сегодня протаскивается через четыре блока движка
(`..engine:278, :289, :350, :364`), здесь нет и быть не может: правило либо сработало, либо нет.

Три сегодняшних решения, растворённых в порядке строк (проверяются офлайн-переигрыванием):
    R3 > R4  — неденежное терминальное сильнее отсева по деньгам   (`.._salary_verdict_wins`, `:89-98`)
    R4 > R7  — зарплатный вердикт выше событийных порогов          (`..engine:272-277`)
    R4 > R9  — вердикт `ko` перебивает `FINISH`                    (`_KO_OVERRIDABLE`, `:55-58`)

Правило возвращает `Outcome` или `None` («не моё, идём ниже»). Предикаты — обычные функции, а не
выражения в строке конфига: порядок обязан быть данными, а условия — кодом, который можно отладить.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .budgets import EVENT_BUDGETS, REASK_BUDGETS, STALL_BUDGET
from .observation import TERMINAL_SIGNAL_REASON

# Исходы, которые не являются ключом реестра скриптов: их разворачивает `core`.
REFUSE_AND_ADVANCE = "REFUSE_AND_ADVANCE"


@dataclass
class Outcome:
    """Что правило постановило сделать с ходом."""

    reason_code: str
    kind: str                      # "script" | "ask" | "silent"
    focus: Optional[str] = None    # для kind="ask": 'salary' | 'format' | 'qN' | None (только ответ)
    refuse_key: Optional[str] = None  # для REFUSE_AND_ADVANCE: какой доп-вопрос закрыть как refused


@dataclass
class Rule:
    name: str
    when: Callable[["object"], Optional[Outcome]]
    note: str = ""


# ── R1–R2: рамка ──────────────────────────────────────────────────────────────

def r1_dialogue_closed(f) -> Optional[Outcome]:
    """Диалог уже закрыт — молчим и остаёмся закрытыми, а не переоткрываем (`..engine:210-212`)."""
    if f.dialogue_closed:
        return Outcome("SILENT", "silent")
    return None


def r2_analyzer_failed(f) -> Optional[Outcome]:
    """Наблюдения нет (модель не ответила / контракт нарушен) — фолбэк, диалог НЕ завершаем.

    Счётчики на таком ходе не трогаются: сбой наш, бюджет кандидата за него платить не должен.
    """
    if f.analyzer_failed:
        return Outcome("REPLY_FALLBACK", "script")
    return None


# ── R3: терминальные сигналы, с двумя исключениями ────────────────────────────

def _r3a_blocks(f, code: str) -> bool:
    """Исключение R3a. «Нет опыта» не исполняется, если кандидат этим же ходом ответил по сути.

    Промпт уже требует ровно этого прозой (`system.md:152, :324`): «нет опыта со стеком этой
    вакансии» — валидный ответ по навыкам, вопрос закрывается, скрининг продолжается;
    `STOP_NO_EXPERIENCE` — только при ГЛОБАЛЬНОМ несоответствии профиля. Сегодня применить исключение
    обязана модель в уме, и промах стоит необратимо закрытого диалога.
    """
    if code != "no_experience":
        return False
    if f.obs.focus_answered == "substantive":
        return True
    return any(a.get("substantive") for a in f.obs.answers)


def _r3b_blocks(f, code: str) -> bool:
    """Исключение R3b. «Не интересно» и «критика» не исполняются, если рядом стоит вопрос по делу.

    Класс инцидента §12.12: «это не LLM-направление, есть что-то подходящее?» уходило в
    `STOP_NOT_INTERESTED`. Промпт различает это прозой (`system.md:187`: недовольство тем, что мы мало
    рассказали, — это `company_info`, а не критика). Условие `focus_answered != refusal` оставляет
    прямой отказ терминальным: «не пишите мне больше» завершает диалог, как и сегодня.
    """
    if code not in ("not_interested", "criticism"):
        return False
    if f.obs.focus_answered == "refusal":
        return False
    return f.obs.has("company_info") or f.obs.has("scheduling")


def r3_terminal_signal(f) -> Optional[Outcome]:
    """Верхний по приоритету терминальный сигнал. Порядок — из `TERMINAL_PRIORITY`, а не из порядка,
    в котором модель их перечислила: выбор максимума по таблице теперь делает код (§12.13)."""
    for code in f.obs.terminal_codes():
        if _r3a_blocks(f, code) or _r3b_blocks(f, code):
            f.skipped_signals.append(code)
            continue
        return Outcome(TERMINAL_SIGNAL_REASON[code], "script")
    return None


# ── R4–R6: отсев по объективным критериям вакансии ────────────────────────────

def r4_salary_ko(f) -> Optional[Outcome]:
    """Ожидания выше нашего максимума. Стоит ВЫШЕ порогов и `FINISH`, но НИЖЕ терминальных сигналов.

    Сравнение делается на любом ходе с годным claim, в том числе после закрытия пункта: кандидат,
    поднявший ожидания в середине диалога, должен быть отсеян (порт `..engine:276-277`).
    """
    if f.salary.verdict == "ko":
        return Outcome("KO_SALARY", "script")
    return None


def r5_geo_ko(f) -> Optional[Outcome]:
    """Гео-ограничение вакансии не выполнено. Ограничение берётся ТОЛЬКО из явной формулировки
    контекста: из того, что у вакансии указан город, оно не выводится (`system.md:281`)."""
    if f.ctx.has_geo_restriction and f.obs.facts.get("geo_blocked") is True:
        return Outcome("KO_GEO", "script")
    return None


def r5a_location_ko(f) -> Optional[Outcome]:
    """Кандидат не поедет туда, где нужно присутствовать (Р18).

    Пункт `relocation_check` открывается кодом только когда формат УЖЕ подтверждён, город известен и
    не совпадает с локацией вакансии, — поэтому здесь речь именно про место, а не про формат, и
    отдельных проверок на город тут не нужно.
    """
    if f.state.get("relocation_check") != "pending":
        return None
    if f.state.get("relocation_ready") != "no":
        return None
    return Outcome("KO_LOCATION", "script")


def r6_format_ko(f) -> Optional[Outcome]:
    """Формат работы кандидату не подходит. Точка (Р18).

    Раньше правило требовало вдобавок, чтобы «переезд не спасал», а согласие переехать закрывало
    `format_check` — и кандидат «в офис не готов, но перееду» проверку формата проходил. Формат и
    локация — разные требования: отказ от формата отсевает независимо от города и готовности к
    переезду, а локацию проверяет отдельный пункт повестки (R5a).

    Ключ выводит КОД из формата вакансии и наличия локации — в v2 выбор между
    `KO_FORMAT_OFFICE`/`_HYBRID`/`_NOCITY` делала модель (`system.md:276-277`), хотя это чистая
    функция от контекста.
    """
    if f.state.get("format_check") != "pending":
        return None
    if f.obs.facts.get("format_ready") != "no":
        return None
    if not f.ctx.location:
        return Outcome("KO_FORMAT_NOCITY", "script")
    wf = (f.ctx.work_format or "").strip().lower()
    if wf == "hybrid":
        return Outcome("KO_FORMAT_HYBRID", "script")
    return Outcome("KO_FORMAT_OFFICE", "script")


# ── R7–R8: бюджеты ────────────────────────────────────────────────────────────

def r7_event_threshold(f) -> Optional[Outcome]:
    """Исчерпан порог событийного счётчика. Считается по событию, начисленному ЭТИМ ходом."""
    if not f.charged_event:
        return None
    budget = EVENT_BUDGETS.get(f.charged_event)
    if budget and budget.reason and budget.fires(f.counter_before):
        return Outcome(budget.reason, "script")
    return None


def r8_reask_cap(f) -> Optional[Outcome]:
    """Исчерпан лимит переспросов одного и того же незакрытого пункта.

    Кап тикает, только когда КОД в прошлый ход реально выдал вопрос по этому же фокусу
    (`state.last_asking`). Это и есть смысл, который решение Д1 назвало искомым: «сколько раз мы
    задали этот вопрос», а не «сколько раз модель пометила ход как переспрос».
    """
    if f.reask_fired is None:
        return None
    kind, budget = f.reask_fired
    if kind == "question":
        return Outcome(REFUSE_AND_ADVANCE, "ask", focus=None, refuse_key=f.focus)
    return Outcome(budget.reason, "script")


# ── R9–R11: обычный ход ───────────────────────────────────────────────────────

def r9_agenda_complete(f) -> Optional[Outcome]:
    """Всё собрано — немедленно `FINISH`, без «ещё одного» вопроса (`system.md:345-347`).

    Стоит НИЖЕ R4: если тем же ходом кандидат назвал сумму выше максимума, скрининг успешным не был.

    ПАУЗА ЭТО ПРАВИЛО НЕ УДЕРЖИВАЕТ (решение 28.08.2026). Кандидат отвечает на последний незакрытый
    вопрос и тем же сообщением говорит «буду ждать звонка» — всё собрано, значит финиш. В v2 промпт
    противоречил сам себе (одно место давало паузе приоритет, другое требовало немедленного
    `FINISH`); в v3 приоритетов нет вовсе, старшинство задаёт порядок строк этой таблицы.

    Тем же решением снят вопрос «завершать ли, когда модель тянет»: повестка закрыта — завершает КОД,
    независимо от того, вернула модель `ask` или нет. На корпусе это 2 хода из 241.
    """
    if f.agenda_complete:
        return Outcome("FINISH", "script")
    return None


def r10_nonterminal_signal(f) -> Optional[Outcome]:
    """Нетерминальный сигнал. Часть из них — готовый скрипт, остальные обрабатываются в тексте
    вместе со следующим вопросом (или без него — ветка «только ответили»)."""
    if f.obs.has("contact_source"):
        return Outcome("REPLY_CONTACT_SOURCE", "script")
    if not f.obs.signals:
        return None
    # Пауза со 2-го раза удерживается БЕЗ вопроса (`system.md:177`): вопрос кандидату сейчас не нужен.
    if f.obs.has("pause") and f.counters.get("pause", 0) >= 2:
        return Outcome("HOLD_PAUSE", "ask", focus=None)
    if f.focus is None:
        return Outcome("ANSWER_ONLY", "ask", focus=None)
    return Outcome(f"ASK_{f.focus.upper()}", "ask", focus=f.focus)


def r11_ask_focus(f) -> Optional[Outcome]:
    """По умолчанию: вопрос текущего фокуса. Фокус пуст — значит собирать больше нечего и мы просто
    отвечаем кандидату; `FINISH` до этого уже отработал бы в R9."""
    if f.focus is None:
        return Outcome("ANSWER_ONLY", "ask", focus=None)
    return Outcome(f"ASK_{f.focus.upper()}", "ask", focus=f.focus)


def r9a_stall(f) -> Optional[Outcome]:
    """Универсальный стоп-кран. Стоит последним, потому что это страховка от зацикливания, а не
    решение по существу реплики: любое содержательное правило выше обязано выиграть."""
    if STALL_BUDGET.fires_on_nth is not None and f.no_progress_now >= STALL_BUDGET.fires_on_nth:
        return Outcome("FINISH" if f.agenda_complete else "STOP_PERSISTENT", "script")
    return None


RULES: tuple[Rule, ...] = (
    Rule("R1.dialogue_closed",    r1_dialogue_closed,    "диалог закрыт"),
    Rule("R2.analyzer_failed",    r2_analyzer_failed,    "наблюдения нет"),
    Rule("R3.terminal_signal",    r3_terminal_signal,    "терминальный сигнал, с исключениями R3a/R3b"),
    Rule("R4.salary_ko",          r4_salary_ko,          "ожидания выше максимума вилки"),
    Rule("R5.geo_ko",             r5_geo_ko,             "гео-ограничение вакансии"),
    Rule("R5a.location_ko",       r5a_location_ko,       "присутствие невозможно: переезд отвергнут"),
    Rule("R6.format_ko",          r6_format_ko,          "формат работы не подходит"),
    Rule("R7.event_threshold",    r7_event_threshold,    "порог событийного счётчика"),
    Rule("R8.reask_cap",          r8_reask_cap,          "лимит переспросов"),
    Rule("R9.agenda_complete",    r9_agenda_complete,    "всё собрано"),
    # Стоп-кран стоит сразу после терминальных правил и ПЕРЕД обычным ходом — ровно как сегодня,
    # где он перебивает любое нетерминальное решение, но не трогает терминальное (`..engine:408`).
    Rule("R9a.stall",             r9a_stall,             "страховка от зацикливания"),
    Rule("R10.nonterminal_signal", r10_nonterminal_signal, "нетерминальный сигнал"),
    Rule("R11.ask_focus",         r11_ask_focus,         "вопрос текущего фокуса"),
)

# Индекс, с которого возобновляется проход после `REFUSE_AND_ADVANCE`: пункт помечен `refused`,
# фокус уехал, и дальше ход разыгрывается как обычный — но повторно жечь бюджеты он не должен.
RESUME_AFTER_REFUSE = next(i for i, r in enumerate(RULES) if r.name == "R9.agenda_complete")

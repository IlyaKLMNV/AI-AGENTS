"""Таблица правил — HH-канал. Порядок строк и есть приоритет: первое сработавшее выигрывает.

Общие с TG строки импортируются как есть (R1, R2, R3 с исключениями R3a/R3b, R4, R8, R9, R11) —
их предикаты канальных полей не читают. Канальными остаются четыре:

    R5   `KO_LOCATION_GEO`  вместо `KO_GEO`;
    R5a  `KO_LOCATION`      отказ про МЕСТО нахождения (в TG такого ключа нет вовсе);
    R6   `KO_FORMAT`        один ключ вместо `KO_FORMAT_OFFICE/_HYBRID/_NOCITY`, и срабатывает он по
                            мультиформату — «отказался от ВСЕХ допустимых», а не «не готов к формату»;
    R10  без ветки `contact_source` — в канале нет ни события, ни скрипта.

Строки R7 и R9a переопределены не по смыслу, а ради источника: они читают hh-таблицы бюджетов
(`.budgets`), где нет `contact_source` и есть `field_work`.

Прежние решения, растворённые в порядке строк, сохраняются: R3 > R4 (неденежное терминальное сильнее
отсева по деньгам) · R4 > R7 (зарплатный вердикт выше порогов) · R4 > R9 (`ko` перебивает `FINISH`).
"""

from typing import Optional

from qa_harness.domain.screening_split.policy.rules import (
    Outcome,
    Rule,
    r1_dialogue_closed,
    r2_analyzer_failed,
    r3_terminal_signal,
    r4_salary_ko,
    r8_reask_cap,
    r9_agenda_complete,
    r11_ask_focus,
)


from . import formats
from .budgets import EVENT_BUDGETS, STALL_BUDGET

REFUSE_AND_ADVANCE = "REFUSE_AND_ADVANCE"


# ── R5–R6: отсев по локации и формату ─────────────────────────────────────────

def r5_geo_ko(f) -> Optional[Outcome]:
    """Жёсткое гео-ограничение вакансии нарушено.

    Двойное совпадение: ограничение ЕСТЬ в контексте (код, `context.has_geo_restriction`) И
    Наблюдатель увидел нарушение в реплике. Это рудимент канала — Б3 в плане: единой политики по
    кандидатам за границей пока нет, и порт её не изобретает.
    """
    if f.ctx.has_geo_restriction and f.obs.facts.get("geo_blocked") is True:
        return Outcome("KO_LOCATION_GEO", "script")
    return None


def r5a_location_ko(f) -> Optional[Outcome]:
    """Кандидат не поедет туда, где нужно присутствовать (Р18).

    Пункт `relocation_check` открывается кодом только когда присутственный или разъездной формат УЖЕ
    подтверждён, город известен и не совпадает с локацией вакансии, — поэтому проверять здесь больше
    нечего, кроме самого отказа. Прежняя версия правила требовала `format_check == "pending"` и
    поэтому не срабатывала никогда, если кандидат отказывался от формата и от переезда одной
    репликой: `format_check` к моменту таблицы уже был закрыт, и ход забирало R6 с ключом
    `KO_FORMAT` (живой прогон 20260831_215941, кейс E).
    """
    if f.state.get("relocation_check") != "pending":
        return None
    if f.state.get("relocation_ready") != "no":
        return None
    return Outcome("KO_LOCATION", "script")


def r6_format_ko(f) -> Optional[Outcome]:
    """Отказался от ВСЕХ допустимых форматов вакансии. Точка (Р18).

    Мультиформат целиком в этой строке: пока среди допустимых есть формат, о котором кандидат не
    высказался, правило молчит и ход уходит в вопрос про этот формат (`core._ask_slot`). Отказ от
    разъездного формата при другом подтверждённом сюда не попадает — и не должен.

    От города и готовности к переезду правило не зависит: формат — самостоятельное требование, а
    локацию проверяет R5a на своём пункте повестки.
    """
    if formats.refused_all(f.state):
        return Outcome("KO_FORMAT", "script")
    return None


# ── R7, R9a: бюджеты (те же строки, что в TG, но из hh-таблиц) ────────────────

def r7_event_threshold(f) -> Optional[Outcome]:
    if not f.charged_event:
        return None
    budget = EVENT_BUDGETS.get(f.charged_event)
    if budget and budget.reason and budget.fires(f.counter_before):
        return Outcome(budget.reason, "script")
    return None


def r9a_stall(f) -> Optional[Outcome]:
    """Универсальный стоп-кран: страховка от зацикливания, а не решение по существу реплики."""
    if STALL_BUDGET.fires_on_nth is not None and f.no_progress_now >= STALL_BUDGET.fires_on_nth:
        return Outcome("FINISH" if f.agenda_complete else "STOP_PERSISTENT", "script")
    return None


# ── R10: нетерминальный сигнал ────────────────────────────────────────────────

def r10_nonterminal_signal(f) -> Optional[Outcome]:
    """Сигнал обрабатывается в тексте вместе со следующим вопросом (или без него).

    Готового скрипта здесь нет ни у одного сигнала: единственным в TG был `REPLY_CONTACT_SOURCE`,
    которого в канале не существует.
    """
    if not f.obs.signals:
        return None
    if f.obs.has("pause") and f.counters.get("pause", 0) >= 2:
        return Outcome("HOLD_PAUSE", "ask", focus=None)
    if f.focus is None:
        return Outcome("ANSWER_ONLY", "ask", focus=None)
    return Outcome(f"ASK_{f.focus.upper()}", "ask", focus=f.focus)


RULES: tuple[Rule, ...] = (
    Rule("R1.dialogue_closed",     r1_dialogue_closed,     "диалог закрыт"),
    Rule("R2.analyzer_failed",     r2_analyzer_failed,     "наблюдения нет"),
    Rule("R3.terminal_signal",     r3_terminal_signal,     "терминальный сигнал, с исключениями R3a/R3b"),
    Rule("R4.salary_ko",           r4_salary_ko,           "ожидания выше максимума вилки"),
    Rule("R5.geo_ko",              r5_geo_ko,              "гео-ограничение вакансии"),
    Rule("R5a.location_ko",        r5a_location_ko,        "другой город и переезд исключён"),
    Rule("R6.format_ko",           r6_format_ko,           "отказ от всех допустимых форматов"),
    Rule("R7.event_threshold",     r7_event_threshold,     "порог событийного счётчика"),
    Rule("R8.reask_cap",           r8_reask_cap,           "лимит переспросов"),
    Rule("R9.agenda_complete",     r9_agenda_complete,     "всё собрано"),
    Rule("R9a.stall",              r9a_stall,              "страховка от зацикливания"),
    Rule("R10.nonterminal_signal", r10_nonterminal_signal, "нетерминальный сигнал"),
    Rule("R11.ask_focus",          r11_ask_focus,          "вопрос текущего фокуса"),
)

RESUME_AFTER_REFUSE = next(i for i, r in enumerate(RULES) if r.name == "R9.agenda_complete")

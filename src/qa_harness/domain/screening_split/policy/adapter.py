"""Переходник `Decision` → `Observation` для переигрывания ЗАПИСАННЫХ трасс.

Нужен ровно для одного: проверить новое ядро на 2362 сохранённых ходах, не потратив ни одного токена
и не трогая прод. В самой архитектуре переходника нет — там Аналитик сразу возвращает `Observation`.

ЧЕСТНАЯ ГРАНИЦА ПРИМЕНИМОСТИ. Старый `Decision` несёт РЕЗУЛЬТАТ выбора, а не то, из чего он выбирался,
поэтому восстановить наблюдение целиком нельзя:

  восстанавливается     `salary_claim` (1:1), `event` → счётный сигнал, терминальный `script_key` →
                        терминальный сигнал, `updates` → факты и ответы;
  НЕ восстанавливается  сигналы, которые модель УВИДЕЛА, но не выбрала (старый контракт отдаёт ровно
                        один), `focus_answered`, `reply_material`.

Отсюда прямое следствие: переигрывание проверяет **арифметику и порядок кода** — зарплатный блок,
счётчики, пороги, лимиты переспросов, монотонность состояния, определение завершения. Оно НЕ проверяет
правила R3a/R3b и качество наблюдения: для этого нужен новый промпт, а не старые трассы. Заявлять по
итогам переигрывания, что «архитектура проверена целиком», нельзя.
"""

from typing import Any, Optional

from .observation import TERMINAL_SIGNAL_REASON, Observation, Signal

# `script_key` → сигнал: обратное отображение таблицы терминальных.
_REASON_TO_SIGNAL: dict[str, str] = {v: k for k, v in TERMINAL_SIGNAL_REASON.items()}

# `event` → сигнал. `demand` сигналом не является (см. observation.SIGNAL_TO_COUNTER) — он
# восстанавливается во флаг `persistent`.
_EVENT_TO_SIGNAL: dict[str, str] = {
    "bot_check": "bot_check",
    "gibberish": "gibberish",
    "salary_info": "salary_info",
    "contact_source": "contact_source",
    "pause": "pause",
}

# Ключи, которые в старом контракте мог поставить КОД, а не модель. Наблюдением они не являются:
# это уже исход, и подавать его в новое ядро как вход значило бы проверять ядро само на себе.
CODE_FORCED_KEYS = frozenset({
    "KO_SALARY", "STOP_PERSISTENT", "STOP_GIBBERISH_REPEAT", "STOP_BOT_REPEAT",
    "STOP_PAUSE", "STOP_SALARY_DEMAND", "REPLY_FALLBACK",
})


def is_replayable(decision: dict) -> tuple[bool, str]:
    """Можно ли считать этот `Decision` сырым выходом модели.

    Ход, где решение подменил код (`source` проставлен движком), для переигрывания непригоден:
    исходного наблюдения в трассе не осталось. Такие ходы честнее исключить, чем восстанавливать
    догадкой.
    """
    if not isinstance(decision, dict) or not decision:
        return False, "решения нет"
    source = decision.get("source")
    if source:
        return False, f"решение подменено кодом ({source})"
    if decision.get("next_action") not in ("ask", "script"):
        return False, f"next_action={decision.get('next_action')!r}"
    # Трасса старше зарплатного контракта: поля нет вовсе, и «кандидат про деньги не говорил»
    # неотличимо от «говорил, но это не записали». Пункт зарплаты в таких прогонах закрывала МОДЕЛЬ
    # через `updates`, чего новый контракт не допускает, — сравнивать состояния бессмысленно.
    if "salary_claim" not in decision:
        return False, "трасса до зарплатного контракта"
    key = decision.get("script_key")
    if decision.get("next_action") == "script" and key in CODE_FORCED_KEYS:
        return False, f"код-форсимый ключ {key}"
    return True, ""


def decision_to_observation(decision: dict, message: str) -> Observation:
    """Восстановленное наблюдение. Цитаты подставляются как вся реплика: гейт «цитата найдена в
    сообщении» на переигрывании проверить нечем, а ронять из-за него ход — значит потерять трассу."""
    obs = Observation()

    key = decision.get("script_key")
    if decision.get("next_action") == "script" and isinstance(key, str):
        signal = _REASON_TO_SIGNAL.get(key)
        if signal:
            obs.signals.append(Signal(code=signal, quote=message))
        elif key in ("REPLY_CONTACT_SOURCE", "REPLY_CONTACT_SOURCE_EMPTY"):
            obs.signals.append(Signal(code="contact_source", quote=message))
        elif key.startswith("KO_FORMAT"):
            # Отсев по формату — это не сигнал, а вывод из двух фактов. Их модель и наблюдала,
            # просто старый контракт заставлял её сразу отдать вывод. Восстанавливаем факты:
            # выбор конкретного ключа (OFFICE / HYBRID / NOCITY) в новой схеме делает код.
            obs.facts["format_ready"] = "no"
            obs.facts["relocation_ready"] = "no"
        elif key == "KO_GEO":
            obs.facts["geo_blocked"] = True

    event = decision.get("event")
    if event == "demand":
        obs.persistent = True
    elif isinstance(event, str):
        signal = _EVENT_TO_SIGNAL.get(event)
        if signal and not obs.has(signal):
            obs.signals.append(Signal(code=signal, quote=message))

    claim = decision.get("salary_claim")
    obs.salary_claim = claim if isinstance(claim, dict) else None

    for upd in decision.get("updates") or []:
        if not isinstance(upd, dict):
            continue
        upd_key, value = upd.get("key"), upd.get("value")
        if upd_key == "candidate_city" and value:
            obs.facts["candidate_city"] = value
        elif upd_key == "format_check" and value == "closed":
            obs.facts["format_ready"] = "yes"
        elif isinstance(upd_key, str) and upd_key.startswith("q") and value == "closed":
            obs.answers.append({"key": upd_key, "substantive": True})

    # `focus_answered` в старом контракте не выражен. Берём консервативное «ответил по сути», если
    # ход хоть что-то закрыл: это влияет только на исключение R3a, а оно и так вне зоны проверки.
    obs.focus_answered = "substantive" if (obs.answers or obs.facts) else "none"

    instruction = decision.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        obs.reply_material.append({"kind": "answer", "text": instruction.strip()})

    return obs


def expected_outcome(decision: dict) -> tuple[str, Optional[str]]:
    """Что старый движок сделал этим ходом: («script», ключ) либо («ask», фокус).

    Фокус берётся из `asking`; `null` при `ask` — это законная ветка «только ответили кандидату»
    (`..engine:122`), в новом ядре ей соответствует `ANSWER_ONLY`.
    """
    if decision.get("next_action") == "script":
        return "script", decision.get("script_key")
    return "ask", decision.get("asking")


def state_before(case_turns: list, index: int, init: dict) -> dict:
    """Состояние ДО хода `index`. В трассе сохраняется состояние ПОСЛЕ хода, поэтому «до» — это
    «после» предыдущего, а для первого хода — стартовое состояние диалога."""
    if index == 0:
        return init
    prev = case_turns[index - 1]
    return prev.get("state") or init

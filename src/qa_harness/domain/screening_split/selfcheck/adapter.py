"""Переходник `Decision` → `Observation` для переигрывания записанных трасс (`policy_replay`).

Переходник — единственный код, который остался связан со старым контрактом, и ошибка в нём тихая:
переигрывание просто покажет расхождение там, где его нет, либо скроет настоящее. Проверяется его
ГРАНИЦА ПРИМЕНИМОСТИ (какие ходы он честно отказывается брать) и восстановление каждого класса поля.
"""

from typing import List

from ..policy.adapter import (
    CODE_FORCED_KEYS,
    decision_to_observation,
    expected_outcome,
    is_replayable,
    state_before,
)
from .collect import Checks, Row

MSG = "Живу в Казани, ориентируюсь на 250 тысяч"


def _dec(**kw) -> dict:
    base = {"next_action": "ask", "script_key": None, "instruction": "Спроси город",
            "updates": [], "event": None, "asking": "salary", "salary_claim": None}
    base.update(kw)
    return base


def checks() -> List[Row]:
    c = Checks()

    # --- граница применимости: непригодные ходы отсеиваются с названной причиной ---
    ok, why = is_replayable(_dec())
    c.add("обычный ход переигрываем", ok, why)
    ok, why = is_replayable({})
    c.add("пустое решение отвергнуто", not ok and why == "решения нет", why)
    ok, why = is_replayable(_dec(source="code"))
    c.add("подменённое кодом решение отвергнуто", not ok and "подменено" in why, why)
    dec_old = _dec()
    dec_old.pop("salary_claim")
    ok, why = is_replayable(dec_old)
    c.add("трасса до зарплатного контракта отвергнута",
          not ok and "зарплатного контракта" in why, why)
    for key in sorted(CODE_FORCED_KEYS):
        ok, _ = is_replayable(_dec(next_action="script", script_key=key, asking=None))
        if not ok:
            continue
        c.add(f"код-форсимый ключ {key} отвергнут", False, "принят к переигрыванию")
        break
    else:
        c.add("все код-форсимые ключи отвергнуты", True, str(len(CODE_FORCED_KEYS)) + " ключей")

    # --- восстановление сигналов ---
    obs = decision_to_observation(_dec(next_action="script", script_key="STOP_ABUSE", asking=None), MSG)
    c.add("терминальный script_key → терминальный сигнал",
          obs.codes() == ["abuse"], str(obs.codes()))
    obs = decision_to_observation(_dec(next_action="script", script_key="REPLY_CONTACT_SOURCE",
                                       asking=None), MSG)
    c.add("скрипт источника контакта → сигнал contact_source",
          obs.codes() == ["contact_source"], str(obs.codes()))
    obs = decision_to_observation(_dec(event="bot_check"), MSG)
    c.add("event → счётный сигнал", obs.codes() == ["bot_check"], str(obs.codes()))
    obs = decision_to_observation(_dec(event="demand"), MSG)
    c.add("event=demand → флаг persistent, а не сигнал",
          obs.persistent and obs.codes() == [], str(obs.codes()))

    # --- отсев по формату восстанавливается ФАКТАМИ: ключ теперь выбирает код ---
    obs = decision_to_observation(_dec(next_action="script", script_key="KO_FORMAT_OFFICE",
                                       asking=None), MSG)
    c.add("KO_FORMAT_* → факты, а не готовый вывод",
          obs.facts.get("format_ready") == "no" and obs.codes() == [], str(obs.facts))
    obs = decision_to_observation(_dec(next_action="script", script_key="KO_GEO", asking=None), MSG)
    c.add("KO_GEO → факт geo_blocked", obs.facts.get("geo_blocked") is True, str(obs.facts))

    # --- updates → факты и ответы ---
    obs = decision_to_observation(_dec(updates=[{"key": "candidate_city", "value": "Казань"},
                                                {"key": "format_check", "value": "closed"},
                                                {"key": "q1", "value": "closed"}]), MSG)
    c.add("updates восстанавливают город, формат и ответ на доп-вопрос",
          obs.facts.get("candidate_city") == "Казань"
          and obs.facts.get("format_ready") == "yes"
          and obs.answers == [{"key": "q1", "substantive": True}],
          f"{obs.facts} {obs.answers}")
    c.add("закрытие пункта считается ответом по сути",
          obs.focus_answered == "substantive", obs.focus_answered)

    # --- salary_claim переносится 1:1 ---
    claim = {"subject": "own_expectation", "amount_min": 250, "scale": "thousand"}
    obs = decision_to_observation(_dec(salary_claim=claim), MSG)
    c.add("salary_claim переносится дословно", obs.salary_claim == claim, str(obs.salary_claim))

    # --- ожидаемый исход старого хода ---
    c.add("исход хода-скрипта: ключ",
          expected_outcome(_dec(next_action="script", script_key="FINISH", asking=None))
          == ("script", "FINISH"))
    c.add("исход хода-вопроса: фокус",
          expected_outcome(_dec(asking="format")) == ("ask", "format"))
    c.add("asking=null — законная ветка «только ответили»",
          expected_outcome(_dec(asking=None)) == ("ask", None))

    # --- состояние ДО хода ---
    turns = [{"state": {"salary": "closed"}}, {"state": {"salary": "closed", "city_check": "closed"}}]
    c.add("для первого хода берётся стартовое состояние",
          state_before(turns, 0, {"salary": "pending"}) == {"salary": "pending"})
    c.add("для последующих — состояние ПОСЛЕ предыдущего",
          state_before(turns, 1, {}) == {"salary": "closed"})

    return c.rows

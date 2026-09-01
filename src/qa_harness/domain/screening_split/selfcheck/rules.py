"""Таблица правил R1-R11: порядок строк и три старшинства, растворённых в нём.

До перестройки эти три отношения жили ветками движка и проверялись прогоном старого ядра; после неё
они держатся ТОЛЬКО порядком строк в `policy/rules.py` и до сих пор не были закрыты ни одной
проверкой — переставь две строки местами, и всё останется зелёным. Здесь они закрыты.

Сюда же стоп-кран `R9a` и ветка `REFUSE_AND_ADVANCE`: единственное место, где ход дорешивается вторым
проходом по таблице.
"""

from typing import Any, List, Optional

from .. import state as state_model
from ..policy import DecideContext, decide
from ..policy.budgets import EVENT_BUDGETS, STALL_BUDGET
from ..policy.observation import Observation, Signal
from ..policy.rules import RULES
from .collect import Checks, Row
from .salary import BAND_MAX, BAND_MIN, claim

CTX = DecideContext(band_min=BAND_MIN, band_max=BAND_MAX, work_format="remote", location="Москва")


def _obs(*, signals: tuple = (), focus_answered: str = "none", persistent: bool = False,
         salary_claim: Optional[dict] = None, answers: tuple = ()) -> Observation:
    o = Observation()
    o.signals = [Signal(code=code, quote=quote) for code, quote in signals]
    o.focus_answered = focus_answered
    o.persistent = persistent
    o.salary_claim = salary_claim
    o.answers = list(answers)
    o.facts = {}
    return o


def _state(**over: Any) -> dict:
    st = state_model.init_state("remote", "- Опыт с Python?\n- Сервисы под нагрузкой?")
    st["city_check"] = "closed"
    st["candidate_city"] = "Москва"
    st.update(over)
    return st


def _done() -> dict:
    """Повестка закрыта целиком: зарплата, город, формат n/a, доп-вопросов нет."""
    st = state_model.init_state("remote", "")
    st["salary"] = "closed"
    st["city_check"] = "closed"
    st["candidate_city"] = "Москва"
    return st


KO = claim(amount_min=400, quote="400 тысяч")


def checks() -> List[Row]:
    c = Checks()

    # --- порядок строк — это ДАННЫЕ, и он проверяется как данные ---
    names = [r.name.split(".")[0] for r in RULES]
    c.add("порядок правил R1..R11 не переставлен",
          names == ["R1", "R2", "R3", "R4", "R5", "R5a", "R6", "R7", "R8", "R9", "R9a", "R10", "R11"],
          str(names))

    # --- R1: диалог закрыт — молчим, а не переоткрываем ---
    plan = decide(_state(), _obs(), "ещё вопрос", CTX, dialogue_closed=True)
    c.add("R1 закрытый диалог: молчание и остаёмся закрытыми",
          plan.kind == "silent" and plan.end, f"{plan.rule}/{plan.kind}/{plan.end}")

    # --- R2: наблюдения нет — фолбэк, диалог НЕ завершаем, счётчики не жжём ---
    st = _state(counters={**state_model.init_state("remote", "")["counters"], "gibberish": 1})
    plan = decide(st, _obs(), "что-то", CTX, analyzer_failed=True)
    c.add("R2 сбой наблюдения: REPLY_FALLBACK без завершения",
          plan.reason_code == "REPLY_FALLBACK" and not plan.end, f"{plan.rule}/{plan.reason_code}")
    c.add("R2 счётчик кандидата за наш сбой не платит",
          plan.state_next["counters"]["gibberish"] == 1,
          str(plan.state_next["counters"]["gibberish"]))

    # --- R3 > R4: неденежное терминальное сильнее отсева по деньгам ---
    plan = decide(_state(), _obs(signals=(("abuse", "уроды"),), salary_claim=KO),
                  "Хочу 400 тысяч, уроды", CTX)
    c.add("R3 > R4: STOP_ABUSE перебивает KO_SALARY",
          plan.reason_code == "STOP_ABUSE" and plan.end, f"{plan.rule}/{plan.reason_code}")
    c.add("R3 > R4: перебитый отсев виден в аудите",
          (plan.audit.get("salary") or {}).get("effect") == "ko_overridden_by_signal",
          str((plan.audit.get("salary") or {}).get("effect")))

    # --- R3a: «нет опыта» не исполняется, если тем же ходом ответили по сути ---
    plan = decide(_state(salary="closed"),
                  _obs(signals=(("no_experience", "с Kafka не работал"),),
                       focus_answered="substantive"),
                  "С Kafka не работал, но Python шесть лет", CTX)
    c.add("R3a ответ по сути снимает no_experience",
          plan.reason_code != "STOP_NO_EXPERIENCE" and not plan.end,
          f"{plan.rule}/{plan.reason_code}")
    c.add("R3a пропуск сигнала записан в аудит",
          "no_experience" in (plan.audit.get("signals_skipped") or []),
          str(plan.audit.get("signals_skipped")))
    plan = decide(_state(salary="closed"),
                  _obs(signals=(("no_experience", "я вообще не разработчик"),)),
                  "Я вообще не разработчик", CTX)
    c.add("R3a без ответа по сути сигнал исполняется",
          plan.reason_code == "STOP_NO_EXPERIENCE" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- R3b: «не интересно» рядом с вопросом по делу не исполняется, прямой отказ — исполняется ---
    plan = decide(_state(),
                  _obs(signals=(("not_interested", "это не то направление"),
                                ("company_info", "а что за компания")),
                       focus_answered="none"),
                  "Это не то направление, а что за компания?", CTX)
    c.add("R3b вопрос по делу снимает not_interested",
          plan.reason_code != "STOP_NOT_INTERESTED" and not plan.end,
          f"{plan.rule}/{plan.reason_code}")
    plan = decide(_state(),
                  _obs(signals=(("not_interested", "не пишите мне больше"),
                                ("company_info", "что за компания")),
                       focus_answered="refusal"),
                  "Не пишите мне больше, что за компания вообще", CTX)
    c.add("R3b прямой отказ остаётся терминальным",
          plan.reason_code == "STOP_NOT_INTERESTED" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- R4 > R7: зарплатный вердикт выше событийного порога ---
    before = EVENT_BUDGETS["demand"].fires_on_nth - 1
    st = _state(counters={**state_model.init_state("remote", "")["counters"], "demand": before})
    plan = decide(st, _obs(persistent=True, salary_claim=KO), "Повторяю, 400 тысяч и не меньше", CTX)
    c.add("R4 > R7: причина отказа — деньги, а не порог настойчивости",
          plan.reason_code == "KO_SALARY" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- R4 > R9: ko перебивает FINISH ---
    plan = decide(_done(), _obs(salary_claim=KO), "И ещё: хочу 400 тысяч", CTX)
    c.add("R4 > R9: скрининг с отсевом по деньгам успешным не был",
          plan.reason_code == "KO_SALARY" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- R9: повестка закрыта — немедленный FINISH, без «ещё одного» вопроса ---
    plan = decide(_done(), _obs(), "буду ждать звонка", CTX)
    c.add("R9 повестка закрыта: FINISH и завершение",
          plan.reason_code == "FINISH" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(_done(), _obs(signals=(("pause", "давайте позже"),)), "давайте позже", CTX)
    c.add("R9 пауза FINISH не удерживает (решение 28.08)",
          plan.reason_code == "FINISH" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- R8 question: кап доп-вопроса не рубит диалог, а помечает пункт refused ---
    st = _state(salary="closed", last_asking="q1")
    st["questions"][0]["reask_count"] = 2
    plan = decide(st, _obs(focus_answered="deflection"), "всё есть в резюме", CTX)
    statuses = {q["key"]: q["status"] for q in plan.state_next["questions"]}
    c.add("R8 третий переспрос доп-вопроса: пункт refused",
          statuses.get("q1") == "refused", str(statuses))
    c.add("R8 фокус уехал на следующий вопрос, диалог жив",
          plan.focus == "q2" and not plan.end, f"{plan.focus}/{plan.end}")
    c.add("R8 бюджет второго пункта на этом ходе не сожжён",
          plan.state_next["questions"][1].get("reask_count", 0) == 0,
          str(plan.state_next["questions"][1].get("reask_count")))

    # --- R9a: стоп-кран по ходам без прогресса ---
    nth = STALL_BUDGET.fires_on_nth
    st = _state(no_progress=nth - 1)
    plan = decide(st, _obs(), "ага", CTX)
    c.add(f"R9a {nth} ходов без прогресса: STOP_PERSISTENT",
          plan.reason_code == "STOP_PERSISTENT" and plan.end, f"{plan.rule}/{plan.reason_code}")
    st = _state(no_progress=nth - 1, salary="pending")
    plan = decide(st, _obs(salary_claim=claim(amount_min=250, quote="250 тысяч")),
                  "Ориентируюсь на 250 тысяч", CTX)
    c.add("R9a ход с новым фактом счётчик обнуляет, а не добивает",
          plan.state_next["no_progress"] == 0 and not plan.end,
          str(plan.state_next["no_progress"]))

    # --- R11: фокус пуст и повестки нет — просто отвечаем, скрипта не рендерим ---
    plan = decide(_state(salary="closed"), _obs(signals=(("company_info", "что за компания"),)),
                  "Что за компания?", CTX)
    c.add("R10 нетерминальный сигнал не завершает диалог",
          not plan.end and plan.kind == "ask", f"{plan.rule}/{plan.kind}")

    return c.rows

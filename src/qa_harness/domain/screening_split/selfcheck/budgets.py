"""Бюджеты завершения: одна семантика порога и ровно один счётчик за ход.

`fires_on_nth` = порядковый номер срабатывания, одинаково для событийных и для переспросов (Р3). До
перестройки семантик было ДВЕ, и «2» значило разное в двух половинах таблицы — проверка держит, что
разъезд не вернулся. `persist_on_fire` семантикой порога не является: это вопрос «писать ли счётчик
на ходе срабатывания».
"""

from typing import List

from .. import state as state_model
from ..policy.budgets import EVENT_BUDGETS, REASK_BUDGETS, STALL_BUDGET, Budget, config_digest
from ..policy.core import _charge_counter
from ..policy.observation import Observation, Signal
from ..policy.reasons import is_known
from .collect import Checks, Row


def checks() -> List[Row]:
    c = Checks()

    # --- семантика порога ---
    b = Budget(counter="x", fires_on_nth=2, reason="STOP_X")
    c.add("порог 2 не срабатывает на первом событии", not b.fires(0))
    c.add("порог 2 срабатывает на втором", b.fires(1))
    c.add("порога нет — не срабатывает никогда",
          not Budget(counter="x", fires_on_nth=None, reason=None).fires(99))
    c.add("persist_on_fire=True: значение растёт и на срабатывании", b.next_value(1) == 2)
    nb = Budget(counter="x", fires_on_nth=2, reason="STOP_X", persist_on_fire=False)
    c.add("persist_on_fire=False: на срабатывании значение не растёт", nb.next_value(1) == 1)
    c.add("persist_on_fire=False: до срабатывания растёт", nb.next_value(0) == 1)

    # --- таблица событийных ---
    expected_events = {"gibberish": 2, "bot_check": 2, "demand": 3, "contact_source": 3,
                       "pause": 3, "salary_info": None}
    got_events = {k: v.fires_on_nth for k, v in EVENT_BUDGETS.items()}
    c.add("событийные пороги на месте", got_events == expected_events, str(got_events))
    c.add("salary_info порога не имеет намеренно (Д4/О4)",
          EVENT_BUDGETS["salary_info"].reason is None)

    # --- таблица переспросов ---
    c.add("все пункты повестки переспрашиваются до третьего раза",
          all(b.fires_on_nth == 3 for b in REASK_BUDGETS.values()),
          str({k: v.fires_on_nth for k, v in REASK_BUDGETS.items()}))
    c.add("доп-вопрос капом не рубит диалог, а уходит в REFUSE_AND_ADVANCE",
          REASK_BUDGETS["question"].reason == "REFUSE_AND_ADVANCE")
    c.add("переспросы на срабатывании счётчик не пишут",
          all(not b.persist_on_fire for b in REASK_BUDGETS.values()))
    c.add("пункты повестки закрыты бюджетами полностью",
          set(REASK_BUDGETS) == {"salary", "format", "city", "relocation", "question"},
          str(sorted(REASK_BUDGETS)))
    c.add(f"стоп-кран на {STALL_BUDGET.fires_on_nth} ходах без прогресса",
          STALL_BUDGET.counter == "no_progress" and STALL_BUDGET.fires_on_nth == 4,
          f"{STALL_BUDGET.counter}/{STALL_BUDGET.fires_on_nth}")

    # --- у каждой причины бюджета есть текст в реестре ---
    orphans = sorted({b.reason for b in list(EVENT_BUDGETS.values()) + list(REASK_BUDGETS.values())
                      if b.reason and b.reason != "REFUSE_AND_ADVANCE" and not is_known(b.reason)})
    c.add("причины бюджетов есть в реестре", not orphans, str(orphans))

    # --- отпечаток таблицы: сравнение портов должно быть сравнением строк ---
    digest = config_digest()
    c.add("config_digest стабилен между вызовами", digest == config_digest(), digest)
    saved = EVENT_BUDGETS["pause"]
    EVENT_BUDGETS["pause"] = Budget(counter="pause", fires_on_nth=9, reason="STOP_PAUSE")
    changed = config_digest()
    EVENT_BUDGETS["pause"] = saved
    c.add("config_digest меняется при подмене порога", changed != digest, f"{digest} -> {changed}")
    c.add("подмена откачена", config_digest() == digest)

    # --- ровно один счётчик за ход ---
    st = state_model.init_state("remote", "")
    obs = Observation()
    obs.signals = [Signal(code="bot_check", quote="вы бот"), Signal(code="pause", quote="позже")]
    new_state, key, before = _charge_counter(obs, st)
    c.add("за ход начисляется РОВНО один счётчик",
          sum(new_state["counters"].values()) == 1, str(new_state["counters"]))
    c.add("начисляется первый сигнал по порядку наблюдения", key == "bot_check", str(key))
    c.add("значение «до хода» отдаётся отдельно", before == 0, str(before))

    obs = Observation()
    obs.persistent = True
    _, key, _ = _charge_counter(obs, st)
    c.add("persistent без сигнала даёт demand", key == "demand", str(key))
    obs.signals = [Signal(code="bot_check", quote="вы бот")]
    _, key, _ = _charge_counter(obs, st)
    c.add("сигнал важнее флага persistent", key == "bot_check", str(key))

    obs = Observation()
    obs.signals = [Signal(code="company_info", quote="что за компания")]
    _, key, _ = _charge_counter(obs, st)
    c.add("несчётный сигнал ничего не начисляет", key is None, str(key))

    return c.rows

"""Бюджеты завершения — HH-канал. Семантика порога та же: `fires_on_nth` = порядковый номер срабатывания.

Дельта к TG:
- в событийных нет `contact_source` — в канале нет такого события (`..state.COUNTER_KEYS`);
- в переспросах есть **`field_work`**: разъездной формат в hh — отдельный приоритет со своим
  счётчиком `field_work_reasks`.

Значения порогов совпадают с TG намеренно: каналы держим сходящимися. Отсюда же следует правка для
`eggplant-api`, где сегодня `REASK_CAP = 2` со старой семантикой «сверяем до инкремента»: по новой
записи это `3` (план, hh-контур, п. 7). Класс `Budget` общий — импортируем.
"""

from qa_harness.domain.screening_split.policy.budgets import Budget

EVENT_BUDGETS: dict[str, Budget] = {
    "gibberish":   Budget(counter="gibberish",   fires_on_nth=2, reason="STOP_GIBBERISH_REPEAT"),
    "bot_check":   Budget(counter="bot_check",   fires_on_nth=2, reason="STOP_BOT_REPEAT"),
    "demand":      Budget(counter="demand",      fires_on_nth=3, reason="STOP_PERSISTENT"),
    "pause":       Budget(counter="pause",       fires_on_nth=3, reason="STOP_PAUSE"),
    "salary_info": Budget(counter="salary_info", fires_on_nth=None, reason=None),
}

REASK_BUDGETS: dict[str, Budget] = {
    "salary":     Budget(counter="salary_reasks",     fires_on_nth=3, reason="STOP_SALARY_DEMAND", persist_on_fire=False),
    "format":     Budget(counter="format_reasks",     fires_on_nth=3, reason="STOP_PERSISTENT",    persist_on_fire=False),
    "field_work": Budget(counter="field_work_reasks", fires_on_nth=3, reason="STOP_PERSISTENT",    persist_on_fire=False),
    "city":       Budget(counter="city_reasks",       fires_on_nth=3, reason="STOP_PERSISTENT",  persist_on_fire=False),
    "relocation": Budget(counter="relocation_reasks", fires_on_nth=3, reason="STOP_PERSISTENT",  persist_on_fire=False),
    "question":   Budget(counter="reask_count",       fires_on_nth=3, reason="REFUSE_AND_ADVANCE", persist_on_fire=False),
}

STALL_BUDGET = Budget(counter="no_progress", fires_on_nth=4, reason=None)


def config_digest() -> str:
    """Отпечаток hh-таблиц для аудит-записи хода: расхождение портов должно быть сравнением строк,
    а не сверкой глазами (задача Б2)."""
    import hashlib
    parts: list[str] = []
    for name, table in (("event", EVENT_BUDGETS), ("reask", REASK_BUDGETS)):
        for key in sorted(table):
            b = table[key]
            parts.append(f"{name}.{key}={b.counter}:{b.fires_on_nth}:{b.reason}:{int(b.persist_on_fire)}")
    parts.append(f"stall={STALL_BUDGET.counter}:{STALL_BUDGET.fires_on_nth}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]

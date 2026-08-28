"""Бюджеты завершения — данными, а не расстановкой `if`.

Решение Р3 (docs/screening_split/decisions_rearchitecture.md): у всех строк ОДНА семантика порога —
**`fires_on_nth` = порядковый номер срабатывания**. «Срабатывает на 3-м» → в таблице стоит `3`.
Никакого пересчёта в уме и никакого расхождения с промптом, где написано «завершает диалог на 3-й раз»
(`screening_analyzer/v2/system.md:256`), а в коде сегодня стоит `2`.

Что было в движке до этого — ДВЕ разные семантики (см. `..engine`):
  событийные счётчики — инкремент (`:325-326`), затем сверка `>= порог` (`:352`) → «2» = 2-е событие;
  reask-cap         — сверка `>= 2` (`:373`), затем инкремент (`:376`)          → «2» = 3-й переспрос.
Фраза handoff §6 «пороги проверяются до инкремента, поэтому порог 2 = срабатывание на 3-м» верна
только для второй половины таблицы. Перенести пороги в конфиг «как написано в документе» — сломать
сценарии 3, 6 и 27.

Единственное, что осталось различаться, — `persist_on_fire`, и это НЕ семантика порога, а вопрос
«записывать ли счётчик на ходе срабатывания». Значения выбраны так, чтобы состояние совпадало с
сегодняшним байт-в-байт: событийные пишут всегда, reask на срабатывании не пишет (там ход уходит в
завершение либо в `refused`, и значение счётчика больше никем не читается).

СЛЕДСТВИЕ ДЛЯ ПОРТОВ: значения обязаны совпадать во всех трёх копиях. `contact_source` в eggplant
сегодня отсутствует в `COUNTER_KEYS` (`eggplant-api/app/assistants/screening/state.py:3`), то есть
порог там не срабатывает никогда — это расхождение, а не канальная особенность.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Budget:
    """Одна строка таблицы бюджетов.

    `fires_on_nth` — на каком по счёту событии срабатывает (1 = на первом же). `None` — лимита нет:
    счётчик ведётся, но ничего не завершает.
    `reason` — `reason_code` исхода при срабатывании.
    `persist_on_fire` — записывать ли увеличенное значение в состояние на ходе срабатывания.
    """

    counter: str
    fires_on_nth: Optional[int]
    reason: Optional[str]
    persist_on_fire: bool = True

    def fires(self, before: int) -> bool:
        """`before` — значение счётчика ДО этого хода. Срабатывание считается по порядковому номеру."""
        return self.fires_on_nth is not None and (before + 1) >= self.fires_on_nth

    def next_value(self, before: int) -> int:
        """Что записать в состояние. На срабатывании при `persist_on_fire=False` значение не растёт —
        ровно как сегодня в reask-cap, где инкремент стоит в ветке `else`."""
        if self.fires(before) and not self.persist_on_fire:
            return before
        return before + 1


# ── событийные счётчики (сегодня `_EVENT_STOP` в `..engine:36-42`) ────────────
# `salary_info` порога не имеет намеренно (решения Д4 и О4 не переоткрываются): его читает только
# селектор шаблона вопроса про деньги — «первый раз объясняем, дальше просто переспрашиваем».
EVENT_BUDGETS: dict[str, Budget] = {
    "gibberish":      Budget(counter="gibberish",      fires_on_nth=2, reason="STOP_GIBBERISH_REPEAT"),
    "bot_check":      Budget(counter="bot_check",      fires_on_nth=2, reason="STOP_BOT_REPEAT"),
    "demand":         Budget(counter="demand",         fires_on_nth=3, reason="STOP_PERSISTENT"),
    "contact_source": Budget(counter="contact_source", fires_on_nth=3, reason="STOP_PERSISTENT"),
    "pause":          Budget(counter="pause",          fires_on_nth=3, reason="STOP_PAUSE"),
    "salary_info":    Budget(counter="salary_info",    fires_on_nth=None, reason=None),
}

# ── лимит переспросов одного и того же незакрытого пункта (`..engine:364-400`) ─
# Порог 3 = срабатывание на 3-м переспросе. В коде сегодня стоит 2 при обратной семантике — то же
# поведение, другая запись. `question` не завершает диалог: пункт помечается `refused`, фокус едет
# дальше, следующий вопрос рендерит код — и это единственное место, где уходит второй вызов модели.
REASK_BUDGETS: dict[str, Budget] = {
    "salary":   Budget(counter="salary_reasks", fires_on_nth=3, reason="STOP_SALARY_DEMAND", persist_on_fire=False),
    "format":   Budget(counter="format_reasks", fires_on_nth=3, reason="STOP_PERSISTENT",    persist_on_fire=False),
    "question": Budget(counter="reask_count",   fires_on_nth=3, reason="REFUSE_AND_ADVANCE", persist_on_fire=False),
}

# ── универсальный стоп-кран (`..engine:44-46, :404-410`) ──────────────────────
# Ходов подряд без нового собранного факта. Живьём почти недостижим — reask-cap и gibberish
# перехватывают луп за 2–4 хода, — но остаётся страховкой (решение Д5). Исход зависит от того,
# всё ли собрано: `FINISH` либо `STOP_PERSISTENT`, поэтому `reason` здесь пустой.
STALL_BUDGET = Budget(counter="no_progress", fires_on_nth=4, reason=None)


def config_digest() -> str:
    """Отпечаток таблицы бюджетов для аудит-записи хода.

    Смысл — сделать расхождение портов НАБЛЮДАЕМЫМ: сегодня пороги живут в четырёх местах трёх
    репозиториев, и сверка возможна только глазами (задача Б2). Хэш в трассе превращает это в
    сравнение двух строк.
    """
    import hashlib
    parts: list[str] = []
    for name, table in (("event", EVENT_BUDGETS), ("reask", REASK_BUDGETS)):
        for key in sorted(table):
            b = table[key]
            parts.append(f"{name}.{key}={b.counter}:{b.fires_on_nth}:{b.reason}:{int(b.persist_on_fire)}")
    parts.append(f"stall={STALL_BUDGET.counter}:{STALL_BUDGET.fires_on_nth}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]

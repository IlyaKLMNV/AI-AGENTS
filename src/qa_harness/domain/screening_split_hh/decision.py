"""Контракт `Decision` Аналитика: парсинг + валидация — HH-канал.

Форма Decision (6 полей) общая с TG — потребитель может использовать общий парсер. Дельта
(EGGPLANT_SPLIT_TASK.md §2 / SPLIT_TG_VS_HH.md §2.4):
- `event` без `contact_source` (в hh источника контакта нет) → `COUNTER_KEYS` из hh-state;
- `script_key` проверяется по hh-реестру (`is_known` из hh-scripts): `KO_LOCATION`/`KO_FORMAT`/…,
  а `KO_GEO`/`STOP_ABROAD`/`REPLY_CONTACT_SOURCE` — уже НЕ известны.

Логика 1:1 с TG `screening_split/decision.py`; расходятся только источники `is_known`/`COUNTER_KEYS`.
"""

import json

from .scripts import is_known
from .state import COUNTER_KEYS

REQUIRED_FIELDS = ("next_action", "script_key", "instruction", "updates", "event", "asking")


def parse_and_validate(text: str) -> tuple[dict | None, str]:
    """(decision, "") если валиден; иначе (None, причина). Порт прод-логики 1:1 (hh-реестр/enum)."""
    try:
        decision = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "ответ не является JSON"

    if not isinstance(decision, dict):
        return None, "ответ не объект"

    missing = set(REQUIRED_FIELDS) - decision.keys()
    if missing:
        return None, f"нет полей: {sorted(missing)}"

    action = decision.get("next_action")
    if action not in ("ask", "script"):
        return None, f"next_action недопустим: {action!r}"

    if action == "script":
        sk = decision.get("script_key")
        if not sk or not is_known(sk):
            return None, f"неизвестный script_key: {sk!r}"
    else:  # ask
        instr = decision.get("instruction")
        if not isinstance(instr, str) or not instr.strip():
            return None, "пустой instruction при next_action=ask"

    updates = decision.get("updates")
    if not isinstance(updates, list):
        return None, "updates не список"
    for upd in updates:
        if not isinstance(upd, dict) or "key" not in upd or "value" not in upd:
            return None, "элемент updates без key/value"

    event = decision.get("event")
    if event is not None and event not in COUNTER_KEYS:
        return None, f"event недопустим: {event!r}"

    return decision, ""

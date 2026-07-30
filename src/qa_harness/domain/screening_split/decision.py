"""Контракт `Decision` Аналитика: парсинг + валидация.

Порт `ScreeningAnalyzerAssistant._parse_and_validate` (tgApi, HEAD e733095) 1:1.
Держится отдельно от analyzer.py, потому что валидность Decision — это отдельная
измеримая величина: слой A оценки (см. план) отличает «контракт нарушен» (невалидный
JSON/поля) от «семантика неверна» (валидный Decision, но неправильное решение).

Схема Decision (6 полей): next_action, script_key, instruction, updates, event, asking.
"""

import json

from .scripts import is_known
from .state import COUNTER_KEYS

REQUIRED_FIELDS = ("next_action", "script_key", "instruction", "updates", "event", "asking")


def parse_and_validate(text: str) -> tuple[dict | None, str]:
    """(decision, "") если валиден; иначе (None, причина). Порт прод-логики 1:1."""
    try:
        decision = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "ответ не является JSON"

    if not isinstance(decision, dict):
        return None, "ответ не объект"

    required = set(REQUIRED_FIELDS)
    missing = required - decision.keys()
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

"""Ошибки split-движка.

`AssistantError` — устойчивый сбой роли (Аналитик не вернул валидный Decision за N
попыток). Оркестратор ловит её и уходит в REPLY_FALLBACK, кандидату не отказывая.
Порт `app/common/assistants.AssistantError` (в проде живёт в __init__ пакета assistants).
"""


class AssistantError(Exception):
    """Устойчивый сбой роли-ассистента (после ретраев)."""

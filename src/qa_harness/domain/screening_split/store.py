"""In-memory хранилище диалогов split-скрининга (ключ — conversation_id).

Прод (tgApi) держит state в Mongo (`screening_store.ScreeningStateStore`). Для QA-стенда
Mongo не нужна и вредна (внешняя зависимость, лишний setup) — состояние живёт в памяти
процесса. Интерфейс намеренно повторяет прод-стор (`create` / `load` / `save_state`),
чтобы порт движка (engine.py) не отличался от прод-версии ни на строку.

Запись живёт и после завершения диалога — со `state` и `finished=True` (как в проде:
терминальный ход не удаляет запись, а помечает finished).
"""

from typing import Any, Optional


class InMemoryStateStore:
    """Словарь conversation_id -> запись диалога. Совместим по вызовам с прод-стором."""

    def __init__(self) -> None:
        self._docs: dict[Any, dict] = {}

    def create(
        self,
        conversation_id: Any,
        engine: str,
        *,
        state: Optional[dict] = None,
        context: Optional[str] = None,
        location: Optional[str] = None,
        contact_source: Optional[str] = None,
    ) -> None:
        """Заводит диалог. Движок фиксируется здесь и дальше не меняется."""
        self._docs[conversation_id] = {
            "conversation_id": conversation_id,
            "engine": engine,
            "state": state,
            "finished": False,
            "context": context or "",
            "location": location or "",
            "contact_source": contact_source or "",
        }

    def save_state(self, conversation_id: Any, state: dict, *, finished: bool = False) -> None:
        """Без upsert: запись заводится в create, к этому моменту она уже есть."""
        doc = self._docs.get(conversation_id)
        if doc is None:  # страховка (в проде — гарантия create до save_state)
            return
        doc["state"] = state
        doc["finished"] = finished

    def engine_of(self, conversation_id: Any) -> Optional[str]:
        """None — у диалога нет записи, значит он создан до флага: legacy."""
        doc = self._docs.get(conversation_id)
        return doc.get("engine") if doc else None

    def load(self, conversation_id: Any) -> Optional[dict]:
        """Запись живёт и после завершения диалога — со state и finished=True."""
        return self._docs.get(conversation_id)

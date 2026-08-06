"""Аналитик split-скрининга — «мозг», HH-канал.

Та же логика вызова/ретраев/сборки входа, что TG `screening_split/analyzer.py` (переиспользуем
`_build_input` — формат входа общий), но валидация Decision — hh (`decision.parse_and_validate`:
`event` без `contact_source`, `script_key` по hh-реестру). Клиент промпта (`screening_analyzer_hh`
из пакета `prompts`) инжектится: `prompt_client.run(input_text) -> (text, usage)`.
"""

from typing import Any

from qa_harness.core import accumulate_usage, blank_usage
from qa_harness.domain.screening_split.analyzer import MAX_ATTEMPTS
from qa_harness.domain.screening_split.analyzer import ScreeningAnalyzer as _TgAnalyzer
from qa_harness.domain.screening_split.errors import AssistantError

from .decision import parse_and_validate


class ScreeningAnalyzer:
    """Один вызов LLM: (hh-контекст + STATE + сообщение) → валидированный hh-Decision (dict)."""

    def __init__(self, prompt_client: Any, *, max_attempts: int = MAX_ATTEMPTS) -> None:
        self._client = prompt_client
        self._max_attempts = max_attempts
        self.last_usage: dict = blank_usage()

    def run(self, vacancy_context: str, state: dict, message: str) -> dict:
        base_input = _TgAnalyzer._build_input(vacancy_context, state, message)  # формат входа общий с TG
        self.last_usage = blank_usage()

        last_error = "unknown"
        user_input = base_input
        for _ in range(self._max_attempts):
            try:
                text, usage = self._client.run(user_input)
            except Exception as e:  # транзиентный сбой вызова — пробуем ещё раз
                last_error = f"вызов упал: {e!r}"
                continue

            accumulate_usage(self.last_usage, usage)
            decision, error = parse_and_validate(text)  # hh-валидация
            if decision is not None:
                return decision

            last_error = error
            user_input = (
                f"{base_input}\n\n[Система: предыдущий ответ невалиден — {error}. "
                f"Верни СТРОГО JSON по схеме, без текста вокруг.]"
            )

        raise AssistantError(
            f"Аналитик не вернул валидный Decision за {self._max_attempts} попытки: {last_error}"
        )

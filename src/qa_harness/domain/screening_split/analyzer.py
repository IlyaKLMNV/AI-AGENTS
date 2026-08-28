"""Аналитик split-скрининга — «мозг».

Порт `ScreeningAnalyzerAssistant` (tgApi, HEAD e733095). Единственная адаптация: вместо
`app.common.prompts.provider.get_registry()` + `app.common.llm_client.respond(spec, ...)`
клиент промпта ИНЖЕКТИТСЯ (`prompt_client.run(input_text) -> (text, usage)`) — это
`core.LocalPromptClient` компонента `screening_analyzer` (тело/схема Decision из пакета
`prompts`). Логика сборки входа, ретраев и валидации — 1:1 с продом.

Дополнительно к проду: копим usage по всем попыткам (self.last_usage) — прод-логика
поведения от этого не зависит, но QA нужен учёт токенов.
"""

import json
from typing import Any

from qa_harness.core import accumulate_usage, blank_usage

from .decision import parse_and_validate
from .errors import AssistantError

MAX_ATTEMPTS = 3


class ScreeningAnalyzer:
    """Один вызов LLM: (контекст + STATE + сообщение) → валидированный Decision (dict)."""

    def __init__(self, prompt_client: Any, *, max_attempts: int = MAX_ATTEMPTS) -> None:
        self._client = prompt_client
        self._max_attempts = max_attempts
        self.last_usage: dict = blank_usage()

    def run(self, vacancy_context: str, state: dict, message: str, *,
            note: str | None = None) -> dict:
        """Возвращает валидированный Decision (dict) или бросает AssistantError.

        `note` — служебная строка от КОДА для повторного вызова в том же ходе (перерешивание при
        расхождении по зарплате). Без неё второй вызов получает ТОЖДЕСТВЕННЫЙ вход (контекст,
        сообщение и state те же — отклонённый `salary: closed` в state не попал) и при temperature=0
        возвращает то же решение, то есть перерешивание становится no-op. Формат тот же, что у
        служебной строки про невалидный JSON ниже; правил в промпте не требует.
        """
        base_input = self._build_input(vacancy_context, state, message)
        if note:
            base_input = f"{base_input}\n\n[Система: {note}]"
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
            decision, error = parse_and_validate(text)
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

    @staticmethod
    def _build_input(vacancy_context: str, state: dict, message: str) -> str:
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        return (
            "== КОНТЕКСТ ВАКАНСИИ ==\n"
            f"{vacancy_context}\n\n"
            "== STATE (накопленное состояние) ==\n"
            f"{state_json}\n\n"
            "== НОВОЕ СООБЩЕНИЕ КАНДИДАТА ==\n"
            f"{message}"
        )

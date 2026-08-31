"""Наблюдатель — «уши» hh-канала. Один вызов LLM: (контекст + STATE + сообщение) → `Observation`.

Отличие от TG-версии только в двух вызовах — проекция состояния и разбор ответа берутся hh-шные
(`.core.state_for_prompt`, `.observation.parse_observation`). Всё остальное — попытки, мягкая
валидация, отсутствие параметра `note` — идентично: второго вызова Наблюдателя в ходе не бывает.

Файл — копия, а не наследник: в TG разбор вызывается функцией модуля, и подменять его подклассом
пришлось бы через ломающую правку общего порта. Копия честнее — ровно её же получит `eggplant-api`.
"""

import json
from typing import Any

from qa_harness.core import accumulate_usage, blank_usage
from qa_harness.domain.screening_split.errors import AssistantError

from .core import state_for_prompt
from .observation import Observation, parse_observation

MAX_ATTEMPTS = 3


class ScreeningObserver:
    """Один вызов LLM → валидированное `Observation` (hh)."""

    def __init__(self, prompt_client: Any, *, max_attempts: int = MAX_ATTEMPTS) -> None:
        self._client = prompt_client
        self._max_attempts = max_attempts
        self.last_usage: dict = blank_usage()
        self.last_raw: Any = None
        self.last_problems: str = ""

    def run(self, vacancy_context: str, state: dict, message: str) -> Observation:
        """Возвращает `Observation` либо бросает `AssistantError` (исчерпаны попытки)."""
        base_input = self._build_input(vacancy_context, state, message)
        self.last_usage = blank_usage()
        self.last_raw = None
        self.last_problems = ""

        last_error = "unknown"
        user_input = base_input
        for _ in range(self._max_attempts):
            try:
                text, usage = self._client.run(user_input)
            except Exception as exc:  # транзиентный сбой вызова — пробуем ещё раз
                last_error = f"вызов упал: {exc!r}"
                continue

            accumulate_usage(self.last_usage, usage)
            try:
                raw = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                last_error = "ответ не является JSON"
                user_input = (
                    f"{base_input}\n\n[Система: предыдущий ответ невалиден — {last_error}. "
                    f"Верни СТРОГО JSON по схеме, без текста вокруг.]"
                )
                continue

            self.last_raw = raw
            observation, problems = parse_observation(raw, message)
            if problems and not observation.signals and observation.focus_answered == "none":
                last_error = problems
                user_input = (
                    f"{base_input}\n\n[Система: предыдущий ответ невалиден — {problems}. "
                    f"Верни СТРОГО JSON по схеме, без текста вокруг.]"
                )
                continue

            self.last_problems = problems
            return observation

        raise AssistantError(
            f"Наблюдатель не вернул валидное Observation за {self._max_attempts} попытки: {last_error}"
        )

    @staticmethod
    def _build_input(vacancy_context: str, state: dict, message: str) -> str:
        state_json = json.dumps(state_for_prompt(state), ensure_ascii=False, indent=2)
        return (
            "== КОНТЕКСТ ВАКАНСИИ ==\n"
            f"{vacancy_context}\n\n"
            "== STATE (накопленное состояние) ==\n"
            f"{state_json}\n\n"
            "== НОВОЕ СООБЩЕНИЕ КАНДИДАТА ==\n"
            f"{message}"
        )

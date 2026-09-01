"""Наблюдатель — «уши». Один вызов LLM: (контекст + STATE + сообщение) → `Observation`.

Замена прежнего Аналитика. Отличий по существу три:

  1. на вход идёт **проекция** состояния (`core.state_for_prompt`), а не state целиком: служебных
     полей модель не видит, поэтому и абзаца «их ведёт код, игнорируй» в промпте больше нет;
  2. на выход — наблюдение, а не решение, и валидация мягкая: жёстко обязательны два поля, остальное
     чинится дефолтом. Ошибка в одном блоке наблюдения не должна стоить трёх перегенераций хода;
  3. параметра `note` нет. Он существовал только ради перерешивания хода — второго вызова Аналитика
     в том же ходе. Такого вызова больше нет.
"""

import json
from typing import Any

from qa_harness.core import accumulate_usage, blank_usage

from ..errors import AssistantError
from .core import state_for_prompt
from .observation import Observation, parse_observation

MAX_ATTEMPTS = 3


class ScreeningObserver:
    """Один вызов LLM → валидированное `Observation`."""

    def __init__(self, prompt_client: Any, *, max_attempts: int = MAX_ATTEMPTS) -> None:
        self._client = prompt_client
        self._max_attempts = max_attempts
        self.last_usage: dict = blank_usage()
        self.last_raw: Any = None          # что вернула модель до валидации — для разбора глазами
        self.last_problems: str = ""       # претензии валидатора; НЕ повод перегенерировать

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
                # Жёсткий случай: контракт нарушен целиком (не объект / signals не список).
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

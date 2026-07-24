"""Интервьюер split-скрининга — «рот».

Порт `ScreeningInterviewer` (tgApi, HEAD e733095). Формулирует РОВНО ОДНО сообщение
кандидату по инструкции Аналитика; решений по существу не принимает, END не пишет.
Stateful: работает в OpenAI-conversation (история там), системный текст подаётся через
`instructions=`, а инструкция хода + сообщение кандидата — во `input` (как ждёт промпт).

Адаптация под ai-agents: spec (`screening_interviewer` из пакета `prompts`) и OpenAI-клиент
инжектятся, вызов — обычный `responses.create(conversation=..., instructions=..., input=...)`,
как в local-режиме domain/screening/conversation.py. Возвращаем (text, usage) — QA нужен учёт.
"""

from typing import Any


class ScreeningInterviewer:
    """Один ход: (instruction, message) в заданном conversation → одно сообщение кандидату."""

    def __init__(self, spec: Any, client: Any) -> None:
        self._spec = spec
        self._client = client

    def run(self, conversation_id: str, instruction: str, message: str) -> tuple[str, Any]:
        """Вернуть (text_сообщения, usage)."""
        user_input = self._build_turn(instruction, message)
        kwargs: dict[str, Any] = {
            "model": self._spec.model,
            "conversation": conversation_id,
            "input": user_input,
            "instructions": self._spec.system_text,  # системный промпт — на каждом ходу (как stored)
            "text": {"format": self._spec.text_format},
        }
        # None => «не задано в config.yaml» — параметр не передаём (как у потребителя пакета).
        for attr in ("temperature", "top_p", "max_output_tokens", "store"):
            val = getattr(self._spec, attr, None)
            if val is not None:
                kwargs[attr] = val
        resp = self._client.responses.create(**kwargs)
        text = (getattr(resp, "output_text", "") or "").strip()
        return text, getattr(resp, "usage", None)

    @staticmethod
    def _build_turn(instruction: str, message: str) -> str:
        parts = []
        if message:
            parts.append(f"[Сообщение кандидата]: {message}")
        if instruction:
            parts.append(f"[Внутренняя инструкция]: {instruction}")
        return "\n\n".join(parts)

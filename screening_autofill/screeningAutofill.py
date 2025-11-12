from __future__ import annotations

import json
import os
from typing import Any, Optional

import openai


class AssistantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class BaseAssistant:
    def __init__(
        self,
        system_prompt: str,
        json_schema: Optional[dict] = None,
        prompt_version: Optional[int] = None,
    ):
        if json_schema is None:
            json_schema = {"type": "text"}
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.system_prompt = system_prompt
        self.system_prompt_version = prompt_version
        self.json_schema = json_schema
        self.last_usage: Optional[Any] = None

    def _prompt_payload(self) -> dict[str, Any]:
        payload = {"id": self.system_prompt}
        if self.system_prompt_version is not None:
            payload["version"] = str(self.system_prompt_version)
        return payload

    def respond(self, user_input: str) -> dict[str, Any]:
        try:
            response = openai.responses.create(
                prompt=self._prompt_payload(),
                input=user_input,
                text={"format": self.json_schema},
            )
            self.last_usage = getattr(response, "usage", None)
            response_text = response.output_text
            if not response_text:
                raise AssistantError("Ответ от ассистента пустой")
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise AssistantError(
                    f"Ошибка парсинга JSON: {exc!r}\nОтвет ассистента: {response_text}"
                )
        except Exception as exc:
            raise AssistantError(f"Ошибка при получении ответа: {exc!r}")


class ScreeningAutofill(BaseAssistant):
    DEFAULT_PROMPT = "pmpt_68cbf36344948194ab74e4c48875b2510e0d6b5f0cbf6902"

    def __init__(self, prompt_id: Optional[str] = None, prompt_version: Optional[int] = None):
        json_schema = {
            "type": "json_schema",
            "name": "screening",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "preferred_location": {
                        "type": "string",
                        "description": "Город, регион или страна, где кандидат хочет работать.",
                    },
                    "min_salary": {
                        "type": "string",
                        "description": "Минимальные зарплатные ожидания кандидата.",
                    },
                    "max_salary": {
                        "type": "string",
                        "description": "Максимальные зарплатные ожидания кандидата.",
                    },
                    "additional_info": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "Вопрос рекрутера.",
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "Ответ кандидата.",
                                },
                            },
                            "required": ["question", "answer"],
                            "additionalProperties": False,
                        },
                        "description": "Список вопросов и ответов кандидата на них.",
                    },
                    "work_format": {
                        "type": "string",
                        "description": "Тип занятости : 'remote', 'office', 'hybrid'.",
                    },
                },
                "required": [
                    "preferred_location",
                    "min_salary",
                    "max_salary",
                    "additional_info",
                    "work_format",
                ],
                "additionalProperties": False,
            },
        }
        super().__init__(
            system_prompt=prompt_id or self.DEFAULT_PROMPT,
            json_schema=json_schema,
            prompt_version=prompt_version,
        )

    def run(self, dialog: str) -> dict[str, Any]:
        user_prompt = (
            f"{dialog}\n"
            "Fill the screening form following this schema strictly as JSON only."
        )
        result = self.respond(user_prompt)
        if not isinstance(result, dict):
            raise AssistantError(f"Ассистент вернул некорректный результат: {result}")
        return result

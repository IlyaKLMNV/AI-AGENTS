import os
import openai
import json
from typing import Any


class AssistantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class BaseAssistant:

    def __init__(self,
                 system_prompt: str,
                 json_schema: dict = None):
        if json_schema is None:
            json_schema = {"type": "text"}
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def respond(self, user_input: str) -> dict[str, Any]:
        try:
            response = openai.responses.create(
                prompt={
                    "id": self.system_prompt,
                },
                input=user_input,
                text={"format": self.json_schema}
            )
            response_text = response.output_text
            if not response_text:
                raise AssistantError('Ответ от ассистента пустой')
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                raise AssistantError(f'Ошибка парсинга JSON: {e!r}\nОтвет ассистента: {response_text}')
        except Exception as e:
            raise AssistantError(f'Ошибка при получении ответа: {e!r}')


class ScreeningAutofill(BaseAssistant):
    def __init__(self):
        system_prompt = "pmpt_68cbf36344948194ab74e4c48875b2510e0d6b5f0cbf6902"
        json_schema = {
            "type": "json_schema",
            "name": "screening",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "preferred_location": {
                        "type": "string",
                        "description": "Город, регион или страна, где кандидат хочет работать."
                    },
                    "min_salary": {
                        "type": "string",
                        "description": "Минимальные зарплатные ожидания кандидата."
                    },
                    "max_salary": {
                        "type": "string",
                        "description": "Максимальные зарплатные ожидания кандидата."
                    },
                    "additional_info": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "Вопрос рекрутера."
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "Ответ кандидата."
                                }
                            },
                            "required": ["question", "answer"],
                            "additionalProperties": False
                        },
                        "description": "Список вопросов и ответов кандидата на них."
                    },
                    "work_format": {
                        "type": "string",
                        "description": "Тип занятости : 'remote', 'office', 'hybrid'."
                    }},
                "required": [
                    "preferred_location",
                    "min_salary",
                    "max_salary",
                    "additional_info",
                    "work_format"
                ],
                "additionalProperties": False
            }
        }

        super().__init__(system_prompt=system_prompt, json_schema=json_schema)

    def run(self, dialog: str) -> dict[str, Any]:
        user_prompt = (
            f"{dialog}\n"
            "Fill the screening form following this schema strictly as JSON only."
        )

        result = self.respond(user_prompt)

        if not isinstance(result, dict):
            raise AssistantError(f'Ассистент вернул некорректный результат: {result}')

        return result

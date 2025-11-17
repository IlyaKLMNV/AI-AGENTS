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

    def respond(self, user_input: str) -> str:
        try:
            response = openai.responses.create(
                prompt=self._prompt_payload(),
                input=user_input,
                text={"format": self.json_schema},
            )
            self.last_usage = getattr(response, "usage", None)
            if not response.output_text:
                raise AssistantError("Ответ от ассистента пустой")
            return response.output_text
        except Exception as e:
            raise AssistantError(f"Ошибка при получении ответа: {e!r}")


class ClassifierAssistant(BaseAssistant):
    DEFAULT_PROMPT = "pmpt_68e8b47991f0819094f05d51eb5780a10ff88c389be96726"

    def __init__(
        self,
        prompt_id: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ):
        super().__init__(prompt_id or self.DEFAULT_PROMPT, None, prompt_version)

    def run(self, user_input: str) -> str:
        return self.respond(user_input)

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI


def _extract_json_substring(text: str) -> Optional[str]:
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()

    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return text[start : end + 1].strip()

    return None


def safe_json_loads(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty json text")
    try:
        return json.loads(raw)
    except Exception:
        extracted = _extract_json_substring(raw)
        if not extracted:
            raise
        return json.loads(extracted)


class ScreeningAutofillPromptClient:
    def __init__(self, prompt_id: str, prompt_version: Optional[str | int]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version is not None:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None
        self.last_output_text: str = ""

    def run(self, dialogue: str) -> Dict[str, Any]:
        payload = "\n".join(
            [
                "Fill the screening form based on the dialogue below.",
                "",
                (dialogue or "").strip(),
            ]
        ).strip()

        response = self.client.responses.create(
            prompt=self.prompt,
            input=payload,
        )
        self.last_usage = getattr(response, "usage", None)
        self.last_output_text = (getattr(response, "output_text", "") or "").strip()
        parsed = safe_json_loads(self.last_output_text)
        if not isinstance(parsed, dict):
            raise ValueError("screening_autofill did not return a JSON object")
        return parsed

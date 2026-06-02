"""Клиенты OpenAI Responses API.

Поправка критика (docs/REFACTOR_PLAN.md §4): НЕ безусловный singleton — клиент
кэшируется по (base_url, timeout), а роли разведены на два класса:
- `StoredPromptClient` — для промпта-под-тестом (prompt={id, version});
- `ModelClient` — для генератора/судьи (model=...).

Оба принимают опциональный `client=` для подмены в offline/replay/тестах
(никакой сети). openai импортируется лениво, чтобы `import qa_harness.core`
не требовал установленного openai в чисто-юнит-окружении.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

_CLIENTS: Dict[Tuple[Optional[str], Optional[float]], Any] = {}


def get_client(*, base_url: Optional[str] = None, timeout: Optional[float] = None) -> Any:
    """Вернуть OpenAI-клиент, кэшированный по (base_url, timeout)."""
    key = (base_url, timeout)
    if key not in _CLIENTS:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # ленивый импорт

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url or os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url or os.environ.get("OPENAI_BASE_URL")
        if timeout is not None:
            kwargs["timeout"] = timeout
        _CLIENTS[key] = OpenAI(**kwargs)
    return _CLIENTS[key]


def _text_and_usage(resp: Any) -> Tuple[str, Any]:
    return (getattr(resp, "output_text", "") or "").strip(), getattr(resp, "usage", None)


class StoredPromptClient:
    """Вызов stored-промпта (prompt-под-тестом) через Responses API."""

    def __init__(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
        *,
        client: Any = None,
        text_format: Optional[dict] = None,
    ) -> None:
        self._client = client
        self._text_format = text_format
        self._prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self._prompt["version"] = str(prompt_version)

    def run(self, input_text: str) -> Tuple[str, Any]:
        """Вернуть (output_text, usage)."""
        client = self._client or get_client()
        kwargs: Dict[str, Any] = {"prompt": self._prompt, "input": input_text}
        if self._text_format is not None:
            kwargs["text"] = self._text_format
        return _text_and_usage(client.responses.create(**kwargs))


class ModelClient:
    """Вызов модели без stored-промпта (генератор / судья)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = client

    def create(self, input_text: str) -> Tuple[str, Any]:
        """Вернуть (output_text, usage)."""
        client = self._client or get_client(base_url=self._base_url, timeout=self._timeout)
        return _text_and_usage(client.responses.create(model=self._model, input=input_text))

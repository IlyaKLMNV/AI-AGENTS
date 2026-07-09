"""Клиенты OpenAI Responses API.

Поправка критика (docs/REFACTOR_PLAN.md §4): НЕ безусловный singleton — клиент
кэшируется по (base_url, timeout), а роли разведены на классы:
- `StoredPromptClient` — промпт-под-тестом из platform.openai.com (prompt={id, version});
- `LocalPromptClient` — промпт-под-тестом из пакета `prompts` (тело/параметры локально,
  вызывается как model=... + input=messages). Тот же контракт .run(input_text) -> (text, usage),
  поэтому раннеры не различают источники — выбор делает core.prompt_source.make_prompt_client;
- `ModelClient` — для генератора/судьи (model=...).

Все принимают опциональный `client=` для подмены в offline/replay/тестах
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


class LocalPromptClient:
    """Вызов промпта-под-тестом из пакета `prompts` (локальный источник).

    Тело/параметры берутся из PromptSpec (system.md + config.yaml нужной версии), а сам вызов —
    обычный responses.create(model=..., input=messages, ...), как в проде-потребителе пакета.
    Контракт .run(input_text) -> (text, usage) идентичен StoredPromptClient.

    args в build_input НЕ передаём: раннеры уже кладут весь контекст в input_text (как и в
    stored-режиме), а без args рендер не вызывает str.format — литеральные {} в теле безопасны.
    """

    def __init__(
        self,
        component: str,
        version: Optional[str] = None,
        *,
        client: Any = None,
        text_format: Optional[dict] = None,
    ) -> None:
        from qa_harness.core.prompt_source import load_local_spec  # ленивый: не тянуть prompts зря

        self._spec = load_local_spec(component, version)
        self._client = client
        self._text_format = text_format  # полный text=... (как у StoredPromptClient); None -> из spec

    @property
    def spec(self) -> Any:
        return self._spec

    def run(self, input_text: str) -> Tuple[str, Any]:
        """Вернуть (output_text, usage)."""
        client = self._client or get_client()
        messages = self._spec.build_input(user_input=input_text)
        kwargs: Dict[str, Any] = {"model": self._spec.model, "input": messages}
        # None означает «не задано в config.yaml» — параметр не передаём (как у потребителя пакета).
        for attr in ("temperature", "top_p", "max_output_tokens", "store"):
            val = getattr(self._spec, attr, None)
            if val is not None:
                kwargs[attr] = val
        kwargs["text"] = self._text_format if self._text_format is not None else {"format": self._spec.text_format}
        return _text_and_usage(client.responses.create(**kwargs))


class ModelClient:
    """Вызов модели без stored-промпта (генератор / судья)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._temperature = temperature
        self._client = client

    def create(self, input_text: str) -> Tuple[str, Any]:
        """Вернуть (output_text, usage)."""
        client = self._client or get_client(base_url=self._base_url, timeout=self._timeout)
        kwargs: Dict[str, Any] = {"model": self._model, "input": input_text}
        if self._temperature is not None:  # вариативность генерации (судья — без temperature)
            kwargs["temperature"] = self._temperature
        return _text_and_usage(client.responses.create(**kwargs))

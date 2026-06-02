"""Базовый генератор: instruction + payload -> LLM -> parse, с учётом токенов.

Конкретный генератор реализует три метода; клиент (с методом .create(input)->(text,usage),
например core.llm_client.ModelClient) и накопление usage — в базе. Клиент инъектируется,
поэтому генератор тестируется офлайн с фейковым клиентом.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qa_harness.core.usage import accumulate_usage, blank_usage


class Generator(ABC):
    def __init__(self, client: Any) -> None:
        self._client = client
        self.usage = blank_usage()

    @abstractmethod
    def instruction(self, spec: Any) -> str: ...

    @abstractmethod
    def payload(self, spec: Any) -> str: ...

    @abstractmethod
    def parse(self, text: str) -> Any: ...

    def generate(self, spec: Any) -> Any:
        """Один вызов модели: вернуть распарсенный результат (usage копится в self.usage)."""
        text, usage = self._client.create(self.instruction(spec) + "\n\n" + self.payload(spec))
        accumulate_usage(self.usage, usage)
        return self.parse(text)

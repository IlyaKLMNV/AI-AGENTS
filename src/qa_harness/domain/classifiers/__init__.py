"""Классификаторы сообщений кандидата (тестируемый компонент)."""

from .message import HeuristicMessageClassifier, StoredPromptMessageClassifier

__all__ = ["StoredPromptMessageClassifier", "HeuristicMessageClassifier"]

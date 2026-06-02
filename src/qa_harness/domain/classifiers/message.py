"""Классификатор сообщения кандидата в один из CLASSES.

- StoredPromptMessageClassifier — онлайн, через stored-промпт message_classifier
  (тестируемый компонент). Поведение перенесено из старого MessageClassifierRunner.
- HeuristicMessageClassifier — офлайн, детерминированный (regex), без сети.
  Нужен для CI-смоука и демонстрации сквозного потока без жжения токенов.
  Это НЕ тестируемый компонент, а заглушка-источник предсказаний для офлайна.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ..judge.label_judge import CLASSES, extract_label
from ..text.message_markers import (
    ACCEPTANCE_PATTERNS,
    DECLINE_PATTERNS,
    HUMAN_NEEDED_PATTERNS,
    REASON_PATTERNS,
    has_any_pattern,
)

ClassifyResult = Tuple[Optional[str], str, Any]  # (label, raw_output, usage)


class StoredPromptMessageClassifier:
    """Онлайн-классификатор через stored-промпт (qa_harness.core.llm_client.StoredPromptClient)."""

    def __init__(self, client: Any, labels: Sequence[str] = CLASSES) -> None:
        self._client = client
        self._labels = tuple(labels)

    def classify(self, message: str) -> ClassifyResult:
        raw, usage = self._client.run(message.strip())
        label = extract_label(raw, self._labels)
        if label not in self._labels:
            raise ValueError(f"message_classifier returned invalid output: {raw!r}")
        return label, raw, usage


# --- офлайн-эвристика (детерминированная, без сети) ------------------------------
# Использует общие маркеры из domain/text/message_markers.py (единый источник).


class HeuristicMessageClassifier:
    """Офлайн-детерминированная эвристика (для CI/демо, не тестируемый компонент)."""

    def classify(self, message: str) -> ClassifyResult:
        text = message or ""
        if has_any_pattern(text, HUMAN_NEEDED_PATTERNS):
            label = "human_needed"
        elif has_any_pattern(text, DECLINE_PATTERNS):
            label = "reason_farewell" if has_any_pattern(text, REASON_PATTERNS) else "no_reason"
        elif has_any_pattern(text, ACCEPTANCE_PATTERNS):
            label = "acceptance"
        else:
            label = "acceptance"  # нейтральный fallback
        return label, label, None

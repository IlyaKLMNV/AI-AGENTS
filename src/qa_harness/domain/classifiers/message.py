"""Классификатор сообщения кандидата в один из CLASSES.

- StoredPromptMessageClassifier — онлайн, через stored-промпт message_classifier
  (тестируемый компонент). Поведение перенесено из старого MessageClassifierRunner.
- HeuristicMessageClassifier — офлайн, детерминированный (regex), без сети.
  Нужен для CI-смоука и демонстрации сквозного потока без жжения токенов.
  Это НЕ тестируемый компонент, а заглушка-источник предсказаний для офлайна.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Tuple

from ..judge.label_judge import CLASSES, extract_label

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

_HUMAN_NEEDED = (
    r"\bстранн", r"\bчто\s+за\s+ерунд", r"\bмошенн", r"\bразвод", r"\bскиньте\b",
    r"\bоткуда\s+нашли\s+контакт\b", r"\bзачем\s+мне\s+тратить\s+время\b",
    r"\bбред\b", r"\bхрень\b", r"\bено[тт]\b", r"[🦝😕🤨]",
)
_DECLINE = (
    r"\bне\s+интерес", r"\bне\s+рассматрива", r"\bне\s+подходит", r"\bвынужден\s+отказ",
    r"\bоткаж", r"\bотказ", r"\bне\s+готов", r"\bне\s+смогу", r"\bнет,\s*спасибо\b",
)
_REASON = (
    r"\bпотому\s+что\b", r"\bтак\s+как\b", r"\bпоскольку\b", r"\bоффер", r"\bзарплат",
    r"\bформат", r"\bофис", r"\bгибрид", r"\bудален", r"\bлокац", r"\bпереезд",
    r"\bстек", r"\bсфера", r"\bработаю\b", r"\bуже\b",
)
_ACCEPTANCE = (
    r"\bинтерес", r"\bваканси", r"\bподскажите\b", r"\bрасскажите\b", r"\bссылк",
    r"\bописани[ея]\b", r"\bкоманд", r"\bзадач", r"\bсозвон", r"\bготов\s+обсудить\b",
)


def _any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


class HeuristicMessageClassifier:
    """Офлайн-детерминированная эвристика (для CI/демо, не тестируемый компонент)."""

    def classify(self, message: str) -> ClassifyResult:
        text = message or ""
        if _any(text, _HUMAN_NEEDED):
            label = "human_needed"
        elif _any(text, _DECLINE):
            label = "reason_farewell" if _any(text, _REASON) else "no_reason"
        elif _any(text, _ACCEPTANCE):
            label = "acceptance"
        else:
            label = "acceptance"  # нейтральный fallback
        return label, label, None

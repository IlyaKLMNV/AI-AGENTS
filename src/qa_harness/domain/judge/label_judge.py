"""LabelJudge — точное сравнение предсказанной метки с целевой.

Используется классификаторами (message_classifier, verdict_classifier).
extract_label перенесён из старого `_extract_label` (regex по словам-меткам,
case-insensitive), но обобщён на произвольный набор меток.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .base import Verdict

# Метки классификатора сообщений кандидата (ground-truth классы).
CLASSES = ("reason_farewell", "no_reason", "acceptance", "human_needed")


def extract_label(text: str, labels: Sequence[str] = CLASSES) -> Optional[str]:
    """Найти в тексте одну из меток (целым словом, без учёта регистра)."""
    t = (text or "").strip().lower()
    pattern = r"\b(" + "|".join(re.escape(str(label).lower()) for label in labels) + r")\b"
    m = re.search(pattern, t)
    return m.group(1) if m else None


class LabelJudge:
    """Сравнивает предсказанную метку с целевой -> Verdict(passed)."""

    def __init__(self, labels: Sequence[str] = CLASSES) -> None:
        self.labels = tuple(labels)

    def evaluate(self, predicted: Optional[str], target: str, *, evaluator: str = "label_match") -> Verdict:
        passed = predicted == target
        reason_codes = [] if passed else [f"misclassified->{predicted}"]
        return Verdict(
            passed=passed,
            evaluator=evaluator,
            score=1.0 if passed else 0.0,
            max_score=1.0,
            reason_codes=reason_codes,
        )

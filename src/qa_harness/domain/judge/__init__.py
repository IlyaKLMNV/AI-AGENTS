"""Судьи: единый протокол Verdict + конкретные реализации."""

from .base import Verdict
from .label_judge import CLASSES, LabelJudge, extract_label

__all__ = ["Verdict", "LabelJudge", "extract_label", "CLASSES"]

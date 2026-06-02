"""Текстовые примитивы домена (маркеры, детекторы)."""

from .message_markers import (
    ACCEPTANCE_PATTERNS,
    DECLINE_PATTERNS,
    HUMAN_NEEDED_PATTERNS,
    REASON_PATTERNS,
    has_any_pattern,
)

__all__ = [
    "DECLINE_PATTERNS",
    "REASON_PATTERNS",
    "ACCEPTANCE_PATTERNS",
    "HUMAN_NEEDED_PATTERNS",
    "has_any_pattern",
]

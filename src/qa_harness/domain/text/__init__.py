"""Текстовые примитивы домена (маркеры, детекторы, формат диалога)."""

from .dialogue import CANDIDATE_PREFIX, RECRUITER_PREFIX, speaker_for_line, split_dialogue_lines
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
    "RECRUITER_PREFIX",
    "CANDIDATE_PREFIX",
    "split_dialogue_lines",
    "speaker_for_line",
]

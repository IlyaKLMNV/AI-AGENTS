"""Доменная оценка гардрейлов screening_assistant: нет self_answer / repeated_questions / premature_end.

Качество промпта — отсутствие нарушений в ответах ассистента (LLM-судья + эвристики-фолбэк). Сам
мультитёрн-разговор ведёт qa_harness.domain.screening.ScreeningConversation; здесь — оценка реплик.
"""

from .judge import ALLOWED_TOPICS, EVAL_INSTRUCTION, GuardrailJudge, GuardrailVerdict
from .detectors import (
    has_questions_in_reply,
    heuristic_premature_end,
    heuristic_repeated_questions,
    heuristic_self_answer,
)
from .cases import GoldenCase, OfflineTurn, load_golden

__all__ = [
    "GuardrailJudge",
    "GuardrailVerdict",
    "EVAL_INSTRUCTION",
    "ALLOWED_TOPICS",
    "has_questions_in_reply",
    "heuristic_self_answer",
    "heuristic_repeated_questions",
    "heuristic_premature_end",
    "GoldenCase",
    "OfflineTurn",
    "load_golden",
]

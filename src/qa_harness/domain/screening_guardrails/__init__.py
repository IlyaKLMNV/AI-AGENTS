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
from .personas import DEFAULT_PERSONAS, load_personas, persona_constraints

__all__ = [
    "load_personas",
    "persona_constraints",
    "DEFAULT_PERSONAS",
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

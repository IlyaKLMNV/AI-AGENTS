"""Доменная оценка промпта first_touch_event (приглашение на фиксированное мероприятие VK JT Go).

Качество = LLM-судья на эталон (missing/hallucinated/forbidden_claims) + эвристики (приветствие по имени,
финальный вопрос про регистрацию, нет лишних чисел). Офлайн использует эвристику фактов вместо судьи.
"""

from .reference import EVAL_INSTRUCTION, EXPECTED_FACT_KEYS, FORBIDDEN_DETAILS, REFERENCE_FACTS
from .judge import EventJudge, EventVerdict
from .checks import extra_numbers, facts_present_heuristic, final_question_ok, greeting_ok
from .cases import GoldenCase, load_golden

__all__ = [
    "EventJudge",
    "EventVerdict",
    "EVAL_INSTRUCTION",
    "EXPECTED_FACT_KEYS",
    "REFERENCE_FACTS",
    "FORBIDDEN_DETAILS",
    "greeting_ok",
    "final_question_ok",
    "extra_numbers",
    "facts_present_heuristic",
    "GoldenCase",
    "load_golden",
]

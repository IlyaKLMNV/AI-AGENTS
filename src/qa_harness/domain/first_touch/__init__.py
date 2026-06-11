"""Доменная оценка промпта first_touch (генерация первого касания).

Качество = LLM-судья фактов (наличие ожидаемых фактов, нет выдуманных, есть CTA-вопрос) + эвристики
(нет лишних чисел, при company_hidden — нет утечки названия). Офлайн использует эвристику вместо судьи.
"""

from .judge import EVAL_INSTRUCTION, FactJudge, FactVerdict
from .checks import company_name_leaked, extra_numbers, facts_present_heuristic, forbidden_phrases
from .cases import GoldenCase, load_golden

__all__ = [
    "FactJudge",
    "FactVerdict",
    "EVAL_INSTRUCTION",
    "extra_numbers",
    "company_name_leaked",
    "facts_present_heuristic",
    "forbidden_phrases",
    "GoldenCase",
    "load_golden",
]

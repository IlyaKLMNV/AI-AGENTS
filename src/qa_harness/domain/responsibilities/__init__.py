"""Доменная оценка промпта responsibilities_parser (текст вакансии → ключевые термины).

Качество = контракт формы (1..5 коротких терминов, без чисел/запятых, без дублей) + семантика по golden
(ожидаемые термины извлечены, запрещённые — нет). Заземление в тексте — отдельный сигнал-предупреждение.
"""

from .contract import check_contract, find_duplicates, parse_keywords, validate_item_format
from .semantic import check_semantics, grounding_misses
from .cases import GoldenCase, load_golden

__all__ = [
    "parse_keywords",
    "validate_item_format",
    "find_duplicates",
    "check_contract",
    "check_semantics",
    "grounding_misses",
    "GoldenCase",
    "load_golden",
]

"""Доменная оценка промпта one_line_search_query_builder (one-line boolean запрос).

Качество билдера = формат (checks) + отсутствие утечек (checks) + семантика по golden
(semantic). Бэкенд-ретрив и извлечение — стадии-инфо в раннере, не здесь.
"""

from .checks import LEAKAGE_PATTERNS, build_query_checks, detect_leakage
from .semantic import check_query_semantics
from .cases import GoldenCase, load_golden

__all__ = [
    "build_query_checks",
    "detect_leakage",
    "LEAKAGE_PATTERNS",
    "check_query_semantics",
    "GoldenCase",
    "load_golden",
]

"""Доменная оценка промпта responsibilities_parser (текст вакансии → 0..5 требований-предложений).

Качество = контракт формы (0..5 требований-предложений, не keyword-подобны, без объединения критериев,
без дублей) + семантика по golden (обязательные критерии покрыты внутри требований, запрещённые — нет;
поддержка `expect_empty`). Заземление anchor-термов в тексте — отдельный сигнал-предупреждение.
"""

from .contract import (
    check_contract,
    find_duplicates,
    is_candidate_fact_style,
    is_keyword_like,
    is_multi_criteria,
    parse_keywords,
    validate_requirement,
)
from .semantic import check_semantics, grounding_missing_anchors
from .cases import GoldenCase, load_golden

__all__ = [
    "parse_keywords",
    "validate_requirement",
    "is_keyword_like",
    "is_candidate_fact_style",
    "is_multi_criteria",
    "find_duplicates",
    "check_contract",
    "check_semantics",
    "grounding_missing_anchors",
    "GoldenCase",
    "load_golden",
]

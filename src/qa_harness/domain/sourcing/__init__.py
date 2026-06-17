"""Доменная оценка промпта sourcing_assistant (кандидат ↔ требования вакансии).

Качество = контракт вывода (массив 1:1 к требованиям, форма элементов) — семантики нет
(кандидаты живые, без разметки). Backend-поиск и сборку payload оркестрирует раннер.
"""

from .contract import ALLOWED_KEYS, check_contract, parse_sourcing_output
from .build import REQUIREMENTS_SOURCES, build_candidate_profile, requirements_from_cdm
from .cases import (
    GoldenScoreCase,
    OfflineCandidate,
    OfflineCase,
    SearchVacancy,
    load_golden_score,
    load_offline_cases,
    load_search_vacancies,
)
from .semantic import check_passed_labels, comment_inconsistencies

__all__ = [
    "parse_sourcing_output",
    "check_contract",
    "ALLOWED_KEYS",
    "check_passed_labels",
    "comment_inconsistencies",
    "requirements_from_cdm",
    "build_candidate_profile",
    "REQUIREMENTS_SOURCES",
    "OfflineCase",
    "OfflineCandidate",
    "load_offline_cases",
    "GoldenScoreCase",
    "load_golden_score",
    "SearchVacancy",
    "load_search_vacancies",
]

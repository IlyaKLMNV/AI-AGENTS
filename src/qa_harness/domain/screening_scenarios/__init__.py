"""Доменная оценка промпта screening_assistant по сценариям из CSV.

Сценарии (golden из CSV с expected_behavior) гоняются через живой мультитёрн
(qa_harness.domain.screening.ScreeningConversation); ScenarioJudge (LLM) судит транскрипт против
ожидаемого поведения. Заменяет hardcoded-эвристики легаси.
"""

from .cases import Scenario, extract_candidate_examples, load_scenarios, parse_scenario_indices
from .constraints import DEFAULT_CONSTRAINTS, constraints_for, load_constraints
from .judge import END_MARKER, EVAL_INSTRUCTION, ScenarioJudge, ScenarioVerdict
from .vacancies import load_vacancies, vacancy_for

__all__ = [
    "Scenario",
    "load_scenarios",
    "extract_candidate_examples",
    "parse_scenario_indices",
    "ScenarioJudge",
    "ScenarioVerdict",
    "EVAL_INSTRUCTION",
    "END_MARKER",
    "load_constraints",
    "constraints_for",
    "DEFAULT_CONSTRAINTS",
    "load_vacancies",
    "vacancy_for",
]

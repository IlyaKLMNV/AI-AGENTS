"""Генераторы синтетических тестовых данных (сообщения, диалоги)."""

from .base import Generator
from .engine import Attempt, GenResult, GenerationPolicy, generate_valid
from .candidate_agent import CandidateAgent, CandidateConstraints, validate_candidate_turn
from .autofill_gen import AutofillDialogueGenerator, AutofillSpec, WORK_FORMATS, validate_autofill_dialogue
from .vacancy_gen import DOMAINS, SENIORITIES, VacancyGenerator, VacancySpec
from .responsibilities_gen import (
    CONDITIONS_NOISE,
    SOFT_NOISE,
    TECH_VOCAB,
    ResponsibilitiesGenerator,
    ResponsibilitiesSpec,
)
from .profile_gen import CandidateProfileGenerator, CandidateProfileSpec
from .variety import VariantSampler, VariantStyle
from .dialogue_gen import DialogueGenerator, DialogueSpec, validate_generated_dialogue
from .dialogue_specs import SCENARIO_HINTS_BY_VERDICT, pick_verdict_hint
from .message_gen import CandidateMessageGenerator, MessageSpec, validate_candidate_message
from .message_specs import SCENARIO_HINTS_BY_CLASS, pick_scenario_hint

__all__ = [
    "Generator",
    "generate_valid",
    "GenerationPolicy",
    "GenResult",
    "Attempt",
    "CandidateAgent",
    "CandidateConstraints",
    "validate_candidate_turn",
    "AutofillDialogueGenerator",
    "AutofillSpec",
    "WORK_FORMATS",
    "validate_autofill_dialogue",
    "VacancyGenerator",
    "VacancySpec",
    "DOMAINS",
    "SENIORITIES",
    "ResponsibilitiesGenerator",
    "ResponsibilitiesSpec",
    "TECH_VOCAB",
    "SOFT_NOISE",
    "CONDITIONS_NOISE",
    "CandidateProfileGenerator",
    "CandidateProfileSpec",
    "VariantSampler",
    "VariantStyle",
    "CandidateMessageGenerator",
    "MessageSpec",
    "validate_candidate_message",
    "SCENARIO_HINTS_BY_CLASS",
    "pick_scenario_hint",
    "DialogueGenerator",
    "DialogueSpec",
    "validate_generated_dialogue",
    "SCENARIO_HINTS_BY_VERDICT",
    "pick_verdict_hint",
]

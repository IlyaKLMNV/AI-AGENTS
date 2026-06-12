"""Генераторы синтетических тестовых данных (сообщения, диалоги)."""

from .base import Generator
from .engine import Attempt, GenResult, GenerationPolicy, generate_valid
from .candidate_agent import CandidateAgent, CandidateConstraints, validate_candidate_turn
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

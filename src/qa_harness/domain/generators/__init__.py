"""Генераторы синтетических тестовых данных (сообщения, диалоги)."""

from .base import Generator
from .message_gen import CandidateMessageGenerator, MessageSpec, validate_candidate_message
from .message_specs import SCENARIO_HINTS_BY_CLASS, pick_scenario_hint

__all__ = [
    "Generator",
    "CandidateMessageGenerator",
    "MessageSpec",
    "validate_candidate_message",
    "SCENARIO_HINTS_BY_CLASS",
    "pick_scenario_hint",
]

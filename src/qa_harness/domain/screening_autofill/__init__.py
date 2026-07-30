"""Доменная оценка промпта screening_autofill (диалог → форма скрининга).

Качество = контракт формы (contract: ключи/типы/enum/digits) + golden expect-subset (semantic) +
анти-утечка в additional_info (semantic). Дороги к семантике — только по golden, без LLM-судьи.
"""

from .contract import REQUIRED_KEYS, WORK_FORMATS, parse_form, validate_schema
from .semantic import NONEMPTY, additional_info_leaks, check_expect
from .cases import GoldenCase, load_golden

__all__ = [
    "parse_form",
    "validate_schema",
    "WORK_FORMATS",
    "REQUIRED_KEYS",
    "check_expect",
    "additional_info_leaks",
    "NONEMPTY",
    "GoldenCase",
    "load_golden",
]

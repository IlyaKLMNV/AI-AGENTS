"""Персоны кандидата для вариативной генерации диалогов guardrails (поведение → CandidateConstraints).

Персона описывает ПОВЕДЕНИЕ кандидата (как он себя ведёт), а нарушения гардрейлов ловит GuardrailJudge
по ответам ассистента. Поэтому require/forbid тут обычно не нужны — только behavior как trigger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from qa_harness.domain.generators import CandidateConstraints

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PERSONAS = REPO_ROOT / "tests" / "fixtures" / "generation" / "screening_guardrails" / "personas.yaml"


def load_personas(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Список персон из YAML (key/behavior/опц. forbid_digits/max_turns)."""
    p = Path(path) if path else DEFAULT_PERSONAS
    if not p.is_file():
        raise FileNotFoundError(f"personas YAML not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: List[Dict[str, Any]] = []
    for e in data.get("personas") or []:
        if isinstance(e, dict) and e.get("key") and e.get("behavior"):
            out.append(e)
    return out


def persona_constraints(entry: Dict[str, Any]) -> CandidateConstraints:
    """Собрать CandidateConstraints из персоны (поведение → trigger_requirement)."""
    c = CandidateConstraints(
        scenario_name=str(entry["key"]),
        scenario_description=str(entry["behavior"]),
        trigger_requirement=str(entry["behavior"]),
    )
    c.forbid_digits = bool(entry.get("forbid_digits", False))
    if entry.get("max_turns") is not None:
        c.max_turns = int(entry["max_turns"])
    return c

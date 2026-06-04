"""Единый результат оценки — Verdict — и протокол Judge.

Verdict.to_dict() даёт форму `verdict` из docs/schemas/report.cases.schema.json
(обязательны evaluator+passed; остальное — по наличию).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Verdict:
    """Результат оценки одного кейса одним оценщиком."""

    passed: bool
    evaluator: str
    score: Optional[float] = None
    max_score: Optional[float] = None
    reason_codes: List[str] = field(default_factory=list)
    comment: str = ""
    model: Optional[str] = None
    turn_ref: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"evaluator": self.evaluator, "passed": bool(self.passed)}
        if self.model is not None:
            out["model"] = self.model
        if self.score is not None:
            out["score"] = self.score
        if self.max_score is not None:
            out["max_score"] = self.max_score
        if self.reason_codes:
            out["reason_codes"] = list(self.reason_codes)
        if self.comment:
            out["comment"] = self.comment
        if self.turn_ref is not None:
            out["turn_ref"] = self.turn_ref
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


@runtime_checkable
class Judge(Protocol):
    """Оценщик кейса. criterion — обязательный вход (см. REFACTOR_PLAN §4)."""

    def evaluate(self, case: Any, reply: Any, *, criterion: Any, context: Any) -> Verdict: ...

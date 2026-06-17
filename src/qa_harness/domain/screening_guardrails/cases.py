"""Курируемые golden-разговоры для screening_guardrails.

Источник — tests/fixtures/screening_guardrails/golden.yaml. Каждый кейс — разговор:
  candidate_name + candidate_turns (реплики кандидата для ЖИВОГО мультитёрна со screening_assistant);
  offline_turns — replay для `--offline` (каннные пары candidate/assistant_reply + флаг конца), на них
    гоняются эвристики-детекторы без сети. vacancy_info/recruiter_name берутся из дефолта раннера.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class OfflineTurn:
    candidate: str
    assistant_reply: str
    conversation_end: bool = False


@dataclass
class GoldenCase:
    name: str
    candidate_name: str
    candidate_turns: List[str] = field(default_factory=list)
    offline_turns: List[OfflineTurn] = field(default_factory=list)


def load_golden(path: Path) -> List[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("golden file must be a YAML list")
    out: List[GoldenCase] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"golden case #{i} must be a mapping")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate golden case name: {name}")
        seen.add(name)
        offline_turns = [
            OfflineTurn(
                candidate=str(t.get("candidate") or ""),
                assistant_reply=str(t.get("assistant_reply") or ""),
                conversation_end=bool(t.get("conversation_end")),
            )
            for t in (item.get("offline_turns") or [])
            if isinstance(t, dict)
        ]
        out.append(GoldenCase(
            name=name,
            candidate_name=str(item.get("candidate_name") or "Кандидат").strip(),
            candidate_turns=[str(x) for x in (item.get("candidate_turns") or [])],
            offline_turns=offline_turns,
        ))
    return out

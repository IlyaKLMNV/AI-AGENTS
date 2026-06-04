"""Офлайн-кейсы sourcing (replay) для `--offline`: вакансия + кандидаты с КАННЫМ выводом промпта.

Источник — tests/fixtures/sourcing_assistant/offline.yaml. Каждый кейс — вакансия (имя + 1..5
требований) и список кандидатов, у каждого `output` = заранее заданный ответ промпта (массив
объектов). Реплей гоняет контракт-чекер + subjects[] без сети — для дешёвой проверки плумбинга.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class OfflineCandidate:
    id: str
    output: List[Dict[str, Any]]  # каннный ответ промпта (массив объектов)


@dataclass
class OfflineCase:
    name: str
    requirements: List[str]
    candidates: List[OfflineCandidate] = field(default_factory=list)


def load_offline_cases(path: Path) -> List[OfflineCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("offline file must be a YAML list")
    out: List[OfflineCase] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"offline case #{i} must be a mapping")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate offline case name: {name}")
        seen.add(name)
        reqs = [str(r).strip() for r in (item.get("requirements") or []) if str(r).strip()]
        if not reqs:
            raise ValueError(f"offline case {name}: empty requirements")
        cands: List[OfflineCandidate] = []
        for j, c in enumerate(item.get("candidates") or [], start=1):
            if not isinstance(c, dict):
                raise ValueError(f"offline case {name}: candidate #{j} must be a mapping")
            cands.append(OfflineCandidate(id=str(c.get("id") or f"c{j}"), output=list(c.get("output") or [])))
        if not cands:
            raise ValueError(f"offline case {name}: no candidates")
        out.append(OfflineCase(name=name, requirements=reqs, candidates=cands))
    return out

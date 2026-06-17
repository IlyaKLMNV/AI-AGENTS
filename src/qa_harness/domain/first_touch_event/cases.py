"""Курируемые golden-кейсы для first_touch_event.

Мероприятие фиксировано (VK JT Go), поэтому кейс варьируется только именем кандидата.
Источник — tests/fixtures/first_touch_event/golden.yaml: candidate_name + offline_message (replay).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class GoldenCase:
    name: str
    candidate_name: str
    offline_message: Optional[str] = None


def load_golden(path: Path) -> List[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("golden file must be a YAML list")
    out: List[GoldenCase] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"golden case #{i} must be a mapping")
        candidate_name = str(item.get("candidate_name") or "").strip()
        if not candidate_name:
            raise ValueError(f"golden case #{i} has empty candidate_name")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate golden case name: {name}")
        seen.add(name)
        om = item.get("offline_message")
        out.append(GoldenCase(name=name, candidate_name=candidate_name,
                              offline_message=str(om) if om is not None else None))
    return out

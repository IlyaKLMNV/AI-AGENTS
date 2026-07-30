"""Курируемые golden-кейсы для теста промпта screening_autofill.

Источник — tests/fixtures/screening_autofill/golden.yaml: диалог (вход) + ожидания:
  expect — подмножество полей формы (work_format точным значением; зарплата/локация — `<nonempty>`);
  forbid_in_additional_info — темы, которых не должно быть в доп. вопросах (salary/location/work_format);
  expect_additional_info_nonempty — должен ли быть непустой additional_info;
  offline_output — каннная форма для `--offline` (replay без сети).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GoldenCase:
    name: str
    dialogue: str
    expect: Dict[str, Any] = field(default_factory=dict)
    forbid_in_additional_info: List[str] = field(default_factory=list)
    expect_additional_info_nonempty: bool = False
    offline_output: Optional[Dict[str, Any]] = None


def load_golden(path: Path) -> List[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("golden file must be a YAML list")
    out: List[GoldenCase] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"golden case #{i} must be a mapping")
        dialogue = str(item.get("dialogue") or "").strip()
        if not dialogue:
            raise ValueError(f"golden case #{i} has empty dialogue")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate golden case name: {name}")
        seen.add(name)
        oo = item.get("offline_output")
        out.append(
            GoldenCase(
                name=name,
                dialogue=dialogue,
                expect=item.get("expect") or {},
                forbid_in_additional_info=list(item.get("forbid_in_additional_info") or []),
                expect_additional_info_nonempty=bool(item.get("expect_additional_info_nonempty")),
                offline_output=oo if isinstance(oo, dict) else None,
            )
        )
    return out

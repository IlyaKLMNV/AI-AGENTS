"""Курируемые якорные кейсы для теста промпта extractor_agent.

Источник один — tests/fixtures/extractor_agent/anchors.yaml: курируемые фразы с
golden-ожиданиями (expect/forbid). Раньше тут были real(263)/suite/synthetic/mix —
удалено: для регрессии промпта маленький курируемый набор с эталоном надёжнее.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Anchor:
    name: str
    input: str
    expect: Dict[str, Any] = field(default_factory=dict)   # presence-ожидания по bucket'ам
    forbid: Dict[str, List[str]] = field(default_factory=dict)  # запрещённые размещения


def load_anchors(path: Path) -> List[Anchor]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("anchors file must be a YAML list")
    anchors: List[Anchor] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"anchor #{i} must be a mapping")
        text = str(item.get("input") or "").strip()
        if not text:
            raise ValueError(f"anchor #{i} has empty input")
        name = str(item.get("name") or f"anchor_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate anchor name: {name}")
        seen.add(name)
        anchors.append(
            Anchor(
                name=name,
                input=text,
                expect=item.get("expect") or {},
                forbid=item.get("forbid") or {},
            )
        )
    return anchors


def parse_steps(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [1, 2, 3]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        try:
            v = int(part)
            if v in (1, 2, 3) and v not in out:
                out.append(v)
        except ValueError:
            pass
    return out or [1, 2, 3]

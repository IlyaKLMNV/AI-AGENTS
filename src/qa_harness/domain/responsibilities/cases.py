"""Курируемые golden-кейсы для теста промпта responsibilities_parser.

Источник — tests/fixtures/responsibilities_parser/golden.yaml: текст вакансии (вход) + golden-ожидания
на СПИСОК ключевых слов (expect — ИЛИ-группы обязательных, forbid — запреты) + опциональный
offline_output (каннный массив ключевых слов) для `--offline`-прогона без сети.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class GoldenCase:
    name: str
    vacancy: str
    expect: List[Any] = field(default_factory=list)   # ИЛИ-группы обязательных критериев (внутри требований)
    forbid: List[Any] = field(default_factory=list)    # запрещённые концепты (nice-to-have/soft/условия)
    expect_empty: bool = False                         # правильный ответ — пустой массив (нет обязательных требований)
    offline_output: Optional[List[str]] = None         # replay для --offline


def load_golden(path: Path) -> List[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("golden file must be a YAML list")
    out: List[GoldenCase] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"golden case #{i} must be a mapping")
        vacancy = str(item.get("vacancy") or "").strip()
        if not vacancy:
            raise ValueError(f"golden case #{i} has empty vacancy")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate golden case name: {name}")
        seen.add(name)
        oo = item.get("offline_output")
        out.append(
            GoldenCase(
                name=name,
                vacancy=vacancy,
                expect=item.get("expect") or [],
                forbid=item.get("forbid") or [],
                expect_empty=bool(item.get("expect_empty")),
                offline_output=[str(x) for x in oo] if isinstance(oo, list) else None,
            )
        )
    return out

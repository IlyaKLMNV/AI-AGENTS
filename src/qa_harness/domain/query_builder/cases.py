"""Курируемые golden-кейсы для теста промпта one_line_search_query_builder.

Источник — tests/fixtures/one_line_search_query_builder/golden.yaml: вакансия (вход
билдера) + golden-ожидания на СТРОКУ запроса (expect/forbid) + опциональный
offline_query (что мог бы выдать билдер) для офлайн-прогона без сети.
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
    expect: List[Any] = field(default_factory=list)  # ИЛИ-группы обязательных терминов
    forbid: List[Any] = field(default_factory=list)   # запрещённые термины
    offline_query: Optional[str] = None               # replay для --offline (без сети)


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
        oq = item.get("offline_query")
        out.append(
            GoldenCase(
                name=name,
                vacancy=vacancy,
                expect=item.get("expect") or [],
                forbid=item.get("forbid") or [],
                offline_query=str(oq).strip() if oq is not None else None,
            )
        )
    return out

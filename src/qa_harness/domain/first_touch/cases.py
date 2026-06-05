"""Курируемые golden-кейсы для теста промпта first_touch.

Источник — tests/fixtures/first_touch/golden.yaml. Каждый кейс:
  input — payload для генерации (candidate_name, recruiter_name, candidate_source,
    reason_of_communication, hiring_company_name, vacancy_name, vacancy_responsibilities,
    message_formality, company_description, vacancy_stack, salary_range);
  expected_facts — факты, которые сообщение ДОЛЖНО содержать (судит LLM-судья);
  allowed_context_facts — допустимые контекстные факты (источник/причина), не галлюцинации;
  optional_facts — ключи из expected_facts, отсутствие которых НЕ валит passed (напр. зарплата);
  company_hidden — название компании НЕ должно упоминаться;
  require_question — требуется вопросительный CTA;
  offline_message — каннное сообщение для `--offline` (replay без сети/судьи).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GoldenCase:
    name: str
    input: Dict[str, Any]
    expected_facts: Dict[str, str] = field(default_factory=dict)
    allowed_context_facts: Dict[str, str] = field(default_factory=dict)
    optional_facts: List[str] = field(default_factory=list)
    company_hidden: bool = False
    require_question: bool = True
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
        inp = item.get("input")
        if not isinstance(inp, dict) or not inp:
            raise ValueError(f"golden case #{i} has empty input")
        expected = item.get("expected_facts") or {}
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"golden case #{i} has empty expected_facts")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate golden case name: {name}")
        seen.add(name)
        om = item.get("offline_message")
        out.append(
            GoldenCase(
                name=name,
                input={k: ("" if v is None else v) for k, v in inp.items()},
                expected_facts={k: str(v) for k, v in expected.items()},
                allowed_context_facts={k: str(v) for k, v in (item.get("allowed_context_facts") or {}).items()},
                optional_facts=list(item.get("optional_facts") or []),
                company_hidden=bool(item.get("company_hidden")),
                require_question=bool(item.get("require_question", True)),
                offline_message=str(om) if om is not None else None,
            )
        )
    return out

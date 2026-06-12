"""Резолв CandidateConstraints для сценария: дефолт из name+description + оверлей из YAML.

Сценарий без записи в constraints.yaml всё равно работает — кандидат строится из name+description
(адаптивная генерация не требует примеров). YAML-запись лишь уточняет триггер/маркеры/фолбэк для
тонких сценариев (перенос легаси-правил как данных). Резолв: по index, иначе по match (подстрока имени).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from qa_harness.domain.generators import CandidateConstraints

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONSTRAINTS = REPO_ROOT / "tests" / "fixtures" / "generation" / "screening_scenarios" / "constraints.yaml"


def load_constraints(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Список записей-констрейнтов из YAML (пусто, если файла нет)."""
    p = Path(path) if path else DEFAULT_CONSTRAINTS
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = data.get("scenarios") or []
    return [e for e in entries if isinstance(e, dict)]


def _match_entry(scenario: Any, entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    name_low = (getattr(scenario, "name", "") or "").lower()
    idx = getattr(scenario, "index", None)
    for e in entries:                                  # точное совпадение по index приоритетнее
        if e.get("index") is not None and e.get("index") == idx:
            return e
    # match — запасной резолвер ТОЛЬКО для записей без index (иначе подстрока вещает на чужие имена,
    # напр. "бот" ⊂ "работе"). Запись с index адресует один сценарий — её match не бродкастим.
    for e in entries:
        if e.get("index") is None:
            m = (e.get("match") or "").strip().lower()
            if m and m in name_low:
                return e
    return None


def _as_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def constraints_for(scenario: Any, entries: List[Dict[str, Any]]) -> CandidateConstraints:
    """Собрать CandidateConstraints: база из сценария, оверлей — из найденной YAML-записи."""
    c = CandidateConstraints(
        scenario_name=getattr(scenario, "name", "") or "",
        scenario_description=getattr(scenario, "description", "") or "",
    )
    e = _match_entry(scenario, entries)
    if not e:
        return c
    if e.get("trigger_requirement"):
        c.trigger_requirement = str(e["trigger_requirement"])
    c.guidelines = _as_list(e.get("guidelines"))
    c.require_any = _as_list(e.get("require_any"))
    c.forbid_any = _as_list(e.get("forbid_any"))
    c.examples = _as_list(e.get("examples"))
    c.fallback = _as_list(e.get("fallback"))
    if e.get("language"):
        c.language = str(e["language"])
    return c

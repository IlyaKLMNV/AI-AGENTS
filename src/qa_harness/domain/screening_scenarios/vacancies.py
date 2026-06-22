"""Пер-сценарный контекст вакансии для screening_scenarios.

Многие сценарии предполагают КОНКРЕТНУЮ вакансию (формат ON_SITE/HYBRID/FIELD_WORK, скрытый поиск =
пустая компания, рекрутинговое агентство, развёрнутое описание). По умолчанию раннер подаёт один
DEFAULT_VACANCY_INFO на все сценарии — из-за чего такие сценарии падали ложно: ассистент вёл себя
правильно для дефолтной remote/открытой вакансии, а сценарий ждал поведение под другую.

Здесь — оверрайды по index сценария, мерджатся поверх дефолта. Файл per-component (index у base и hh
не пересекаются — как и у constraints). Резолв: строго по index = номер строки CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_vacancies(path: Optional[Path] = None) -> Dict[int, Dict[str, Any]]:
    """index -> частичный оверрайд vacancy_info (пусто, если файла нет)."""
    if not path or not Path(path).is_file():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: Dict[int, Dict[str, Any]] = {}
    for e in (data.get("scenarios") or []):
        if isinstance(e, dict) and e.get("index") is not None:
            out[int(e["index"])] = {k: v for k, v in e.items() if k not in ("index", "match")}
    return out


def vacancy_for(scenario: Any, overrides: Dict[int, Dict[str, Any]], default: Dict[str, Any]) -> Dict[str, Any]:
    """Слить дефолтную вакансию с оверрайдом сценария (company_info — вложенный мердж)."""
    ov = overrides.get(getattr(scenario, "index", None)) if overrides else None
    if not ov:
        return default
    merged = {**default, **ov}
    if isinstance(ov.get("company_info"), dict):
        merged["company_info"] = {**(default.get("company_info") or {}), **ov["company_info"]}
    return merged

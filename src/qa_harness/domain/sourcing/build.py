"""Сборка входов sourcing из CDM: требования (1..5 строк) + профиль кандидата из backend-выдачи.

Чистый домен (без сети и без pipeline): сам backend-поиск и сборку payload оркестрирует раннер.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

REQUIREMENTS_SOURCES = ("cdm_key_requirements", "stack_skills")


def _split_list_like(s: Any) -> List[str]:
    if not s:
        return []
    out: List[str] = []
    for part in re.split(r"[,\n;|]+", str(s)):
        t = re.sub(r"\s+", " ", (part or "").strip())
        if t:
            out.append(t)
    return out


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = (x or "").strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def requirements_from_cdm(vacancy: Dict[str, Any], source: str = "cdm_key_requirements") -> List[str]:
    """1..5 требований из CDM. cdm_key_requirements (с фолбэком на stack+skills) или stack_skills."""
    if source == "stack_skills":
        items = _split_list_like(vacancy.get("vacancy_stack")) + _split_list_like(vacancy.get("vacancy_skills"))
    else:
        raw = vacancy.get("key_requirements")
        if isinstance(raw, list):
            items = [re.sub(r"\s+", " ", v.strip()) for v in raw if isinstance(v, str) and v.strip()]
        else:
            items = _split_list_like(str(raw) if raw is not None else None)
        if not items:  # фолбэк, как в легаси
            items = _split_list_like(vacancy.get("vacancy_stack")) + _split_list_like(vacancy.get("vacancy_skills"))
    return _dedupe(items)[:5]


def build_candidate_profile(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """backend-кандидат -> profile {about, skills[], positions[]} в формате входа промпта."""
    skills_out: List[Dict[str, Any]] = []
    for skill in candidate.get("skills") or []:
        if isinstance(skill, dict) and isinstance(skill.get("skill"), str) and skill["skill"].strip():
            skills_out.append({"skill": re.sub(r"\s+", " ", skill["skill"].strip())})

    positions_out: List[Dict[str, Any]] = []
    for pos in candidate.get("positions") or []:
        if not isinstance(pos, dict):
            continue
        categories: List[Dict[str, str]] = []
        for category in ((pos.get("company_norm") or {}).get("categories") or []):
            if isinstance(category, dict) and isinstance(category.get("title"), str) and category["title"].strip():
                categories.append({"title": re.sub(r"\s+", " ", category["title"].strip())})
            elif isinstance(category, str) and category.strip():
                categories.append({"title": re.sub(r"\s+", " ", category.strip())})

        positions_norm: List[Any] = []
        for value in pos.get("positions_norm") or []:
            if isinstance(value, dict):
                item: Dict[str, str] = {}
                for key in ("title", "name", "raw_text"):
                    v = value.get(key)
                    if isinstance(v, str) and v.strip():
                        item[key] = re.sub(r"\s+", " ", v.strip())
                if item:
                    positions_norm.append(item)
            elif isinstance(value, str) and value.strip():
                positions_norm.append(re.sub(r"\s+", " ", value.strip()))

        positions_out.append({
            "name": re.sub(r"\s+", " ", str(pos.get("name") or "").strip()),
            "pos": re.sub(r"\s+", " ", str(pos.get("pos") or "").strip()),
            "description": re.sub(r"\s+", " ", str(pos.get("description") or "").strip()),
            "rangeStr": re.sub(r"\s+", " ", str(pos.get("rangeStr") or "").strip()),
            "dates": pos.get("dates") or [],
            "current": bool(pos.get("current")),
            "positions_norm": positions_norm,
            "company_norm": {"categories": categories},
        })

    return {
        "about": re.sub(r"\s+", " ", str(candidate.get("about") or "").strip()),
        "skills": skills_out,
        "positions": positions_out,
    }

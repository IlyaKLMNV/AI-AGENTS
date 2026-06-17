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


@dataclass
class GoldenScoreCase:
    """Golden-кейс scoring (НОВАЯ задача): требования + данные кандидата + эталонные passed.

    requirements — требования-предложения (может быть пустым); candidate_data — данные кандидата (резюме /
    профиль / анкета / явно переданные данные — НЕ только резюме); expect_passed — эталон 0/1 той же длины;
    offline_output — каннный ответ промпта для --offline (replay).
    """
    name: str
    requirements: List[str]
    candidate_data: str
    expect_passed: List[int] = field(default_factory=list)
    offline_output: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchVacancy:
    """Вакансия для ЖИВОГО поиска кандидатов (--search): полный текст + требования для оценки.

    vacancy — текст вакансии (идёт в extractor → entities → backend-поиск кандидатов); title — заголовок
    (для payload); requirements — требования-предложения, по которым промпт sourcing оценивает найденных
    живых кандидатов (эталона passed нет — это реальные люди, поэтому только contract-качество).
    """
    name: str
    title: str
    vacancy: str
    requirements: List[str] = field(default_factory=list)


def load_search_vacancies(path: Path) -> List[SearchVacancy]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("vacancies file must be a YAML list")
    out: List[SearchVacancy] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"vacancy #{i} must be a mapping")
        name = str(item.get("name") or f"vacancy_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate vacancy name: {name}")
        seen.add(name)
        vacancy = str(item.get("vacancy") or "").strip()
        if not vacancy:
            raise ValueError(f"vacancy {name}: empty vacancy text")
        reqs = [str(r).strip() for r in (item.get("requirements") or []) if str(r).strip()]
        out.append(SearchVacancy(name=name, title=str(item.get("title") or "").strip(),
                                 vacancy=vacancy, requirements=reqs))
    return out


def load_golden_score(path: Path) -> List[GoldenScoreCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("golden file must be a YAML list")
    out: List[GoldenScoreCase] = []
    seen = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"golden case #{i} must be a mapping")
        name = str(item.get("name") or f"case_{i:04d}").strip()
        if name in seen:
            raise ValueError(f"duplicate golden case name: {name}")
        seen.add(name)
        reqs = [str(r).strip() for r in (item.get("requirements") or []) if str(r).strip()]
        # данные кандидата: candidate_data (новое) с фолбэком на resume_text/input (compat)
        cdata = item.get("candidate_data")
        if cdata is None:
            cdata = item.get("resume_text")
        if cdata is None:
            cdata = item.get("input") or ""
        expect = [int(x) for x in (item.get("expect_passed") or [])]
        if len(expect) != len(reqs):
            raise ValueError(f"golden case {name}: expect_passed ({len(expect)}) != requirements ({len(reqs)})")
        oo = item.get("offline_output")
        out.append(GoldenScoreCase(
            name=name, requirements=reqs, candidate_data=str(cdata), expect_passed=expect,
            offline_output=list(oo) if isinstance(oo, list) else [],
        ))
    return out

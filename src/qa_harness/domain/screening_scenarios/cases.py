"""Сценарии скрининга из CSV (golden-источник) для screening_scenarios.

CSV (tests/fixtures/screening_scenarios.csv) даёт по строке-сценарию: название, описание,
**ожидаемое поведение модели** и примеры диалогов. Из примеров вытаскиваем реплики кандидата
(`[candidate]`/`[кандидат]`) — ими гоняем живой screening_assistant. Оценка — LLM-судья против
expected_behavior (см. judge.py). Перенос load_scenarios/extract_candidate_examples из легаси.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Scenario:
    index: int
    name: str
    description: str
    expected_behavior: str
    examples_raw: str


_NAME_KEY = "Название сценария"
_DESC_KEYS = ["Краткое описание сценария", "Описание сценария"]
_BEHAVIOR_KEYS = [
    "Ожидаемое поведение модели (согласно промпту) ",
    "Ожидаемое поведение модели (согласно промпту)",
    "Ожидаемое поведение модели (как она должна отработать)",
]
_EXAMPLES_KEYS = ["Сообщениия с примерами диалогов ", "Сообщения с примерами диалогов"]


def _first(row: dict, keys: List[str]) -> str:
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return ""


def load_scenarios(csv_path: Path) -> List[Scenario]:
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"CSV with scenarios not found: {csv_path}")
    out: List[Scenario] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            name = (row.get(_NAME_KEY) or "").strip()
            if not name:
                continue
            out.append(Scenario(
                index=idx,
                name=name,
                description=_first(row, _DESC_KEYS).strip(),
                expected_behavior=_first(row, _BEHAVIOR_KEYS).strip(),
                examples_raw=_first(row, _EXAMPLES_KEYS),
            ))
    return out


_ROLE_RE = re.compile(r"\[(recruiter|candidate|кандидат|рекрутер|assistant|ассистент)\]", re.IGNORECASE)
_CANDIDATE_ROLES = {"candidate", "кандидат"}


def _iter_full_texts(examples_raw: str):
    """Извлекает full_text из всех JSON-объектов примера (объекты разделены произвольными \\r\\n/пробелами).

    Сканируем raw_decode'ом, перескакивая между `{...}` — устойчиво к `\\r\\n\\r\\n`, склейкам и мусору
    между объектами. Если JSON-объектов нет вовсе — отдаём весь текст как один full_text (raw-fallback).
    """
    decoder = json.JSONDecoder()
    found = False
    i, n = 0, len(examples_raw)
    while i < n:
        j = examples_raw.find("{", i)
        if j < 0:
            break
        try:
            obj, end = decoder.raw_decode(examples_raw, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        i = end
        if isinstance(obj, dict) and obj.get("full_text"):
            found = True
            yield str(obj["full_text"])
    if not found and examples_raw.strip():
        yield examples_raw


def extract_candidate_examples(examples_raw: str, max_examples: int = 4) -> List[str]:
    """Реплики кандидата из примеров.

    Примеры — JSON-объекты `{full_text}`, где весь диалог лежит ОДНОЙ строкой с инлайн-метками
    `[recruiter] ... [candidate] ... [recruiter] ...`. Разбиваем full_text по меткам ролей и собираем
    сегменты кандидата (по строкам не делим — переводов строк внутри full_text нет).
    """
    if not examples_raw:
        return []
    candidates: List[str] = []
    for full_text in _iter_full_texts(examples_raw):
        parts = _ROLE_RE.split(full_text)  # [pre, role1, text1, role2, text2, ...]
        for k in range(1, len(parts) - 1, 2):
            role = parts[k].lower()
            text = (parts[k + 1] or "").strip()
            if role in _CANDIDATE_ROLES and text:
                candidates.append(text)
                if len(candidates) >= max_examples:
                    return candidates
    return candidates


def parse_scenario_indices(raw: str) -> List[int]:
    out: List[int] = []
    for tok in (t.strip() for t in re.split(r"[,\s]+", raw or "") if t.strip()):
        if tok.isdigit():
            v = int(tok)
            if v not in out:
                out.append(v)
    return out

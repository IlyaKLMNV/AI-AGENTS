"""Семантика screening_autofill: golden-ожидания по полям формы + анти-утечка в additional_info.

- `check_expect(parsed, expect)` (gate): по каждому ключу из expect значение совпадает. Спецзначение
  `"<nonempty>"` — поле должно быть непустой строкой (для зарплаты/локации, где точный формат варьируется;
  work_format — enum, поэтому проверяется точным значением).
- `additional_info_leaks(parsed, forbid_topics)` (gate): в `additional_info` НЕ должно быть запрещённых тем
  (salary/location/work_format) и меток спикера («Рекрутер:»/«Кандидат:»). По правилам промпта эти темы
  идут в отдельные поля формы, а не в доп. вопросы.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

NONEMPTY = "<nonempty>"

_SALARY = re.compile(
    r"(зарплат|оклад|компенсац|вилк|вознагражд|на руки|нетто|netto|gross|гросс|брутто|₽|\$|€|\bруб|\bтыс)",
    re.IGNORECASE,
)
_WORKFORMAT = re.compile(
    r"(удал[её]н|дистанцион|remote|офис|\bочно\b|гибрид|смешан|формат работы|режим работы)",
    re.IGNORECASE,
)
_LOCATION = re.compile(
    r"(город|локац|переезд|релокац|москв|петербург|\bспб\b|казан|новосибирск|екатеринбург|"
    r"нижн|самар|краснодар|ростов|воронеж|пермь|минск|алмат|астан|тбилиси)",
    re.IGNORECASE,
)
_SPEAKER = re.compile(r"(Рекрутер|Кандидат)\s*:", re.IGNORECASE)

_TOPIC_PATTERNS = {"salary": _SALARY, "work_format": _WORKFORMAT, "location": _LOCATION}


def check_expect(parsed: Any, expect: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Вернуть (ok, diffs). expect — подмножество ожидаемых полей формы."""
    if not isinstance(parsed, dict):
        return False, ["output_not_object"]
    diffs: List[str] = []
    for k, v in (expect or {}).items():
        actual = parsed.get(k)
        if v == NONEMPTY:
            if not (isinstance(actual, str) and actual.strip()):
                diffs.append(f"{k}:expected_nonempty:actual={actual!r}")
        elif actual != v:
            diffs.append(f"{k}:expected={v!r}:actual={actual!r}")
    return (len(diffs) == 0), diffs


def additional_info_leaks(parsed: Any, forbid_topics: List[str]) -> List[str]:
    """Утечки в additional_info: метки спикера всегда запрещены; темы — из forbid_topics."""
    leaks: List[str] = []
    ai = parsed.get("additional_info") if isinstance(parsed, dict) else None
    if not isinstance(ai, list):
        return leaks
    topics = [t for t in (forbid_topics or []) if t in _TOPIC_PATTERNS]
    for i, item in enumerate(ai):
        if not isinstance(item, dict):
            continue
        blob = f"{item.get('question') or ''} {item.get('answer') or ''}"
        if _SPEAKER.search(blob):
            leaks.append(f"additional_info[{i}]:speaker_label")
        for t in topics:
            if _TOPIC_PATTERNS[t].search(blob):
                leaks.append(f"additional_info[{i}]:{t}")
    return leaks

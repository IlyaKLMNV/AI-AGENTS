"""Хелперы формата диалога Рекрутер/Кандидат (переиспользуются verdict и screening).

Перенесено из verdict_classifier_runner (_split_dialogue_lines, _speaker_for_line).
"""

from __future__ import annotations

from typing import List, Optional

RECRUITER_PREFIX = "Рекрутер:"
CANDIDATE_PREFIX = "Кандидат:"


def split_dialogue_lines(text: str) -> List[str]:
    """Разбить диалог на непустые строки; обрезать после первой реплики рекрутера с END."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith(RECRUITER_PREFIX) and "END" in line:
            return lines[: idx + 1]
    return lines


def speaker_for_line(line: str) -> Optional[str]:
    """'recruiter' | 'candidate' | None по префиксу строки."""
    if line.startswith(RECRUITER_PREFIX):
        return "recruiter"
    if line.startswith(CANDIDATE_PREFIX):
        return "candidate"
    return None

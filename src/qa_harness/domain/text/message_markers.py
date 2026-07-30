"""Regex-маркеры классов сообщений кандидата (единый источник).

Перенесены дословно из старого message_classifier_runner (DECLINE/REASON/
ACCEPTANCE/HUMAN_NEEDED). Используются и офлайн-классификатором, и валидацией
сгенерированных сообщений — чтобы паттерны жили в одном месте, а не дублировались.
"""

from __future__ import annotations

import re
from typing import Sequence

DECLINE_PATTERNS = (
    r"\bне\s+интерес",
    r"\bне\s+рассматрива",
    r"\bне\s+подходит",
    r"\bвынужден\s+отказ",
    r"\bоткаж",
    r"\bотказ",
    r"\bне\s+готов",
    r"\bне\s+смогу",
    r"\bнет,\s*спасибо\b",
)
REASON_PATTERNS = (
    r"\bпотому\s+что\b",
    r"\bтак\s+как\b",
    r"\bпоскольку\b",
    r"\bуже\b",
    r"\bоффер",
    r"\bзарплат",
    r"\bформат",
    r"\bофис",
    r"\bгибрид",
    r"\bудален",
    r"\bлокац",
    r"\bпереезд",
    r"\bстек",
    r"\bсфера",
    r"\bработаю\b",
    r"\bвышел\s+на\s+работу\b",
    r"\bпринял\s+оффер\b",
)
ACCEPTANCE_PATTERNS = (
    r"\bинтерес",
    r"\bваканси",
    r"\bподскажите\b",
    r"\bрасскажите\b",
    r"\bможете\s+уточнить\b",
    r"\bкакая\s+компания\b",
    r"\bкак\s+ваша\s+компания\s+называется\b",
    r"\bссылка\s+на\s+ваканси",
    r"\bописани[ея]\b",
    r"\bкоманд",
    r"\bзадач",
    r"\bстек",
    r"\bформат",
    r"\bзарплат",
    r"\bсозвон",
    r"\bготов\s+обсудить\b",
)
HUMAN_NEEDED_PATTERNS = (
    r"\bстранн",
    r"\bчто\s+за\s+ерунд",
    r"\bмошенн",
    r"\bразвод",
    r"\bденьги\b",
    r"\bскиньте\b",
    r"\bоткуда\s+нашли\s+контакт\b",
    r"\bзачем\s+мне\s+тратить\s+время\b",
    r"\bне\s+совсем\s+понимаю\b",
    r"\bбред\b",
    r"\bхрень\b",
    r"\bено[тт]\b",
    r"[🦝😕🤨]",
)


def has_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

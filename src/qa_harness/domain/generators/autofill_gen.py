"""Генератор диалога с ИЗВЕСТНЫМ work_format для теста промпта screening_autofill.

В отличие от DialogueGenerator (диалог под target_verdict) здесь цель — диалог, из которого форма
извлекается ОДНОЗНАЧНО: кандидат явно называет город, зарплатные ожидания (числом) и предпочитаемый
формат работы (hybrid/remote/office). Это даёт детерминированный `expect` для autofill-судьи
(work_format точным значением; зарплата/локация — непустые). Перенос идеи легаси-регрессии work_format
(_build_regression_dialogue/_candidate_work_format_response) на вариативную LLM-генерацию + валидацию.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..text.dialogue import split_dialogue_lines
from .base import Generator

WORK_FORMATS = ("hybrid", "remote", "office")

# Маркеры формата в репликах кандидата — для валидации, что нужный work_format реально заявлен.
_FORMAT_MARKERS: Dict[str, tuple] = {
    "hybrid": ("гибрид", "1-2 дня", "1–2 дня", "частичн", "смешан"),
    "remote": ("удал", "дистанцион", "remote", "из дома", "онлайн"),
    "office": ("офис", "очно", "на площадк", "в офисе"),
}
_FORMAT_RU = {"hybrid": "гибрид (1-2 дня в офисе)", "remote": "полностью удалённо (remote)", "office": "офис (очно)"}
_SALARY_RE = re.compile(r"(\d{2,}|тыс|тысяч|к\b)", re.IGNORECASE)
_LOCATION_RE = re.compile(
    r"(город|живу|нахожусь|переезд|москв|петербург|спб|казан|новосиб|екатеринб|нижн|самар|"
    r"краснодар|ростов|воронеж|перм|минск|алмат|астан|тбилиси)", re.IGNORECASE)


@dataclass
class AutofillSpec:
    cdm: Dict[str, Any]
    work_format: str          # один из WORK_FORMATS
    noise_level: int = 1


def validate_autofill_dialogue(text: str, work_format: str) -> Optional[str]:
    """None == ок; строка == причина отказа. Проверяем, что форма извлекаема: формат+зарплата+город заявлены."""
    lines = split_dialogue_lines(text or "")
    if not lines:
        return "пустой диалог"
    cand = " ".join(l for l in lines if l.lower().startswith("кандидат")).lower()
    if not cand:
        return "нет реплик кандидата"
    markers = _FORMAT_MARKERS.get(work_format, ())
    if not any(m in cand for m in markers):
        return f"кандидат не заявил work_format={work_format} (нет маркеров {markers})"
    if not _SALARY_RE.search(cand):
        return "кандидат не назвал зарплатные ожидания (числом)"
    if not _LOCATION_RE.search(cand):
        return "кандидат не указал город/локацию"
    return None


class AutofillDialogueGenerator(Generator):
    """Диалог рекрутёр/кандидат, из которого work_format/зарплата/локация извлекаются однозначно."""

    def instruction(self, spec: AutofillSpec) -> str:
        return (
            "Ты генерируешь реалистичный диалог первичного скрининга между рекрутёром и кандидатом.\n"
            "Формат строго построчно: каждая строка начинается с 'Рекрутер:' или 'Кандидат:', по-русски.\n"
            "Жёсткие требования к содержанию (иначе тест бессмыслен):\n"
            "- Кандидат ЯВНО называет свой город (где находится).\n"
            "- Кандидат ЯВНО называет зарплатные ожидания КОНКРЕТНЫМ числом в рублях.\n"
            "- Кандидат ЯВНО обозначает предпочитаемый ФОРМАТ РАБОТЫ согласно заданию.\n"
            "- Рекрутёр НЕ раскрывает зарплатную вилку/бюджет числом.\n"
            "- Диалог естественный, 4-8 реплик, без markdown и пояснений вне диалога."
        )

    def payload(self, spec: AutofillSpec) -> str:
        vacancy = spec.cdm.get("vacancy") or {}
        candidate = spec.cdm.get("candidate") or {}
        noise = ["низкий", "средний", "высокий"][min(max(spec.noise_level, 0), 2)]
        ctx = {
            "TARGET_WORK_FORMAT": spec.work_format,
            "формат_словами": _FORMAT_RU.get(spec.work_format, spec.work_format),
            "noise_level": noise,
            "vacancy": {"title": vacancy.get("title"), "company_name": vacancy.get("company_name"),
                        "responsibilities": vacancy.get("responsibilities")},
            "candidate": {"recruiter_name": candidate.get("recruiter_name"),
                          "candidate_name": candidate.get("candidate_name")},
        }
        return (
            "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\n"
            "INSTRUCTIONS:\n"
            f"1) Кандидат должен предпочесть формат: {_FORMAT_RU.get(spec.work_format, spec.work_format)} "
            f"(TARGET_WORK_FORMAT={spec.work_format}).\n"
            "2) Кандидат обязательно называет город и зарплату числом.\n"
            "3) Верни только диалог построчно (Рекрутер:/Кандидат:)."
        )

    def parse(self, text: str, spec: AutofillSpec) -> str:
        text = (text or "").strip()
        err = validate_autofill_dialogue(text, spec.work_format)
        if err:
            raise ValueError(err)
        return text

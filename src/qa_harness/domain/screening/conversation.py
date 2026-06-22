"""Мультитёрн-разговор со screening_assistant через OpenAI Conversations API.

Переписано из легаси screeningAssistant/screeningAss.py (БЕЗ импорта легаси): разговор сидируется
сообщением с деталями вакансии и инструкциями квалификации, затем реплики кандидата шлются в
stored-промпт screening_assistant с conversation=<id>. Общая инфра для screening_guardrails и
screening_scenarios. Сеть только здесь; детекторы/судья — отдельно.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

MATERNITY_RE = re.compile(r"\bдекрет\w*\b", re.IGNORECASE)
_OFFTOPIC_END = "мне нужно будет уточнить этот момент у коллег"


def _salary_phrase(vacancy_info: Dict[str, Any]) -> str:
    lo, hi = vacancy_info.get("min_salary"), vacancy_info.get("max_salary")
    if lo and hi:
        return f"от {lo} до {hi} рублей"
    if lo:
        return f"от {lo} рублей"
    if hi:
        return f"до {hi} рублей"
    return ""


def build_seed_message(vacancy_info: Dict[str, Any], recruiter_name: str, candidate_name: str) -> str:
    """Первое assistant-сообщение разговора: детали вакансии + инструкции квалификации (как в легаси)."""
    ci = vacancy_info.get("company_info") or {}
    # Опц. поля попадают в seed только если заданы (per-scenario контекст вакансии): нанимающая
    # сторона (рекрутинговое агентство) и источник контакта. Дефолт без них — поведение неизменно.
    _hiring = vacancy_info.get("hiring_company_name")
    _contact = vacancy_info.get("contact_source")
    extra_company = f"- Нанимающая сторона: {_hiring}\n" if _hiring else ""
    extra_contact = f"Источник контакта кандидата: {_contact}\n" if _contact else ""
    return (
        f"Ваше имя: {recruiter_name}\n"
        f"Имя кандидата: {candidate_name}\n"
        f"{extra_contact}"
        "\n**Детали вакансии**:\n"
        f"- Должность: {vacancy_info.get('title', '')}\n"
        f"- Название компании: {vacancy_info.get('company_name', '')}\n"
        f"{extra_company}"
        f"- Обязанности: {vacancy_info.get('responsibilities', '')}\n"
        f"- Формат работы: {vacancy_info.get('work_format', '')}\n"
        f"- Локация: {vacancy_info.get('location', '')}\n"
        f"- Описание компании: {ci.get('firm_description', '')}\n"
        f"- Ссылка на вакансию: {ci.get('vacancy_url', '')}\n"
        f"- Зарплатная вилка: {_salary_phrase(vacancy_info)}\n\n"
        "## Вопросы для квалификации:\n\n"
        "### ПРИОРИТЕТНЫЕ ВОПРОСЫ (задавать ПЕРВЫМИ ВСЕГДА):\n\n"
        "1. **Зарплатные ожидания** - для проверки соответствия бюджету\n"
        "2. **Локация/город проживания** - для проверки соответствия формату работы\n\n"
        "### ДОПОЛНИТЕЛЬНЫЕ ВОПРОСЫ (только если кандидат прошел первичный отбор):\n\n"
        f"{vacancy_info.get('questions', '')}\n\n"
        "**Контекст диалога**:\n"
        "Кандидат уже ознакомлен с базовой информацией о вакансии из первичного контакта. Ваша задача — провести "
        "квалифицирующее интервью и собрать необходимую информацию для передачи внутреннему рекрутеру.\n\n"
        "**ОБЯЗАТЕЛЬНО начните диалог с приветствия и сразу же задайте приоритетные вопросы**\n\n"
        "**КРИТИЧЕСКИ ВАЖНО:** После получения ответов на приоритетные вопросы — ОБЯЗАТЕЛЬНО проверьте "
        "соответствие требованиям перед продолжением диалога!"
    )


@dataclass
class TurnResult:
    response: Optional[str]
    conversation_end: bool
    usage: Any = None


class ScreeningConversation:
    """Stateful-разговор со screening_assistant. start() сидирует тред, respond() — один ход кандидата."""

    def __init__(self, client: Any, prompt_id: str, prompt_version: Optional[str],
                 vacancy_info: Dict[str, Any], recruiter_name: str, candidate_name: str) -> None:
        self._client = client
        self._prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self._prompt["version"] = str(prompt_version)
        self._seed = build_seed_message(vacancy_info, recruiter_name, candidate_name)
        self._conversation_id: Optional[str] = None

    def start(self) -> str:
        conv = self._client.conversations.create(
            items=[{"type": "message", "role": "assistant", "content": self._seed}]
        )
        self._conversation_id = conv.id
        return self._conversation_id

    def respond(self, candidate_message: str) -> TurnResult:
        if MATERNITY_RE.search(candidate_message or ""):
            return TurnResult("Извините за беспокойство!", True, None)
        resp = self._client.responses.create(
            prompt=self._prompt, conversation=self._conversation_id, input=candidate_message
        )
        usage = getattr(resp, "usage", None)
        text = getattr(resp, "output_text", "") or ""
        if not text:
            return TurnResult(None, True, usage)
        if _OFFTOPIC_END in text.lower():
            return TurnResult(None, True, usage)
        end = "END" in text
        if end:
            text = text.replace("END", "").strip()
        return TurnResult(text, end, usage)

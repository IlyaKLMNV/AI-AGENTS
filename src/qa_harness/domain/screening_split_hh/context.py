"""Сборка контекста вакансии для split-движка — HH-канал.

Аналог `screening_split/context.py` (TG), дельта по EGGPLANT_SPLIT_TASK.md §7 / SPLIT_TG_VS_HH.md §2.2:
- убраны поля «Источник контакта», «Обязанности», «Описание компании», «Ссылка на вакансию»;
- добавлено «Описание вакансии» (свободный текст с hh);
- «Название компании» — реальное ЛИБО литерал «рекрутинговое агентство» (механики СКРЫТО нет);
- «Формат работы» — из нормализованного `allowed_formats` (список id), а не сырое поле.

Порядок и лейблы строк совпадают с блоком «Контекст вакансии» в
`prompts/screening_analyzer_hh/v1/system.md` — промпт ссылается на эти лейблы.
`salary_display` и сид Интервьюера идентичны TG — переиспользуем импортом.
"""

from typing import Any

# идентичны TG — импортируем, чтобы формат вилки и least-privilege-сид не разъезжались между каналами.
from qa_harness.domain.screening_split.context import build_interviewer_seed, salary_display

from .state import normalize_work_formats

_AGENCY_LITERAL = "рекрутинговое агентство"


def allowed_formats_of(vacancy_info: dict) -> list[str]:
    """Нормализованный список форматов из вакансии (`allowed_formats` или сырое `work_format`)."""
    raw = vacancy_info.get("allowed_formats")
    if raw is None:
        raw = vacancy_info.get("work_format")
    return normalize_work_formats(raw)


def build_context(
    recruiter_name: str,
    candidate_name: str,
    vacancy_info: dict,
) -> str:
    """Полный блок фактов вакансии — ТОЛЬКО для Аналитика (умная модель, с секретами).

    Интервьюеру НЕ передаётся: он получает урезанный seed (build_interviewer_seed);
    нужные факты Аналитик кладёт в instruction уже отредактированными.
    """
    company_name = (vacancy_info.get("company_name") or "").strip() or _AGENCY_LITERAL
    allowed = allowed_formats_of(vacancy_info)
    description = vacancy_info.get("vacancy_description") or vacancy_info.get("description") or ""
    return (
        f"Ваше имя: {recruiter_name}\n"
        f"Имя кандидата: {candidate_name}\n"
        f"Должность: {vacancy_info.get('title', '')}\n"
        f"Название компании: {company_name}\n"
        f"Формат работы: {', '.join(allowed)}\n"
        f"Локация: {vacancy_info.get('location', '') or ''}\n"
        f"Описание вакансии: {description}\n"
        f"Зарплатная вилка: {salary_display(vacancy_info.get('min_salary'), vacancy_info.get('max_salary'))} (НЕ РАСКРЫВАТЬ!)\n"
        "Дополнительные вопросы:\n"
        f"{vacancy_info.get('questions', '') or ''}"
    )


__all__ = ["build_context", "build_interviewer_seed", "salary_display", "allowed_formats_of"]

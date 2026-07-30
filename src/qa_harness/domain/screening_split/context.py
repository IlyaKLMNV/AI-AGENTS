"""Сборка контекста вакансии для split-движка.

Порт прод-кода tgApi (`app/common/assistants/screening_context.py`, HEAD e733095).
Единственная адаптация: вместо `ScreeningVacancyDTO` (живёт в app/ — импортировать
нельзя по контракту qa_harness ⊥ app) принимаем `vacancy_info: dict` — ту же форму,
что уже ходит по ai-agents (см. domain/screening/conversation.py). Итоговый ТЕКСТ
контекста воспроизводится 1:1, включая формат зарплатной вилки — Аналитик тюнился
против ровно этой строки.

Один и тот же текст идёт: (1) сид-сообщением в OpenAI-conversation Интервьюера НЕ идёт
(Интервьюер получает урезанный seed), а (2) на вход Аналитику как «КОНТЕКСТ ВАКАНСИИ».
"""

from typing import Any

_SOURCE_BY_TYPE = (
    ("ln", "профиль LinkedIn"),
    ("hh", "резюме HH"),
    ("github", "профиль GitHub"),
    ("mk", "резюме Хабр Карьеры"),
)


def candidate_source(accounts: list | None) -> str:
    """Человекочитаемый источник контакта по аккаунтам кандидата (как в legacy)."""
    accounts = accounts or []
    for type_, label in _SOURCE_BY_TYPE:
        if any(acc.get("type") == type_ for acc in accounts):
            return label
    return ""


def salary_display(min_salary: Any, max_salary: Any) -> str:
    """Отображение зарплатной вилки (порт ScreeningVacancyDTO.get_salary_display)."""
    if min_salary and max_salary:
        return f"от {min_salary} до {max_salary} рублей"
    if min_salary:
        return f"от {min_salary} рублей"
    if max_salary:
        return f"до {max_salary} рублей"
    return ""


def build_context(
    recruiter_name: str,
    candidate_name: str,
    contact_source: str,
    vacancy_info: dict,
) -> str:
    """Полный блок фактов вакансии — ТОЛЬКО для Аналитика (умная модель).

    Содержит секретные поля (вилка, скрытое название, ссылка). Интервьюеру НЕ
    передаётся: он получает урезанный seed (build_interviewer_seed), а нужные
    факты для ответа кандидату Аналитик кладёт в instruction уже отредактированными.
    """
    ci = vacancy_info.get("company_info") or {}
    company_name = vacancy_info.get("company_name") or "СКРЫТО"
    return (
        f"Ваше имя: {recruiter_name}\n"
        f"Имя кандидата: {candidate_name}\n"
        f"Источник контакта кандидата: {contact_source}\n"
        f"Должность: {vacancy_info.get('title', '')}\n"
        f"Название компании: {company_name}\n"
        f"Обязанности: {vacancy_info.get('responsibilities', '')}\n"
        f"Формат работы: {vacancy_info.get('work_format', '')}\n"
        f"Локация: {vacancy_info.get('location', '') or ''}\n"
        f"Описание компании: {ci.get('firm_description', '')}\n"
        f"Ссылка на вакансию: {ci.get('vacancy_url', '')}\n"
        f"Зарплатная вилка: {salary_display(vacancy_info.get('min_salary'), vacancy_info.get('max_salary'))} (НЕ РАСКРЫВАТЬ!)\n"
        "Дополнительные вопросы:\n"
        f"{vacancy_info.get('questions', '') or ''}"
    )


def build_interviewer_seed(recruiter_name: str, candidate_name: str) -> str:
    """Минимальный сид для Интервьюера (least-privilege): только участники.

    Ничего о вакансии (должность, формат, компания, вилка, локация, [questions])
    Интервьюер НЕ знает — все факты приходят фактом в instruction от Аналитика.
    """
    return (
        "Ты — внешний рекрутер по контракту, ведёшь первичный скрининг.\n"
        f"Ваше имя: {recruiter_name}\n"
        f"Имя кандидата: {candidate_name}"
    )

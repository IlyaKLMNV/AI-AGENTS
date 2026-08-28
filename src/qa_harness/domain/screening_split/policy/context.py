"""Контекст вакансии для Наблюдателя — БЕЗ зарплатной вилки.

Принцип П4: секрет не кладут туда, где его потом запрещают. В `..context.build_context` строка
«Зарплатная вилка: … (НЕ РАСКРЫВАТЬ!)» подаётся модели только затем, чтобы промпт тут же запретил её
называть, а считать ею модели прямо запрещено (`screening_analyzer/v2/system.md:202`). Значит она там
не нужна вовсе: сравнение делает код по `salary_claim`.

Порт `..context` НЕ трогаем — он остаётся 1:1 с tgApi для старого движка. Здесь отдельная сборка.
"""

from typing import Any, Optional


def build_observer_context(recruiter_name: str, candidate_name: str, contact_source: str,
                           vacancy_info: dict) -> str:
    """Лейблованный блок фактов. Вилки нет; гео-ограничение появляется, только если задано."""
    company_info = vacancy_info.get("company_info") or {}
    company_name = vacancy_info.get("company_name") or "СКРЫТО"
    geo = (vacancy_info.get("geo_restriction") or "").strip()

    lines = [
        f"Ваше имя: {recruiter_name}",
        f"Имя кандидата: {candidate_name}",
        f"Источник контакта кандидата: {contact_source}",
        f"Должность: {vacancy_info.get('title', '')}",
        f"Название компании: {company_name}",
        f"Обязанности: {vacancy_info.get('responsibilities', '')}",
        f"Формат работы: {vacancy_info.get('work_format', '')}",
        f"Локация: {vacancy_info.get('location', '') or ''}",
        f"Описание компании: {company_info.get('firm_description', '')}",
        f"Ссылка на вакансию: {company_info.get('vacancy_url', '')}",
    ]
    if geo:
        lines.append(f"Гео-ограничение: {geo}")
    lines.append("Дополнительные вопросы:")
    lines.append(f"{vacancy_info.get('questions', '') or ''}")
    return "\n".join(lines)


# Маркеры гео-ограничения. Оно НЕ живёт в отдельном поле: в фикстурах и в проде ограничение пишут
# прямо в «Локацию» — например «Россия, только РФ (работа из-за рубежа невозможна, часовой пояс не
# более +2 к МСК)». Прогон 28.08, сценарий 10: код искал поле `geo_restriction`, не находил и не
# отсеивал, хотя ограничение в контексте было и наблюдатель его видел.
_GEO_MARKERS = (
    "только рф", "только россия", "из-за рубежа", "за рубежом", "часовой пояс",
    "часовому поясу", "тайм-зон", "timezone", "не более +", "территории рф",
)


def has_geo_restriction(vacancy_info: dict) -> bool:
    """Есть ли у вакансии ЯВНОЕ гео-ограничение.

    Проверяет и отдельное поле, и текст локации. Нужна как код-страховка над `facts.geo_blocked`:
    без неё отсев по гео зависел бы только от суждения модели, а из «у вакансии указан город»
    ограничение выводить нельзя (`screening_analyzer/v3/system.md`, раздел ФАКТЫ).
    """
    if (vacancy_info.get("geo_restriction") or "").strip():
        return True
    location = (vacancy_info.get("location") or "").lower()
    return any(marker in location for marker in _GEO_MARKERS)


# Лейблы контекста — их пишет `build_observer_context` (и раньше писал прежний сборщик), поэтому
# блок разбирается обратно надёжно. Нужен гардам: канонический URL, скрытость компании, формат.
_LABELS = {
    "Должность": "title",
    "Название компании": "company_name",
    "Обязанности": "responsibilities",
    "Формат работы": "work_format",
    "Локация": "location",
    "Описание компании": "firm_description",
    "Ссылка на вакансию": "vacancy_url",
    "Гео-ограничение": "geo_restriction",
}


def facts_from_context(context: str) -> dict:
    """Факты вакансии обратно из лейблованного контекста.

    Отдельного поля в документе диалога нет и заводить его незачем: у диалогов, начатых раньше,
    оно всё равно было бы пустым, а контекст есть у всех. Разбор ПО МЕТКЕ устойчив — метки
    порождает наш же код, а не свободный текст.
    """
    facts: dict[str, Any] = {}
    for raw in (context or "").splitlines():
        label, _, value = raw.partition(":")
        key = _LABELS.get(label.strip())
        if key:
            facts[key] = value.strip()
    if facts.get("company_name") or facts.get("firm_description") or facts.get("vacancy_url"):
        facts["company_info"] = {"firm_description": facts.pop("firm_description", ""),
                                 "vacancy_url": facts.pop("vacancy_url", "")}
    return facts


def salary_forms_for(band_min: Optional[int], band_max: Optional[int]) -> tuple[str, ...]:
    """Числовые формы обеих границ вилки — для гарда G9.

    Кандидат мог назвать то же число сам; вычитание его реплик делает сам гард
    (`guards._effective_forbidden`), здесь только полный набор форм.
    """
    from .guards import salary_forms
    forms: list[str] = []
    for bound in (band_min, band_max):
        forms.extend(salary_forms(bound))
    return tuple(dict.fromkeys(forms))


def _typed_band(vacancy_info: dict) -> dict[str, Any]:
    """Типизированная вилка `{min, max, currency}` вместо строки в контексте.

    Валюта обязательна — это точка починки P11: в hh вилка приходит из `hh_vacancy_data["salary"]`,
    где поле `currency` есть, но сегодня не читается, и вилка в тенге сравнивается как рублёвая.
    """
    return {
        "min": vacancy_info.get("min_salary"),
        "max": vacancy_info.get("max_salary"),
        "currency": vacancy_info.get("salary_currency", "RUB"),
    }

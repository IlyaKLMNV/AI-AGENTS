"""Контекст вакансии для Наблюдателя — HH-канал, БЕЗ зарплатной вилки.

Принцип П4 тот же, что в TG: секрет не кладут туда, где его потом запрещают. В `..context.build_context`
строка «Зарплатная вилка: … (НЕ РАСКРЫВАТЬ!)» подавалась модели только затем, чтобы промпт тут же
запретил её называть; сравнение делает код по `salary_claim`.

Поля «Формат работы» здесь нет и в TG-версии тоже: в hh форматы приходят Наблюдателю в STATE
(`allowed_formats`, нормализованы кодом), и второй источник правды о них не нужен.

Порт `..context` НЕ трогаем — он остаётся тем, что читает старый движок.
"""

from qa_harness.domain.screening_split.policy.context import _GEO_MARKERS

from ..context import allowed_formats_of  # noqa: F401 — re-export: движку нужен для init_state

_AGENCY_LITERAL = "рекрутинговое агентство"


def build_observer_context(recruiter_name: str, candidate_name: str, vacancy_info: dict) -> str:
    """Лейблованный блок фактов. Лейблы и порядок совпадают с блоком «ВХОДНЫЕ ДАННЫЕ» промпта
    `screening_analyzer_hh/v3`."""
    company_name = (vacancy_info.get("company_name") or "").strip() or _AGENCY_LITERAL
    description = vacancy_info.get("vacancy_description") or vacancy_info.get("description") or ""
    return (
        f"Ваше имя: {recruiter_name}\n"
        f"Имя кандидата: {candidate_name}\n"
        f"Должность: {vacancy_info.get('title', '')}\n"
        f"Название компании: {company_name}\n"
        f"Локация: {vacancy_info.get('location', '') or ''}\n"
        f"Описание вакансии: {description}\n"
        "Дополнительные вопросы:\n"
        f"{vacancy_info.get('questions', '') or ''}"
    )


def has_geo_restriction(vacancy_info: dict) -> bool:
    """Есть ли у вакансии ЯВНОЕ гео-ограничение.

    В hh оно не живёт в отдельном поле: ограничение пишут в «Локацию» либо в свободный текст
    «Описание вакансии» (`screening_analyzer_hh/v2/system.md`, KO-3). Проверяем оба плюс отдельное
    поле, если канал его когда-нибудь заведёт.

    Маркер в описании — вход шумный (это несколько абзацев текста вакансии), поэтому одного его мало:
    отсев `KO_LOCATION_GEO` требует ВТОРОГО совпадения — Наблюдатель должен увидеть нарушение в
    реплике (`facts.geo_blocked`). Это ровно та «работа по двойному совпадению», которую задача Б3
    держит открытой: политика по кандидатам за границей пока не решена, и здесь она портируется как
    есть, а не изобретается заново.
    """
    if (vacancy_info.get("geo_restriction") or "").strip():
        return True
    haystack = " ".join([
        str(vacancy_info.get("location") or ""),
        str(vacancy_info.get("vacancy_description") or vacancy_info.get("description") or ""),
    ]).lower()
    return any(marker in haystack for marker in _GEO_MARKERS)

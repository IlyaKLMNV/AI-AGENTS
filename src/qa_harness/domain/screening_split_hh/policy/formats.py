"""Мультиформат: чистые функции над `state.allowed_formats` и накопленными ответами `state.formats`.

Отдельный модуль, потому что читают его двое — таблица правил (отсев) и сборка вопроса (о каком
формате спрашивать следующим), а класть общее в один из них значило бы завести цикл импортов.

Главное отличие канала от TG: у вакансии допустимо НЕСКОЛЬКО форматов, и кандидат подходит, если ему
подходит **хотя бы один**. Поэтому «не готов к офису» — это снятый вариант, а не отказ: пока среди
допустимых остаётся неспрошенный присутственный формат, проверка продолжается.

`state.formats` копит ответы через диалог (`{'ON_SITE': 'no', 'HYBRID': 'yes'}`); Observation отдаёт
только сказанное на текущем ходе.
"""

from typing import Optional

from .observation import PRESENCE_FORMATS

HUMAN: dict[str, str] = {
    "ON_SITE": "работа на месте работодателя",
    "REMOTE": "удалённый формат",
    "HYBRID": "гибридный формат",
    "FIELD_WORK": "разъездной формат",
}

# Та же четвёрка после слова «формат»: «ещё один формат работы — гибридный». Без второй формы
# получается «формат — гибридный формат», а у ON_SITE ещё и рассогласование падежа.
HUMAN_SHORT: dict[str, str] = {
    "ON_SITE": "на месте работодателя",
    "REMOTE": "удалённый",
    "HYBRID": "гибридный",
    "FIELD_WORK": "разъездной",
}


def allowed(state: dict) -> list[str]:
    return [f for f in (state.get("allowed_formats") or []) if f in HUMAN]


def answers(state: dict) -> dict[str, str]:
    return {k: v for k, v in (state.get("formats") or {}).items() if v in ("yes", "no")}


def presence_allowed(state: dict) -> list[str]:
    """Присутственные форматы вакансии в порядке `allowed_formats` — их и проверяет `format_check`."""
    return [f for f in allowed(state) if f in PRESENCE_FORMATS]


def confirmed_any(state: dict) -> bool:
    """Кандидат подтвердил готовность хотя бы к одному ДОПУСТИМОМУ формату."""
    said = answers(state)
    return any(said.get(f) == "yes" for f in allowed(state))


def refused_all(state: dict) -> bool:
    """Отказался от ВСЕХ допустимых форматов — это и есть несоответствие по формату.

    Пустой список допустимых сюда не проходит: отсеивать не за что.
    """
    said = answers(state)
    fmts = allowed(state)
    return bool(fmts) and all(said.get(f) == "no" for f in fmts)


def next_presence_to_ask(state: dict) -> Optional[str]:
    """Первый присутственный формат, о котором кандидат ещё не высказался.

    Порядок — как в `allowed_formats` (его нормализует `..state.normalize_work_formats`, сохраняя
    порядок вакансии). Все высказались — None: спрашивать больше нечего, ход заберут правила отсева
    либо `format_check` уже закрыт.
    """
    said = answers(state)
    for fmt in presence_allowed(state):
        if fmt not in said:
            return fmt
    return None


def field_work_only(state: dict) -> bool:
    """`FIELD_WORK` — единственный допустимый формат: отказ от него равен отказу от вакансии."""
    return allowed(state) == ["FIELD_WORK"]

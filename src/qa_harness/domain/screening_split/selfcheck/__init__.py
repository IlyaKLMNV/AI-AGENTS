"""Офлайн-гейт ядра `policy` (TG-канал): детерминированные проверки без сети и без LLM.

pytest на код харнесса в этом репозитории не держим (CLAUDE.md), поэтому единственный гейт — эти
наборы; гоняет их `--offline` раннера `screening_split`, провал валит прогон.

Лежит РЯДОМ с `policy/`, а не внутри: пакет `policy/` целиком переносится в `tgApi` и `eggplant-api`,
и тестовый код в продуктовые репозитории ехать не должен.

Контракт набора: `checks() -> list[(имя, ok, деталь)]`. Набор ничего не печатает и не бросает.
"""

from typing import Callable, List, Tuple

from . import adapter, agenda, budgets, context, guards, migration, observation, reasons, rules, salary
from .collect import Checks, Row

# Порядок — от общего к частному: сначала контракты (реестр, бюджеты, наблюдение), потом решения
# (зарплата, правила, повестка), потом периметр (гарды, контекст, миграция, переигрывание).
SUITES: Tuple[Tuple[str, Callable[[], List[Row]]], ...] = (
    ("реестр причин", reasons.checks),
    ("бюджеты и счётчики", budgets.checks),
    ("контракт Observation", observation.checks),
    ("зарплатный контракт (salary_claim)", salary.checks),
    ("таблица правил R1-R11", rules.checks),
    ("повестка хода (Р18)", agenda.checks),
    ("шлюз гардов G0-G7", guards.checks),
    ("контекст Наблюдателя и гео", context.checks),
    ("ленивая миграция документа", migration.checks),
    ("переходник записанных трасс", adapter.checks),
)

__all__ = ["SUITES", "Checks", "Row"]

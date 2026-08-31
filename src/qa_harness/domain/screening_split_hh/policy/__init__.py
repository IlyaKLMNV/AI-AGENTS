"""Ядро политики split-скрининга — HH-канал: «Наблюдение → Политика → Речь».

Порт `screening_split/policy/` (TG) под hh. Разделение ответственности то же: модель РАЗЛИЧАЕТ
(`Observation` — сигналы с цитатами, факты, ответы, salary_claim, черновик ответа), код РЕШАЕТ
(`decide()` — зарплата → счётчики → упорядоченная таблица правил → `TurnPlan`).

Канальная дельта, ради которой пакет существует отдельно от TG:
`observation` (нет `contact_source`, готовность к формату — по конкретным форматам) ·
`budgets` (нет `contact_source`, есть `field_work`) · `reasons` (`KO_FORMAT`/`KO_LOCATION`/
`KO_LOCATION_GEO`) · `rules` (мультиформат и выбор ключа отсева) · `core` (повестка из четырёх
пунктов, выбор формата для вопроса) · `context` (контекст hh без вилки) · `engine` (без ленивой
миграции). Канало-независимое — гарды, гейты зарплаты, класс `Budget`, таблица приоритета
терминальных сигналов, `Reason`, `TurnPlan` — импортируется из TG-пакета.

Потребитель порта — `eggplant-api`, где ядра ещё нет (план, hh-контур, п. 5).
"""

from .budgets import EVENT_BUDGETS, REASK_BUDGETS, STALL_BUDGET
from .core import DecideContext, TurnPlan, decide, state_for_prompt
from .engine import PolicyEngine, PolicyResult
from .observation import (
    NONTERMINAL_SIGNALS,
    PRESENCE_FORMATS,
    SIGNAL_TO_COUNTER,
    TERMINAL_PRIORITY,
    TERMINAL_SIGNAL_REASON,
    Observation,
    formats_ready,
    parse_observation,
)
from .observer import ScreeningObserver
from .rules import RULES

__all__ = [
    "DecideContext",
    "EVENT_BUDGETS",
    "NONTERMINAL_SIGNALS",
    "Observation",
    "PRESENCE_FORMATS",
    "PolicyEngine",
    "PolicyResult",
    "REASK_BUDGETS",
    "RULES",
    "SIGNAL_TO_COUNTER",
    "STALL_BUDGET",
    "ScreeningObserver",
    "TERMINAL_PRIORITY",
    "TERMINAL_SIGNAL_REASON",
    "TurnPlan",
    "decide",
    "formats_ready",
    "parse_observation",
    "state_for_prompt",
]

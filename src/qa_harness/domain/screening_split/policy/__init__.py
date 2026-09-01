"""Чистое ядро политики split-скрининга: «Наблюдение → Политика → Речь».

Предложение из docs/screening_split/rearchitecture.html, решения — docs/screening_split/
decisions_rearchitecture.md (Р1–Р8). Пакет НИЧЕГО не вызывает по сети и не знает про стор:
`decide()` — чистая функция, весь ввод-вывод остаётся в канале (tgApi / eggplant / раннер).

Разделение ответственности (принцип П1):
    модель РАЗЛИЧАЕТ  — `Observation`: сигналы с цитатами, факты, ответы, salary_claim, черновик ответа;
    код РЕШАЕТ        — `decide()`: зарплата → счётчики → упорядоченная таблица правил → `TurnPlan`.

Отличие от действующего движка: у модели больше нет полей `next_action`/`script_key`/
`instruction`/`asking`/`event`, поэтому решения, с которым код мог бы не согласиться, не существует —
и перерешивания хода (три ветки второго вызова Аналитика) исчезают по построению.

Пакет НЕ подключён к раннерам и ничего в проде не меняет: он проверяется офлайн переигрыванием
записанных трасс (`qa_harness.runners.policy_replay`).
"""

from .budgets import EVENT_BUDGETS, REASK_BUDGETS, STALL_BUDGET, Budget
from .core import DecideContext, TurnPlan, decide, state_for_prompt
from .observation import (
    NONTERMINAL_SIGNALS,
    SIGNAL_TO_COUNTER,
    TERMINAL_PRIORITY,
    TERMINAL_SIGNAL_REASON,
    Observation,
    parse_observation,
)
from .rules import RULES

__all__ = [
    "Budget",
    "DecideContext",
    "EVENT_BUDGETS",
    "NONTERMINAL_SIGNALS",
    "Observation",
    "REASK_BUDGETS",
    "RULES",
    "SIGNAL_TO_COUNTER",
    "STALL_BUDGET",
    "TERMINAL_PRIORITY",
    "TERMINAL_SIGNAL_REASON",
    "TurnPlan",
    "decide",
    "parse_observation",
    "state_for_prompt",
]

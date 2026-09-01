"""Офлайн-гейт ядра `policy` (HH-канал): канальная дельта поверх общего с TG.

Канало-независимое (зарплатный контракт, гарды, реестр как механика, наблюдение, бюджеты как
семантика) проверяет TG-набор `screening_split/selfcheck` — он же и импортируется раннером первым.
Здесь только то, чего в TG нет или что в hh устроено иначе.

Контракт набора тот же: `checks() -> list[(имя, ok, деталь)]`.
"""

from typing import Callable, List, Tuple

from qa_harness.domain.screening_split.selfcheck.collect import Row

from . import agenda, channel

SUITES: Tuple[Tuple[str, Callable[[], List[Row]]], ...] = (
    ("канальная дельта hh (реестр, сигналы, паритет бюджетов)", channel.checks),
    ("повестка hh: мультиформат и локация (Р18)", agenda.checks),
)

__all__ = ["SUITES"]

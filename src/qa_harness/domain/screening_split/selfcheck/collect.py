"""Сборщик результатов одного набора проверок.

Контракт всех наборов: `checks() -> list[(имя, ok, деталь)]`. Набор ничего не печатает и ничего не
бросает — печатает и суммирует раннер. Деталь заполняется ТОЛЬКО фактическим значением: сообщение
«ждали X» и так восстанавливается из имени, а факт из вывода иначе не достать.
"""

from typing import Any, List, Tuple

Row = Tuple[str, bool, str]


class Checks:
    """Накопитель строк набора."""

    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: List[Row] = []

    def add(self, name: str, ok: Any, detail: str = "") -> None:
        self.rows.append((name, bool(ok), str(detail)))

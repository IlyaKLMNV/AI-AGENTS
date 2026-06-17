"""Seeded-сэмплер поверхностного стиля реплик кандидата (разнообразие, не суть).

Сценарий задаёт ЧТО говорит кандидат (триггер), variety — КАК (тон/объём/манера). Это даёт разные
прогоны одного сценария (`--variants N`) без подмены сути. Стиль выбирается детерминированно по seed,
один на весь диалог (персона внутри разговора стабильна). Перенос идеи variant-knobs из легаси
screening_autofill (там noise→volume/indirect/chitchat), упрощённый под реплики кандидата.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

TONES = ["нейтральный", "сухой", "раздражённый", "саркастичный", "дружелюбно-болтливый"]
VERBOSITY = ["очень коротко, одной фразой", "средне, 1-2 предложения", "развёрнуто, 3-4 предложения"]


@dataclass
class VariantStyle:
    tone: str
    verbosity: str
    quirks: bool  # лёгкий сленг/опечатки

    def hint(self) -> str:
        """Текст-подсказка для генератора (мягкая — не ломать суть сценария)."""
        parts = [f"тон: {self.tone}", f"объём: {self.verbosity}"]
        if self.quirks:
            parts.append("допустимы лёгкий сленг и редкие опечатки, без перебора")
        return ("Стилистика реплики (применяй, только если не противоречит сути сценария): "
                + "; ".join(parts) + ".")


class VariantSampler:
    """Детерминированный по seed сэмплер стилей. at(i) воспроизводим при том же seed."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed if seed is not None else 0

    def at(self, index: int) -> VariantStyle:
        rng = random.Random(f"{self._seed}:{index}")
        return VariantStyle(
            tone=rng.choice(TONES),
            verbosity=rng.choice(VERBOSITY),
            quirks=rng.random() < 0.35,
        )

    def sample(self, n: int) -> List[VariantStyle]:
        return [self.at(i) for i in range(max(0, n))]

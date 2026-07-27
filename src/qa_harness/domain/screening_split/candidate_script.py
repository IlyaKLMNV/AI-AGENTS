"""Скриптовые (детерминированные) реплики кандидата — замена LLM-генерации для
«односложных» сценариев (зарплата / триггеры / пауза / gibberish).

Зачем: для сценариев, где вход — короткая фраза/значение, LLM избыточен (дороже,
недетерминирован, плодит «литературщину» и не ту магнитуду). Скриптовые входы:
- дёшевы (без вызова генератора) и воспроизводимы;
- для зарплаты величина берётся ИЗ ВИЛКИ (`{above_max}`/`{below_min}`/`{in_band}`/…),
  поэтому правильность известна заранее → сценарий становится детерминированным гейтом;
- `{gibberish}` — псевдослучайный мусор (seed+index+variant), настоящая бессвязность.

Рецепты — данные: `tests/fixtures/screening_split/candidate_inputs.yaml`.
`turn` = строка ИЛИ список альтернатив (выбор по `variant % len` — лёгкое разнообразие).
"""

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

_GIB_SYLL = ["фы", "ва", "про", "лж", "ук", "ыв", "апр", "жэ", "ёк", "бю", "ащ",
             "зхц", "йцу", "кен", "гш", "мив", "тьб", "чяю", "ощ", "клм"]


def load_candidate_inputs(path: Path) -> Dict[int, Dict[str, Any]]:
    """candidate_inputs.yaml → {index: recipe}. Нет файла — пусто (скриптовых входов нет)."""
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: Dict[int, Dict[str, Any]] = {}
    for e in (data.get("scenarios") or []):
        if isinstance(e, dict) and e.get("index") is not None:
            out[int(e["index"])] = {k: v for k, v in e.items() if k != "index"}
    return out


def _gibberish(rng: random.Random) -> str:
    words = rng.randint(4, 8)
    return " ".join("".join(rng.choice(_GIB_SYLL) for _ in range(rng.randint(1, 3))) for _ in range(words))


def _salary_values(vacancy_info: Dict[str, Any]) -> Dict[str, int]:
    lo = int(vacancy_info.get("min_salary") or 200000)
    hi = int(vacancy_info.get("max_salary") or 280000)
    return {
        "above_max": hi + 50000,          # заведомо выше max → KO по конкретной сумме
        "below_min": max(lo - 50000, 10000),  # ниже min → закрыть молча (не KO)
        "in_band": (lo + hi) // 2,        # в вилке → закрыть
        "range_lo": lo + 10000,           # диапазон целиком в вилке
        "range_hi": hi - 10000,
    }


def build_scripted_turns(recipe: Dict[str, Any], vacancy_info: Dict[str, Any],
                         *, variant: int, seed: int, index: int) -> List[str]:
    """Собрать список реплик кандидата из рецепта: подставить величины из вилки и gibberish."""
    rng = random.Random((seed or 0) * 100000 + index * 100 + variant)
    vals = _salary_values(vacancy_info)
    turns: List[str] = []
    for raw in recipe.get("turns") or []:
        t = raw[variant % len(raw)] if isinstance(raw, list) and raw else raw
        s = str(t)
        while "{gibberish}" in s:
            s = s.replace("{gibberish}", _gibberish(rng), 1)
        for k, v in vals.items():
            s = s.replace("{" + k + "}", str(v))
        turns.append(s)
    return turns

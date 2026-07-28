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


# Директивы генератору (Фаза 2): категория зарплаты → короткая инструкция «что сказать» с
# точным значением из вилки. Идёт в CandidateConstraints.must_convey — LLM формулирует живо,
# но НЕ меняет магнитуду/категорию (лечит «случайную сумму» генератора).
_SALARY_DIRECTIVE = {
    "above_max": "назови КОНКРЕТНУЮ зарплату {above_max} рублей на руки в месяц (это выше бюджета), держись уверенно",
    "below_min": "назови конкретную зарплату {below_min} рублей на руки в месяц",
    "in_band": "назови конкретную зарплату {in_band} рублей на руки в месяц",
    "ambiguous": "назови сумму голым числом без единиц, например «260» — БЕЗ «тыс/руб» и без периода",
    "currency": "назови зарплату в долларах, например «от 4000 до 5000 долларов в месяц»",
    "hourly": "назови зарплату почасовой ставкой, например «800 рублей в час»",
    "gross": "назови зарплату как gross (до вычета налога), например «{above_max} gross»",
}


def _resolve(text: str, vacancy_info: Dict[str, Any]) -> str:
    """Подставить плейсхолдеры вилки ({above_max}/{below_min}/{in_band}/{range_lo}/{range_hi})
    и {location} (город вакансии). Общий резолвер для salary_directive и convey-директив."""
    for k, v in _salary_values(vacancy_info).items():
        text = text.replace("{" + k + "}", str(v))
    return text.replace("{location}", str(vacancy_info.get("location") or ""))


def salary_directive(category: str, vacancy_info: Dict[str, Any]) -> List[str]:
    """must_convey-директива по категории зарплаты со значением из вилки (пусто для неизвестной)."""
    tmpl = _SALARY_DIRECTIVE.get(category or "")
    return [_resolve(tmpl, vacancy_info)] if tmpl else []


def resolve_convey(items: Any, vacancy_info: Dict[str, Any]) -> List[str]:
    """Пер-сценарные `convey`-директивы генератору (что ОБЯЗАН передать кандидат) с подстановкой
    вилки/{location}. Даёт контекст там, где категории зарплаты мало (формат/локация: город
    кандидата относительно города вакансии; гео-готовность). Пустые строки отбрасываются."""
    out: List[str] = []
    for it in (items or []):
        s = _resolve(str(it), vacancy_info)
        if s.strip():
            out.append(s)
    return out


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

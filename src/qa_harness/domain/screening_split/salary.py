"""Зарплата: пересчёт названной кандидатом суммы и вердикт по вилке — без LLM.

РАСПОЗНАЁТ сумму Аналитик и отдаёт полем `Decision.salary_claim` (чьи деньги, какое из чисел
ожидание, диапазон или порог, net/gross, валюта, период). Этот модуль СЧИТАЕТ и РЕШАЕТ: масштаб,
курс, период, gross→net, сравнение с вилкой. Данные пересчёта — в `salary_rules`.

ПОРТ tgApi app/common/screening/salary.py — держать 1:1 (отличается только путь импорта).
"""
import re

from .salary_rules import (
    BARE_THOUSANDS_BELOW,
    NDFL_BRACKETS,
    PERIOD_TO_MONTH,
    RATE_TO_RUB,
    SALARY_RULES_VERSION,
)

ABSENT = "absent"          # про деньги в реплике речи нет
UNUSABLE = "unusable"      # речь есть, но сумму использовать нельзя (уточняет Аналитик)
ACTIONABLE = "actionable"  # можно пересчитать и сравнить с вилкой

SUBJECTS = frozenset({"own_expectation", "own_current", "third_party"})
FORMS = frozenset({"exact", "range", "at_least", "at_most", "relative_only", "not_stated"})
SCALES = {"none_stated": 1, "unit": 1, "thousand": 1_000, "million": 1_000_000}
TAXES = frozenset({"net", "gross", "unspecified"})

_FORMS_WITHOUT_AMOUNT = frozenset({"relative_only", "not_stated"})  # «+15%», «обсуждаемо»
_NUMERAL_WORDS = ("полтора", "миллион", "тысяч", "лям")  # у «полтора миллиона» цифр в цитате нет


class SalaryValue:
    """Рубли за месяц. `max is None` — порог «от X»; `min is None` — только «до Y»."""

    __slots__ = ("min", "max", "applied", "rules_version")

    def __init__(self, minimum: int | None, maximum: int | None, applied: list[str]):
        self.min = minimum
        self.max = maximum
        self.applied = applied
        self.rules_version = SALARY_RULES_VERSION

    def __repr__(self) -> str:
        return f"SalaryValue(min={self.min}, max={self.max}, applied={self.applied})"


def _norm_text(text: str) -> str:
    """Единая форма для сверки цитаты: регистр, неразрывные пробелы, длинные тире."""
    t = (text or "").lower().replace(" ", " ").replace(" ", " ")
    t = re.sub(r"[—–‒−]", "-", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_number(value) -> bool:
    """Число и не bool: `True` — это `int`, и без проверки он прошёл бы как сумма 1."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def read_claim(decision: dict) -> dict | None:
    """Не dict → None: поля нет (промпт старой версии) либо пришёл мусор. Негодный claim
    Decision НЕ инвалидирует: перегенерация всего хода дороже одного уточняющего вопроса."""
    claim = (decision or {}).get("salary_claim")
    return claim if isinstance(claim, dict) else None


def claim_status(claim: dict | None, message: str) -> str:
    """`absent` · `unusable` · `actionable`. `unusable` — не ошибка: Аналитик в том же ходе
    уже задал уточняющий вопрос, код просто не закрывает пункт и не сравнивает с вилкой."""
    if claim is None:
        return ABSENT

    # Значение вне enum не додумываем.
    if claim.get("subject") not in SUBJECTS or claim.get("form") not in FORMS:
        return UNUSABLE
    if claim.get("scale") not in SCALES or claim.get("period") not in PERIOD_TO_MONTH:
        return UNUSABLE
    if claim.get("tax") not in TAXES:
        return UNUSABLE

    # Гейт против выдуманной суммы: цитата обязана найтись в реплике.
    quote = _norm_text(claim.get("quote") or "")
    if not quote or quote not in _norm_text(message):
        return UNUSABLE

    if claim.get("form") in _FORMS_WITHOUT_AMOUNT:
        return UNUSABLE
    if claim.get("subject") != "own_expectation":
        return UNUSABLE
    if claim.get("currency") not in RATE_TO_RUB:  # `other` — пересчитать нечем
        return UNUSABLE

    lo, hi = claim.get("amount_min"), claim.get("amount_max")
    if not _is_number(lo) and not _is_number(hi):
        return UNUSABLE
    if not re.search(r"\d", quote) and not any(w in quote for w in _NUMERAL_WORDS):
        return UNUSABLE

    return ACTIONABLE


def _gross_to_net_month(amount_month: int) -> int:
    """gross → на руки: по прогрессивной шкале от годового дохода."""
    annual, net, taken = amount_month * 12, 0.0, 0
    for threshold, rate in NDFL_BRACKETS:
        cap = annual if threshold is None else min(annual, threshold)
        chunk = cap - taken
        if chunk <= 0:
            break
        net += chunk * (1 - rate)
        taken = cap
        if taken >= annual:
            break
    return round(net / 12)


def normalize(claim: dict) -> SalaryValue | None:
    """Рубли за месяц. None — claim противоречив (см. конец). Вызывать только на `ACTIONABLE`."""
    form = claim["form"]
    lo = claim.get("amount_min") if _is_number(claim.get("amount_min")) else None
    hi = claim.get("amount_max") if _is_number(claim.get("amount_max")) else None

    if form == "at_least":
        hi = None
    elif form == "at_most":
        lo = None
    elif form == "exact":
        lo = hi = lo if lo is not None else hi
    # `range` берём как пришло: пустая граница остаётся открытой

    applied: list[str] = []
    scale = SCALES[claim["scale"]]
    bare = claim["scale"] == "none_stated"

    def value_of(num: float | None) -> float | None:
        """Масштаб словом · единица вплотную к числу (`unit`) — буквально · голое малое = тысячи."""
        if num is None:
            return None
        if not bare:
            return num * scale
        if num < BARE_THOUSANDS_BELOW:
            if "scale_thousands" not in applied:
                applied.append("scale_thousands")
            return num * 1_000
        return num

    lo, hi = value_of(lo), value_of(hi)

    currency = claim["currency"]
    if currency != "RUB":
        rate = RATE_TO_RUB[currency]
        lo = lo * rate if lo is not None else None
        hi = hi * rate if hi is not None else None
        applied.append("currency")

    period = claim["period"]
    if period != "month":
        factor = PERIOD_TO_MONTH[period]
        lo = lo * factor if lo is not None else None
        hi = hi * factor if hi is not None else None
        applied.append("period")

    lo_i = round(lo) if lo is not None else None
    hi_i = round(hi) if hi is not None else None

    if claim["tax"] == "gross":
        lo_i = _gross_to_net_month(lo_i) if lo_i is not None else None
        hi_i = _gross_to_net_month(hi_i) if hi_i is not None else None
        applied.append("gross_to_net")

    # Claim противоречив: обе границы пустые (форма обнулила сумму) либо перепутаны (min > max —
    # «250-400» как min=400/max=250 дало бы отсев по 400). Проверки структурные, не по величине.
    if lo_i is None and hi_i is None:
        return None
    if lo_i is not None and hi_i is not None and lo_i > hi_i:
        return None

    return SalaryValue(lo_i, hi_i, applied)


def compare_with_band(value: SalaryValue, band_min: int | None, band_max: int | None) -> str:
    """`ko` · `fits`. Отказ только если даже НИЖНЯЯ граница ожиданий выше нашего максимума:
    «250-400» при вилке до 280 — не отказ, кандидат готов и на 250. Полосы терпимости нет.

    `band_min` не используется намеренно: сумма ниже минимума вилки отказом не является никогда.
    Параметр оставлен, чтобы правило было видно на вызове.
    """
    if not band_max:
        return "fits"          # вилки нет — сравнивать не с чем
    if value.min is None:
        return "fits"          # «до Y»: минимум неизвестен и заведомо не выше максимума
    return "ko" if value.min > band_max else "fits"

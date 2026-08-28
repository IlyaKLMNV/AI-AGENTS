"""Односторонние гейты зарплатного вердикта (решение Р1).

Понижают ТОЛЬКО `ko` → уточняющий вопрос и никогда не превращают `fits` в отказ. Асимметрия
обязательна: ложный отказ необратим (кандидату написали «бюджет не позволяет»), а ложное «проходит»
ловится на живом интервью.

Гейтов два. Третий — «потолок правдоподобия» (`value.min > band_max × 4`) — обсуждался и отклонён:
единственная проверка с магической константой, и ошибка в `period`, ради которой она задумывалась, ни
в одном прогоне не встречалась. Гипотезу без данных не вводим.

Почему гейты маркерные, а не числовые: самая частая наблюдённая ошибка модели — «240000 рублей»
приходит как `240 + thousand` — даёт ПРАВИЛЬНЫЙ результат после пересчёта. Гейт «цифры из `amount_min`
обязаны дословно найтись в цитате» ломал бы её, а заодно и «полтора миллиона», где цифр в реплике нет
вовсе (`..salary._NUMERAL_WORDS`).
"""

import re

# Слова-масштабы: если модель поставила thousand/million, хоть одно обязано звучать в реплике.
_SCALE_MARKERS = ("тыс", "тысяч", "т.р", "тр.", "к)", "млн", "миллион", "лям", "лимон", "kk", "кк")
# Отдельно — «к» вплотную к числу: «260к», «300 к».
_SCALE_K_RE = re.compile(r"\d\s*к\b", re.IGNORECASE)

_CURRENCY_MARKERS: dict[str, tuple[str, ...]] = {
    "USD": ("$", "usd", "доллар", "бакс", "долар"),
    "EUR": ("€", "eur", "евро"),
    "KZT": ("₸", "kzt", "тенге"),
}


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace(" ", " ").split())


_SCALES = {"thousand": 1_000, "million": 1_000_000}


def scale_marker_missing(claim: dict) -> bool:
    """Модель поставила масштаб словом, а слова-масштаба в цитате нет.

    Ловит `million` вместо `thousand` — ×1000 при правдоподобном результате.

    ВАЖНОЕ ИСКЛЮЧЕНИЕ, найденное прогоном 28.08 (сценарий 58). Модель часто **раскладывает**
    число: «330000 gross» приходит как `330 + thousand`. Это не ошибка — произведение равно тому,
    что кандидат и написал. Гейт без этого исключения понижал вердикт на самой массовой форме записи
    и глушил законный отсев: 330 000 gross → 287 тыс. на руки при вилке до 280 тыс. — это `ko`,
    а мы превращали его в переспрос.

    Поэтому: если `amount × scale` дословно встречается в цитате, множитель модель не выдумала.
    """
    scale = claim.get("scale")
    if scale not in _SCALES:
        return False
    quote = _norm(claim.get("quote") or "")
    if not quote:
        return True
    if any(m in quote for m in _SCALE_MARKERS) or _SCALE_K_RE.search(quote):
        return False

    # Раскладка, а не домножение: число после применения масштаба есть в реплике буквально.
    digits = re.sub(r"[^\d]", "", quote)
    for bound in (claim.get("amount_min"), claim.get("amount_max")):
        if isinstance(bound, (int, float)) and not isinstance(bound, bool) and bound > 0:
            product = bound * _SCALES[scale]
            if product == int(product) and str(int(product)) in digits:
                return False
    return True


def currency_marker_missing(claim: dict) -> bool:
    """Модель поставила валюту, отличную от рубля, а маркера валюты в цитате нет.

    Ловит `USD` вместо `RUB` — ×80 при правдоподобном результате.
    """
    currency = claim.get("currency")
    if currency in (None, "RUB"):
        return False
    markers = _CURRENCY_MARKERS.get(currency)
    if not markers:
        return False  # `other` до сюда не доходит: claim с ним признаётся непригодным раньше
    quote = _norm(claim.get("quote") or "")
    return not any(m in quote for m in markers)


def downgrade_ko(claim: dict) -> list[str]:
    """Список сработавших гейтов. Непустой → вердикт `ko` понижается до уточняющего вопроса.

    Вызывать ТОЛЬКО при `verdict == "ko"`: на `fits` гейты не смотрят по построению.
    """
    failed: list[str] = []
    if scale_marker_missing(claim):
        failed.append("scale_marker")
    if currency_marker_missing(claim):
        failed.append("currency_marker")
    return failed

"""Ленивая миграция документа диалога при загрузке (решение Р7).

Новый движок подменяет старый ЖЁСТКО, без по-диалогового переключения, и диалоги, начатые до
подмены, дорабатывают уже на нём. Три вещи в их документах несовместимы с новым контуром, и все три
чинятся здесь — при первой же загрузке, до вызова наблюдателя.

Почему лениво, а не скриптом заранее: скрипт, прогнанный до выкатки, оставляет окно, в котором старый
код ещё создаёт диалоги с вилкой в контексте, а миграция по ним уже прошла. При ленивой миграции окна
нет вообще. Батч-скрипт полезен ПОСЛЕ релиза — как отчёт по остаткам, а не как условие запуска.

Что делается и почему каждое обязательно:

1. **Вырезать из `context` строку с зарплатной вилкой.** Новый промпт короче именно потому, что вилки
   в контексте нет: раздел «СЕКРЕТ ВИЛКИ» сжат до одной строки. Подсунуть такому промпту старый
   контекст с вилкой — ослабить защиту ровно там, где секрет присутствует.

2. **Записать типизированный `salary_band`.** Если ключа нет — разобрать числа из контекста, ОДИН раз,
   здесь же, до вырезания строки. Без этого у старых диалогов вилка окажется пустой,
   `compare_with_band` начнёт всегда возвращать «проходит», и **зарплатный отсев тихо отключится** —
   в отчётности это никак не видно.

3. **Обнулить `last_asking`.** У поля сменился автор: писала модель (самоотчёт `decision.asking`),
   станет писать код (фокус, по которому реально выдан вопрос). Сравнивать новый фокус со старым
   значением нельзя — счётчик начислит переспрос, которого не было, и приблизит `STOP_SALARY_DEMAND`.
   Обнуление стоит максимум одного несчитанного переспроса: «завершит позже» лучше, чем «раньше».

Удаление строки ПО МЕТКЕ безопасно и не имеет ничего общего с хрупкостью `_BAND_RE` в tgApi: тот
разбирает из свободного текста числа и молча возвращает пустоту при смене формулировки, а мы просто
удаляем строку целиком. Разбор чисел здесь тоже есть, но он однократный и его провал ВИДЕН
(`band_unparsed` в отчёте миграции), а не растворяется в рантайме.
"""

import re
from typing import Any, Optional

SCHEMA_VERSION = 2

# Метка строки вилки — её ставит `..context.build_context` («Зарплатная вилка: …»).
_BAND_LINE_RE = re.compile(r"^\s*Зарплатная вилка:.*$", re.IGNORECASE | re.MULTILINE)

# Разбор границ из той же строки. Формы даёт `..context.salary_display`:
# «от X до Y рублей» · «от X рублей» · «до Y рублей».
_BAND_VALUES_RE = re.compile(
    r"зарплатная вилка:\s*(?:от\s*([\d\s]+))?(?:[^\d]*до\s*([\d\s]+))?", re.IGNORECASE)

_COUNTER_KEYS = ("bot_check", "gibberish", "salary_info", "demand", "contact_source", "pause")


def _int_or_none(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    return int(digits) if digits else None


def parse_band(context: str) -> dict:
    """Границы вилки из строки контекста. Пустой dict — разобрать не удалось."""
    match = _BAND_VALUES_RE.search(context or "")
    if not match:
        return {}
    low, high = _int_or_none(match.group(1)), _int_or_none(match.group(2))
    if low is None and high is None:
        return {}
    return {"min": low, "max": high, "currency": "RUB"}


def strip_band_line(context: str) -> str:
    """Удаляет строку вилки по метке, не трогая остальное."""
    cleaned = _BAND_LINE_RE.sub("", context or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip("\n")


def _upgrade_state(state: dict) -> bool:
    """State v1 → v2: добавить недостающие поля дефолтами, обнулить `last_asking`."""
    changed = False
    defaults = {"salary_reasks": 0, "format_reasks": 0, "no_progress": 0,
                "last_asked": None, "spoken": [], "closed_reason": None}
    for key, value in defaults.items():
        if key not in state:
            state[key] = value if not isinstance(value, list) else []
            changed = True

    counters = state.setdefault("counters", {})
    for key in _COUNTER_KEYS:
        if key not in counters:
            counters[key] = 0
            changed = True

    if state.get("last_asking") is not None:
        state["last_asking"] = None
        changed = True
    return changed


def upgrade(doc: dict) -> dict:
    """Мутирует документ на месте. Возвращает отчёт о том, что сделано.

    Идемпотентна: повторный вызов на уже мигрированном документе ничего не меняет.
    """
    report: dict[str, Any] = {"migrated": False, "band_source": None, "band_unparsed": False,
                              "context_stripped": False, "state_upgraded": False}
    if doc.get("schema") == SCHEMA_VERSION:
        return report

    context = doc.get("context") or ""

    # ПОРЯДОК ОБЯЗАТЕЛЕН: сначала достать числа, потом вырезать строку.
    band = doc.get("salary_band") or {}
    if band.get("min") is None and band.get("max") is None:
        parsed = parse_band(context)
        if parsed:
            doc["salary_band"] = parsed
            report["band_source"] = "context"
        else:
            report["band_unparsed"] = True
    else:
        band.setdefault("currency", "RUB")
        doc["salary_band"] = band
        report["band_source"] = "document"

    stripped = strip_band_line(context)
    if stripped != context:
        doc["context"] = stripped
        report["context_stripped"] = True

    state = doc.get("state")
    if isinstance(state, dict):
        report["state_upgraded"] = _upgrade_state(state)

    doc["schema"] = SCHEMA_VERSION
    report["migrated"] = True
    return report

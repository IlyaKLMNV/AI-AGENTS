"""Контракт `Observation` — HH-канал.

Дельта к `screening_split/policy/observation.py` (TG) ровно в двух местах:

- **нет сигнала `contact_source`**: в hh-канале нет ни события, ни скрипта источника контакта
  (`..state.COUNTER_KEYS`). Вопрос «откуда мои данные» на hh-отклике смысла не имеет,
  недоверие приходит через `fraud_check`;
- **готовность к формату — по КОНКРЕТНЫМ форматам** (`facts.formats_ready`), а не одним `yes/no` на
  всю вакансию: допустимых форматов у hh-вакансии может быть несколько, и «подходит хотя бы один»
  считает код по `state.allowed_formats`.

Таблица приоритета терминальных сигналов, структуры `Signal`/`Observation` и разбор ответа модели
идентичны TG — переиспользуем импортом.
"""

from typing import Any, Optional

from qa_harness.domain.screening_split.policy.observation import (  # noqa: F401 — re-export
    FOCUS_ANSWERED,
    INERT_SIGNALS,
    MAX_SIGNALS,
    TERMINAL_PRIORITY,
    TERMINAL_SIGNAL_REASON,
    Observation,
    Signal,
    parse_observation as _tg_parse_observation,
)

# Присутственные форматы: их готовность закрывает `format_check`. `FIELD_WORK` проверяется отдельно
# (`field_work_check`), `REMOTE` вопроса не требует — при нём проверка приходит как `n/a`.
PRESENCE_FORMATS: tuple[str, ...] = ("ON_SITE", "HYBRID")
KNOWN_FORMATS: frozenset[str] = frozenset({"ON_SITE", "REMOTE", "HYBRID", "FIELD_WORK"})

NONTERMINAL_SIGNALS: frozenset[str] = frozenset({
    "gibberish", "bot_check", "answer_aid", "salary_info",
    "company_info", "scheduling", "pause",
}) | INERT_SIGNALS

SIGNAL_TO_COUNTER: dict[str, str] = {
    "gibberish": "gibberish",
    "bot_check": "bot_check",
    "salary_info": "salary_info",
    "pause": "pause",
}

ALL_SIGNALS: frozenset[str] = frozenset(TERMINAL_PRIORITY) | NONTERMINAL_SIGNALS


def formats_ready(obs: Observation) -> dict[str, str]:
    """`facts.formats_ready` → `{формат: 'yes'|'no'}`. Мусор молча отбрасывается.

    Дубликат формата в одной реплике — берём ПОСЛЕДНЮЮ запись: если кандидат в одном сообщении
    поправился, поправка стоит позже.
    """
    out: dict[str, str] = {}
    for item in (obs.facts or {}).get("formats_ready") or []:
        if not isinstance(item, dict):
            continue
        fmt = str(item.get("format") or "").strip().upper()
        ready = str(item.get("ready") or "").strip().lower()
        if fmt in KNOWN_FORMATS and ready in ("yes", "no"):
            out[fmt] = ready
    return out


def parse_observation(raw: Any, message: str) -> tuple[Observation, str]:
    """Разбор TG + hh-фильтр сигналов.

    Фильтр — страховка, а не рабочая ветка: `contact_source` в hh-схеме промпта отсутствует, вернуть
    его модели неоткуда. Но реестр причин hh такого кода не знает, и дойди он до правил — ход ушёл бы
    в скрипт, которого нет.
    """
    obs, problems = _tg_parse_observation(raw, message)
    kept = [s for s in obs.signals if s.code in ALL_SIGNALS]
    for signal in obs.signals:
        if signal.code not in ALL_SIGNALS:
            obs.dropped.append(f"сигнал {signal.code!r} в hh-канале не существует")
    obs.signals = kept
    return obs, problems


def normalize_facts(facts: Optional[dict]) -> dict:
    """Факты с гарантированными ключами — чтобы правила не разбирали отсутствие поля."""
    src = facts if isinstance(facts, dict) else {}
    return {
        "candidate_city": src.get("candidate_city"),
        "formats_ready": src.get("formats_ready") or [],
        "relocation_ready": src.get("relocation_ready"),
        "geo_blocked": bool(src.get("geo_blocked")),
    }

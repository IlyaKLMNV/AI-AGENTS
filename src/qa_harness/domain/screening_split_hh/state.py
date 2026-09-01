"""Модель состояния диалога split-скрининга — HH-канал.

Аналог `screening_split/state.py` (TG), дельта по EGGPLANT_SPLIT_TASK.md §3 / SPLIT_TG_VS_HH.md §2.3:
- `COUNTER_KEYS` — 5 (убран `contact_source`);
- `init_state(allowed_formats, questions_text)` вместо `init_state(work_format, ...)` — инициализация
  `format_check`/`field_work_check` по таблице §3 (мультиформат: «подходит хотя бы один»);
- новые ключи state: `allowed_formats`, `field_work_check`, `field_work_reasks`;
- `apply_updates` — ветка `field_work_check` (симметрично `format_check`);
- `progress_signature`/`is_complete` — учитывают `field_work_check`.

Идентичный TG разбор `[questions]` переиспользуем импортом (без дрейфа).
"""

import copy

# _split_questions идентичен TG — импортируем, чтобы разбор [questions] не разъезжался между каналами.
from qa_harness.domain.screening_split.state import _split_questions

COUNTER_KEYS = ("bot_check", "gibberish", "salary_info", "demand", "pause")  # без contact_source
_DONE_QUESTION_STATUSES = {"closed", "refused"}

KNOWN_WORK_FORMATS = ("ON_SITE", "REMOTE", "HYBRID", "FIELD_WORK")


def normalize_work_formats(raw) -> list[str]:
    """[{id,name}] | 'ON_SITE, REMOTE' | None -> ['ON_SITE','REMOTE'].

    Регистронезависимо (фикстуры содержат нижний регистр), дедуп, порядок сохраняем,
    неизвестные id отбрасываем (EGGPLANT_SPLIT_TASK.md §5, подводный камень 8).
    """
    if not raw:
        return []
    if isinstance(raw, str):
        tokens = [t.strip() for t in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        tokens = []
        for item in raw:
            if isinstance(item, dict):
                tokens.append(str(item.get("id") or item.get("name") or "").strip())
            else:
                tokens.append(str(item or "").strip())
    else:
        return []

    out: list[str] = []
    for tok in tokens:
        up = tok.upper()
        if up in KNOWN_WORK_FORMATS and up not in out:
            out.append(up)
    return out


def _init_format_checks(allowed_formats: list[str]) -> tuple[str, str]:
    """(format_check, field_work_check) по таблице §3.

    - REMOTE среди допустимых → format_check = n/a (удалёнка подходит по умолчанию);
    - есть ON_SITE/HYBRID и нет REMOTE → format_check = pending;
    - иначе (только FIELD_WORK / пусто / нераспознано) → format_check = n/a;
    - field_work_check = pending, если среди допустимых есть FIELD_WORK, иначе n/a.
    """
    af = set(allowed_formats or [])
    if ("ON_SITE" in af or "HYBRID" in af) and "REMOTE" not in af:
        format_check = "pending"
    else:
        format_check = "n/a"
    field_work_check = "pending" if "FIELD_WORK" in af else "n/a"
    return format_check, field_work_check


def init_state(allowed_formats, questions_text: str) -> dict:
    """Создаёт начальное состояние для нового hh-диалога.

    allowed_formats — уже нормализованный список id (или сырой вход: нормализуем на входе)."""
    af = normalize_work_formats(allowed_formats) if not (
        isinstance(allowed_formats, list) and all(x in KNOWN_WORK_FORMATS for x in allowed_formats)
    ) else list(allowed_formats)
    format_check, field_work_check = _init_format_checks(af)

    questions = [
        {"key": f"q{i + 1}", "text": text, "status": "pending", "reask_count": 0}
        for i, text in enumerate(_split_questions(questions_text))
    ]

    return {
        "salary": "pending",
        "format_check": format_check,       # pending | closed | n/a
        "field_work_check": field_work_check,  # pending | closed | n/a
        # Город спрашиваем ВСЕГДА, включая вакансии с REMOTE: без него гео-ограничение вакансии не
        # отсеивает никого, кто сам не сказал, что за границей (Р18).
        "city_check": "pending",
        # Переезд — только когда присутственный формат подтверждён, город известен и не совпадает с
        # локацией. Разъездной формат считается присутственным (Р18).
        "relocation_check": "n/a",
        "candidate_city": None,
        "allowed_formats": af,              # нормализован кодом — единственный источник правды о форматах
        "questions": questions,
        "counters": {k: 0 for k in COUNTER_KEYS},
        "last_asked": None,
        # Канальное: приветствие приклеивается к ПЕРВОМУ сообщению кандидату и больше никогда.
        # Флаг ведёт код — у промпта памяти о прошлых ходах нет. В tg приветствия нет вовсе.
        "greeted": False,
        "last_asking": None,    # 'salary'|'format'|'field_work'|'qN'|None (код-лимит переспросов)
        "salary_reasks": 0,
        "format_reasks": 0,
        "field_work_reasks": 0,  # новая ветка reask-cap (в tg её нет)
        "city_reasks": 0,
        "relocation_reasks": 0,
        "no_progress": 0,
        # Ниже — поля ядра `policy` (старый движок их не читает и не пишет).
        # `formats` копит ответы ПО ФОРМАТАМ через диалог: Observation отдаёт только сказанное на
        # этом ходе, а «отказался от всех допустимых» считается по накопленному.
        "formats": {},          # {'ON_SITE': 'yes'|'no', ...}
        "format_asked": None,   # формат последнего заданного вопроса — по нему модель относит «да»/«нет»
        "relocation_ready": None,   # 'yes'|'no'|None — готовность переехать / работать из локации вакансии
        "questions_intro_sent": False,
        "last_sent": None,
    }


def apply_updates(state: dict, updates: list[dict] | None, event: str | None = None) -> dict:
    """Возвращает НОВОЕ состояние с применённой дельтой (вход не мутируется).

    Мёрж монотонный: salary/format_check/field_work_check `closed` и вопросы `closed`/`refused`
    залипают. `reasked` инкрементит `reask_count`. `event ∈ COUNTER_KEYS` инкрементит счётчик.
    """
    new = copy.deepcopy(state)
    questions_by_key = {q["key"]: q for q in new.get("questions", [])}

    for upd in updates or []:
        key = (upd.get("key") or "").strip()
        value = (upd.get("value") or "").strip()
        if not key:
            continue

        if key == "salary":
            if value == "closed" and new.get("salary") != "closed":
                new["salary"] = "closed"
        elif key == "format_check":
            if value == "closed" and new.get("format_check") == "pending":
                new["format_check"] = "closed"
        elif key == "field_work_check":
            if value == "closed" and new.get("field_work_check") == "pending":
                new["field_work_check"] = "closed"
        elif key == "candidate_city":
            if value:
                new["candidate_city"] = value
        elif key in questions_by_key:
            q = questions_by_key[key]
            if q["status"] in _DONE_QUESTION_STATUSES:
                continue  # уже закрыт/отказ — монотонно, не трогаем
            if value in _DONE_QUESTION_STATUSES:
                q["status"] = value
            elif value == "reasked":
                q["reask_count"] = q.get("reask_count", 0) + 1
        # неизвестные ключи игнорируем (страховка)

    if event and event in new.get("counters", {}):
        new["counters"][event] += 1

    return new


def progress_signature(state: dict) -> tuple:
    """Снимок «что уже собрано» — по нему код видит, продвинулся ли диалог за ход.

    Меняется только при новом факте: закрылась зарплата/формат/разъездной, узнали город,
    доп-вопрос стал closed/refused. `reask_count` не входит (переспрос — не прогресс).

    Ответ про КОНКРЕТНЫЙ формат тоже прогресс: «в офис не готов» при допустимых `[ON_SITE, HYBRID]`
    не закрывает проверку, но снимает один вариант — следующим ходом код спросит про гибрид.
    У старого движка ключ `formats` не заполняется, и элемент остаётся пустым."""
    return (
        state.get("salary"),
        state.get("format_check"),
        state.get("field_work_check"),
        state.get("city_check"),
        state.get("relocation_check"),
        bool(state.get("candidate_city")),
        tuple(sorted((state.get("formats") or {}).items())),
        tuple(q.get("status") for q in state.get("questions", [])),
    )


def pending_questions(state: dict) -> list[dict]:
    return [q for q in state.get("questions", []) if q.get("status") == "pending"]


def is_salary_done(state: dict) -> bool:
    return state.get("salary") == "closed"


def is_format_done(state: dict) -> bool:
    return state.get("format_check") in ("n/a", "closed")


def is_field_work_done(state: dict) -> bool:
    return state.get("field_work_check") in ("n/a", "closed")


def is_complete(state: dict) -> bool:
    """Все приоритеты (зарплата/формат/разъездной) закрыты и по каждому [questions] есть closed/refused."""
    if not is_salary_done(state) or not is_format_done(state) or not is_field_work_done(state):
        return False
    # Локация — такой же пункт повестки, как формат (Р18): без города повестка не собрана.
    if state.get("city_check") not in ("n/a", "closed"):
        return False
    if state.get("relocation_check") not in ("n/a", "closed"):
        return False
    return all(q.get("status") in _DONE_QUESTION_STATUSES for q in state.get("questions", []))

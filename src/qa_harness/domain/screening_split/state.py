"""Модель состояния диалога split-скрининга и его изменение.

Порт прод-кода tgApi (`app/common/assistants/screening_state.py`, HEAD e733095) 1:1 —
чистая логика без БД: `init_state` создаёт стартовое состояние; `apply_updates`
применяет изменения от Аналитика так, что уже закрытые пункты (и отмеченные refused)
назад не открываются, и увеличивает счётчики событий по полю `event`.

Тестируем ровно ту арифметику состояний, что крутится в проде: пороги/лимиты в
split считает КОД (см. engine), а не LLM, поэтому эта модель — часть контракта.

Форма state:
    {
      "salary": "pending|closed",
      "format_check": "n/a|pending|closed",
      "candidate_city": str|None,
      "questions": [ {"key","text","status":"pending|closed|refused","reask_count":int} ],
      "counters": {"bot_check":0,"gibberish":0,"salary_info":0,"demand":0,"contact_source":0,"pause":0},
      "last_asked": str|None,
      "last_asking": str|None,   # 'salary'|'format'|'qN'|None
      "salary_reasks": int, "format_reasks": int,
      "questions_intro_sent": bool,   # вводную перед доп-вопросами уже сказали (Б1)
    }
"""

import copy
import re

COUNTER_KEYS = ("bot_check", "gibberish", "salary_info", "demand", "contact_source", "pause")
_QUESTION_STATUSES = {"closed", "refused", "reasked"}
_DONE_QUESTION_STATUSES = {"closed", "refused"}


def _split_questions(questions_text: str) -> list[str]:
    """Разбивает свободный текст [questions] на отдельные вопросы.

    По строкам: пустые отбрасываем, ведущую нумерацию/маркеры («1.», «-», «*»)
    убираем. Если переносов строк нет — это один вопрос.
    """
    if not questions_text:
        return []
    items: list[str] = []
    for raw in questions_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def init_state(work_format: str, questions_text: str) -> dict:
    """Создаёт начальное состояние для нового диалога."""
    wf = (work_format or "").strip().lower()
    format_check = "n/a" if wf == "remote" else "pending"

    questions = [
        {"key": f"q{i + 1}", "text": text, "status": "pending", "reask_count": 0}
        for i, text in enumerate(_split_questions(questions_text))
    ]

    return {
        "salary": "pending",
        "format_check": format_check,
        "candidate_city": None,
        "questions": questions,
        "counters": {k: 0 for k in COUNTER_KEYS},
        "last_asked": None,
        "last_asking": None,    # что спрашивали в прошлый ход: 'salary'|'format'|'qN'|None (код-лимит переспросов)
        "salary_reasks": 0,     # сколько раз переспросили зарплату (код форсит STOP после 2)
        "format_reasks": 0,     # сколько раз переспросили формат/локацию
        "no_progress": 0,       # ходов подряд без нового собранного факта (код форсит завершение, см. engine)
        # Вводная фраза перед ПЕРВЫМ доп-вопросом говорится ровно один раз за диалог (Б1). Признак
        # ведёт КОД: у промпта памяти о прошлых ходах нет, а `reask_count` для этого не годится —
        # ход со сменой фокуса переспросом не считается, счётчик остаётся 0, и модель решила бы,
        # что переходит к вопросам впервые. Ставится в policy/core при сборке инструкции.
        "questions_intro_sent": False,
    }


def apply_updates(state: dict, updates: list[dict] | None, event: str | None = None) -> dict:
    """Возвращает НОВОЕ состояние с применённой дельтой (вход не мутируется).

    Мёрж монотонный: salary/format_check `closed` и вопросы `closed`/`refused`
    залипают — повторно открыть нельзя. `reasked` инкрементит `reask_count`.
    `event` (если из COUNTER_KEYS) инкрементит соответствующий счётчик.
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

    Меняется ТОЛЬКО когда появился новый факт: закрылась зарплата или формат, узнали город,
    доп-вопрос стал closed/refused. `reask_count` сюда НЕ входит — переспрос это не прогресс.
    Сравнение безопасно: apply_updates монотонен, сигнатура не может отыграть назад. Признак не
    зависит от темы/распознавания Аналитиком — ловит любое зацикливание. Порт tgApi 1:1."""
    return (
        state.get("salary"),
        state.get("format_check"),
        bool(state.get("candidate_city")),
        tuple(q.get("status") for q in state.get("questions", [])),
    )


def pending_questions(state: dict) -> list[dict]:
    return [q for q in state.get("questions", []) if q.get("status") == "pending"]


def is_salary_done(state: dict) -> bool:
    return state.get("salary") == "closed"


def is_format_done(state: dict) -> bool:
    return state.get("format_check") in ("n/a", "closed")


def is_complete(state: dict) -> bool:
    """Все приоритеты закрыты и по каждому [questions] есть closed/refused."""
    if not is_salary_done(state) or not is_format_done(state):
        return False
    return all(q.get("status") in _DONE_QUESTION_STATUSES for q in state.get("questions", []))

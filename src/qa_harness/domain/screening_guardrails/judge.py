"""LLM-судья гардрейлов: одно сообщение ассистента → self_answer / repeated_questions / premature_end.

Перенос evaluate_reply: судья (ModelClient) размечает 3 нарушения + repeated_topics. Жёсткий гейт —
нет вопросов в реплике → premature_end=false. На любой сбой/непарс судьи graceful-фолбэк на эвристики
(detectors.py), как в легаси; в отчёте это видно по used_heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from qa_harness.core import safe_json_loads

from .detectors import (
    has_questions_in_reply,
    heuristic_premature_end,
    heuristic_repeated_questions,
    heuristic_self_answer,
)

ALLOWED_TOPICS = {"location", "salary", "relocation", "experience", "other"}

EVAL_INSTRUCTION = (
    "Ты проверяешь одно сообщение рекрутера на 3 нарушения. Верни строго JSON.\n\n"
    "A) self_answer = true, если рекрутер пишет за кандидата (вставляет ответы кандидата) или ведёт диалог "
    "за обе стороны. Это НЕ self_answer, если рекрутер просто отвечает на вопрос кандидата.\n"
    "B) repeated_questions = true, если рекрутер в ОДНОМ сообщении повторяет по смыслу один и тот же вопрос "
    "(дважды про город/зарплату/переезд и т.п.). Не путать с 2-3 разными вопросами.\n"
    "C) premature_end_after_questions = true, если рекрутер В ТОМ ЖЕ сообщении задаёт вопрос И одновременно "
    "закрывает диалог (END / 'это вся информация, я передам' / явно завершает). Фразы про 'внутренний рекрутер "
    "свяжется' сами по себе НЕ конец. Если в сообщении нет вопросов — premature_end всегда false.\n\n"
    'Верни JSON: {"self_answer": true|false, "repeated_questions": true|false, '
    '"premature_end_after_questions": true|false, '
    '"repeated_topics": ["location"|"salary"|"relocation"|"experience"|"other"], "comment": "кратко почему"}'
)


@dataclass
class GuardrailVerdict:
    self_answer: bool
    repeated_questions: bool
    premature_end: bool
    repeated_topics: List[str] = field(default_factory=list)
    comment: str = ""
    used_heuristics: bool = False


def _sanitize_topics(topics: Any) -> List[str]:
    if not isinstance(topics, list):
        return []
    uniq, seen = [], set()
    for x in topics[:10]:
        s = str(x).strip()
        if not s:
            continue
        s = s if s in ALLOWED_TOPICS else "other"
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _heuristic_verdict(assistant_reply: str, conversation_end_flag: bool, has_q: bool, note: str) -> GuardrailVerdict:
    sa, _sar = heuristic_self_answer(assistant_reply)
    rq, _rqr, topics = heuristic_repeated_questions(assistant_reply)
    pe, _per = heuristic_premature_end(assistant_reply, conversation_end_flag)
    if not has_q:
        pe = False
    return GuardrailVerdict(sa, rq, pe, _sanitize_topics(topics or (["other"] if rq else [])),
                            comment=note, used_heuristics=True)


class GuardrailJudge:
    def __init__(self, model_client: Any) -> None:
        self._client = model_client

    def evaluate(self, candidate_message: str, assistant_reply: str, conversation_end_flag: bool) -> Tuple[GuardrailVerdict, Any]:
        has_q = has_questions_in_reply(assistant_reply)
        payload = json.dumps(
            {"instruction": EVAL_INSTRUCTION, "candidate_message": candidate_message,
             "assistant_reply": assistant_reply, "conversation_end_flag": conversation_end_flag},
            ensure_ascii=False,
        )
        try:
            raw, usage = self._client.create(payload)
            data, _err = safe_json_loads(raw, lenient=True)
            if not isinstance(data, dict):
                raise ValueError("judge did not return a JSON object")
            premature = bool(data.get("premature_end_after_questions"))
            comment = str(data.get("comment") or "").strip()
            if not has_q and premature:
                premature = False
                comment = (comment + " | premature_end overridden: no questions").strip(" |")
            verdict = GuardrailVerdict(
                self_answer=bool(data.get("self_answer")),
                repeated_questions=bool(data.get("repeated_questions")),
                premature_end=premature,
                repeated_topics=_sanitize_topics(data.get("repeated_topics")),
                comment=comment,
                used_heuristics=False,
            )
            return verdict, usage
        except Exception:  # noqa: BLE001 — graceful degrade на эвристики (как в легаси)
            return _heuristic_verdict(assistant_reply, conversation_end_flag, has_q, "judge failed, heuristics used"), None

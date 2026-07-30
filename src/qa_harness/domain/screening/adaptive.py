"""Адаптивный мультитёрн: живой screening_assistant ↔ адаптивный LLM-кандидат.

Кандидат (`CandidateAgent`) генерит реплику, ассистент (`ScreeningConversation`, промпт-под-тестом)
отвечает, кандидат реагирует на ответ — и так до завершения диалога ассистентом (END / фильтр) или до
`max_turns`. Это замена батч-списка реплик: диалог собирается вживую, поэтому разнится от прогона к прогону.
quality ≠ infra: провал генерации кандидата (исчерпан retry+fallback) → диалог обрывается с пометкой
`gen_failed` (раннер трактует как infra-ошибку), не как «промпт не прошёл».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from qa_harness.domain.generators.candidate_agent import CandidateAgent

from .conversation import ScreeningConversation


@dataclass
class AdaptiveTurn:
    candidate: str
    candidate_source: str          # "llm" | "fallback" | "failed"
    reply: str
    end: bool
    gen_usage: Any = None          # usage генерации реплики кандидата
    assistant_usage: Any = None    # usage ответа ассистента
    gen_attempts: int = 1
    tool_trace: Any = None         # снимок хода, если движок его даёт (split: Decision+state); монолит — None


@dataclass
class AdaptiveResult:
    turns: List[AdaptiveTurn] = field(default_factory=list)
    error: Optional[str] = None    # None == ок; иначе причина обрыва (gen/assistant) — infra


def run_adaptive_conversation(conv: ScreeningConversation, candidate: CandidateAgent,
                              max_turns: int = 6) -> AdaptiveResult:
    """Прогнать живой диалог. conv ещё НЕ должен быть start()-нут — стартуем здесь."""
    res = AdaptiveResult()
    try:
        conv.start()
    except Exception as e:  # noqa: BLE001
        res.error = f"conversation:{type(e).__name__}:{e}"
        return res

    history: List[Tuple[str, str]] = []
    last_reply: Optional[str] = None

    for i in range(max(1, max_turns)):
        gr = candidate.next_turn(history, last_reply, turn_index=i)
        if not gr.ok:
            res.error = f"candidate_gen_failed:{gr.errors[-1] if gr.errors else 'unknown'}"
            return res
        candidate_msg = str(gr.item)
        try:
            tr = conv.respond(candidate_msg)
        except Exception as e:  # noqa: BLE001
            res.error = f"assistant:{type(e).__name__}:{e}"
            return res

        reply = str(tr.response or "")
        res.turns.append(AdaptiveTurn(
            candidate=candidate_msg, candidate_source=gr.source, reply=reply,
            end=tr.conversation_end, gen_usage=gr.usage, assistant_usage=tr.usage,
            gen_attempts=gr.attempts, tool_trace=getattr(tr, "tool_trace", None),
        ))
        history.append(("candidate", candidate_msg))
        history.append(("assistant", reply))
        last_reply = reply
        if tr.conversation_end:
            break

    return res

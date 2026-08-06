"""Драйвер разговора поверх hh-движка split (аналог `screening_split/conversation.py`).

Тот же мини-интерфейс start()/respond()->TurnResult, но крутит hh-`ScreeningSplitEngine` с
hh-Аналитиком. Интервьюер и стор канало-независимы — переиспользуем TG-классы. contact_source в hh
нет, поэтому accounts/source_type здесь не участвуют. `_trace` дополнительно отдаёт `field_work_check`.
"""

from typing import Any, Optional

from qa_harness.domain.screening_split.conversation import TurnResult  # канало-независим
from qa_harness.domain.screening_split.interviewer import ScreeningInterviewer
from qa_harness.domain.screening_split.store import InMemoryStateStore

from .analyzer import ScreeningAnalyzer
from .engine import ScreeningSplitEngine


class SplitConversation:
    """start()/respond() поверх hh-ScreeningSplitEngine с local-промптами (screening_analyzer_hh)."""

    def __init__(
        self,
        *,
        client: Any,
        analyzer_client: Any,
        interviewer_spec: Any,
        vacancy_info: dict,
        recruiter_name: str,
        candidate_name: str,
    ) -> None:
        store = InMemoryStateStore()
        analyzer = ScreeningAnalyzer(analyzer_client)
        interviewer = ScreeningInterviewer(interviewer_spec, client)
        self._engine = ScreeningSplitEngine(store, analyzer, interviewer, client)
        self._vacancy_info = vacancy_info
        self._recruiter = recruiter_name
        self._candidate = candidate_name
        self._cid: Optional[str] = None

    def start(self) -> str:
        self._cid = self._engine.create_thread(self._vacancy_info, self._recruiter, self._candidate)
        return self._cid

    def respond(self, candidate_message: str) -> TurnResult:
        result = self._engine.add_message_and_run(self._cid, candidate_message)
        return TurnResult(
            response=result.response,
            conversation_end=bool(result.conversation_end),
            usage=dict(self._engine.last_usage),
            tool_trace=self._trace(),
        )

    def _trace(self) -> dict:
        """Снимок хода для отчёта: решение Аналитика + компактное hh-состояние (+ field_work_check)."""
        decision = self._engine.last_decision
        st = self._engine.last_state
        state_snap = None
        if st:
            state_snap = {
                "salary": st.get("salary"),
                "format_check": st.get("format_check"),
                "field_work_check": st.get("field_work_check"),
                "city": st.get("candidate_city"),
                "allowed_formats": list(st.get("allowed_formats", [])),
                "questions": {q["key"]: q["status"] for q in st.get("questions", [])},
                "counters": dict(st.get("counters", {})),
            }
        return {"decision": decision, "state": state_snap}

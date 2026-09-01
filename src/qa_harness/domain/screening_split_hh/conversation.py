"""Драйвер разговора поверх ядра `policy` (HH-канал).

Тот же мини-интерфейс start()/respond()->TurnResult, что в TG, но крутит hh-ядро. Стор и Интервьюер
канало-независимы — берём TG-классы. `contact_source` в hh нет, поэтому accounts/source_type здесь
не участвуют.
"""

from typing import Any, Optional

from qa_harness.domain.screening_split.conversation import TurnResult  # канало-независим
from qa_harness.domain.screening_split.interviewer import PolicyInterviewer
from qa_harness.domain.screening_split.policy.observation import snapshot as observation_snapshot
from qa_harness.domain.screening_split.store import InMemoryStateStore

from .policy.engine import PolicyEngine
from .policy.observer import ScreeningObserver


class SplitConversation:
    """start()/respond() поверх hh-ядра с local-промптами (`screening_analyzer_hh`)."""

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
        self._engine = PolicyEngine(InMemoryStateStore(), ScreeningObserver(analyzer_client),
                                    PolicyInterviewer(interviewer_spec, client), client)
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
        """Снимок хода: решение, hh-состояние (+ `field_work_check`), зарплатный разбор, трасса ядра."""
        st = self._engine.last_state
        state_snap = None
        if st:
            state_snap = {
                "salary": st.get("salary"),
                "format_check": st.get("format_check"),
                "field_work_check": st.get("field_work_check"),
                "city": st.get("candidate_city"),
                "relocation_ready": st.get("relocation_ready"),
                "city_check": st.get("city_check"),
                "relocation_check": st.get("relocation_check"),
                "greeted": st.get("greeted"),
                "allowed_formats": list(st.get("allowed_formats", [])),
                "formats": dict(st.get("formats", {}) or {}),
                "questions": {q["key"]: q["status"] for q in st.get("questions", [])},
                "counters": dict(st.get("counters", {})),
                # Переспросы приоритетных пунктов: по ним видно, не сжёг ли мультиформат бюджет на
                # честных ответах кандидата («в офис не готов» → вопрос про гибрид — не переспрос).
                # Какой формат спрашивали последним: по нему Наблюдатель относит короткое «да»/«нет»,
                # и по нему же проверка `formats_asked` видит, отыграна ли лестница мультиформата.
                "format_asked": st.get("format_asked"),
                "reasks": {"salary": st.get("salary_reasks", 0),
                           "format": st.get("format_reasks", 0),
                           "field_work": st.get("field_work_reasks", 0),
                           "city": st.get("city_reasks", 0),
                           "relocation": st.get("relocation_reasks", 0)},
            }
        trace: dict[str, Any] = {"decision": self._engine.last_decision, "state": state_snap,
                                 "salary": getattr(self._engine, "last_salary", None),
                                 "guard_trips": list(getattr(self._engine, "last_guard_trips", []) or [])}
        plan = getattr(self._engine, "last_plan", None)
        if plan is not None:
            trace["audit"] = plan.audit
            trace["rule"] = plan.rule
        obs = getattr(self._engine, "last_observation", None)
        if obs is not None:
            trace["observation"] = observation_snapshot(obs)
        return trace

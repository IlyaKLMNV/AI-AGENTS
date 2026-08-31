"""Драйвер разговора поверх hh-движка split (аналог `screening_split/conversation.py`).

Тот же мини-интерфейс start()/respond()->TurnResult, но крутит hh-движок. Движков два, как и в TG:
`split` (действующий, Аналитик возвращает `Decision`) и `policy` (Наблюдатель + чистое ядро + гарды,
пакет `.policy`). Стор канало-независим, Интервьюер тоже — берём TG-классы: stateful для старого
движка, stateless `PolicyInterviewer` для нового. contact_source в hh нет, поэтому accounts/source_type
здесь не участвуют.
"""

from typing import Any, Optional

from qa_harness.domain.screening_split.conversation import TurnResult  # канало-независим
from qa_harness.domain.screening_split.interviewer import PolicyInterviewer, ScreeningInterviewer
from qa_harness.domain.screening_split.policy.observation import snapshot as observation_snapshot
from qa_harness.domain.screening_split.store import InMemoryStateStore

from .analyzer import ScreeningAnalyzer
from .engine import ScreeningSplitEngine

# Какой движок поднимать по умолчанию: "split" (действующий) | "policy" (новая архитектура).
# Ставится раннером один раз из флага --engine; параметр конструктора её перекрывает.
DEFAULT_ENGINE = "split"


class SplitConversation:
    """start()/respond() поверх hh-движка с local-промптами (`screening_analyzer_hh`)."""

    def __init__(
        self,
        *,
        client: Any,
        analyzer_client: Any,
        interviewer_spec: Any,
        vacancy_info: dict,
        recruiter_name: str,
        candidate_name: str,
        engine: Optional[str] = None,
    ) -> None:
        engine = engine or DEFAULT_ENGINE
        store = InMemoryStateStore()
        if engine == "policy":
            from .policy.engine import PolicyEngine
            from .policy.observer import ScreeningObserver as PolicyObserver
            self._engine = PolicyEngine(store, PolicyObserver(analyzer_client),
                                        PolicyInterviewer(interviewer_spec, client), client)
        else:
            self._engine = ScreeningSplitEngine(store, ScreeningAnalyzer(analyzer_client),
                                                ScreeningInterviewer(interviewer_spec, client), client)
        self._engine_kind = engine
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
        """Снимок хода для отчёта: решение Аналитика + компактное hh-состояние (+ field_work_check).

        У `policy` добавляются трасса ядра (какое правило выиграло, что видел Наблюдатель, что срезали
        гарды) и зарплатный разбор: иначе «гард вырезал» неотличимо от «Интервьюер так и написал».
        """
        decision = self._engine.last_decision
        st = self._engine.last_state
        state_snap = None
        if st:
            state_snap = {
                "salary": st.get("salary"),
                "format_check": st.get("format_check"),
                "field_work_check": st.get("field_work_check"),
                "city": st.get("candidate_city"),
                "relocation_ready": st.get("relocation_ready"),
                "allowed_formats": list(st.get("allowed_formats", [])),
                "formats": dict(st.get("formats", {}) or {}),
                "questions": {q["key"]: q["status"] for q in st.get("questions", [])},
                "counters": dict(st.get("counters", {})),
                # Переспросы приоритетных пунктов: по ним видно, не сжёг ли мультиформат бюджет на
                # честных ответах кандидата («в офис не готов» → вопрос про гибрид — не переспрос).
                "reasks": {"salary": st.get("salary_reasks", 0),
                           "format": st.get("format_reasks", 0),
                           "field_work": st.get("field_work_reasks", 0)},
            }
        trace: dict[str, Any] = {"decision": decision, "state": state_snap,
                                 "salary": getattr(self._engine, "last_salary", None)}
        if self._engine_kind == "policy":
            trace["guard_trips"] = list(getattr(self._engine, "last_guard_trips", []) or [])
            plan = getattr(self._engine, "last_plan", None)
            if plan is not None:
                trace["audit"] = plan.audit
                trace["rule"] = plan.rule
            obs = getattr(self._engine, "last_observation", None)
            if obs is not None:
                trace["observation"] = observation_snapshot(obs)
        return trace

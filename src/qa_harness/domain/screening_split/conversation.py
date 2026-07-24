"""Драйвер разговора поверх ported-движка split (аналог tgApi QA driver.py).

Даёт тот же мини-интерфейс, что монолитный `domain/screening/ScreeningConversation`
(`start()` / `respond()->TurnResult`), но внутри крутит `ScreeningSplitEngine`: Аналитик →
код-оркестрация → скрипт/Интервьюер. Дополнительно к контракту — снимок хода (`tool_trace`
= Decision Аналитика + состояние) и usage (analyzer + interviewer за ход).

Каждый разговор держит СВОЙ движок и in-memory стор (per-conversation) — это делает
прогон потокобезопасным (несколько сценариев в разных потоках не гоняют общий mutable
last_decision/last_state). Дорогие спеки/клиент промптов инжектятся снаружи (строятся один
раз в раннере): analyzer_client (LocalPromptClient) и interviewer_spec — read-only, шарятся.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .analyzer import ScreeningAnalyzer
from .engine import ScreeningSplitEngine
from .interviewer import ScreeningInterviewer
from .store import InMemoryStateStore


@dataclass
class TurnResult:
    response: Optional[str]
    conversation_end: bool
    usage: Any = None
    tool_trace: Any = None


class SplitConversation:
    """start()/respond() поверх ScreeningSplitEngine с local-промптами."""

    def __init__(
        self,
        *,
        client: Any,
        analyzer_client: Any,
        interviewer_spec: Any,
        vacancy_info: dict,
        recruiter_name: str,
        candidate_name: str,
        accounts: Optional[list] = None,
    ) -> None:
        store = InMemoryStateStore()
        analyzer = ScreeningAnalyzer(analyzer_client)
        interviewer = ScreeningInterviewer(interviewer_spec, client)
        self._engine = ScreeningSplitEngine(store, analyzer, interviewer, client)
        self._vacancy_info = vacancy_info
        self._recruiter = recruiter_name
        self._candidate = candidate_name
        # contact_source в проде выводится из аккаунтов кандидата (type -> hh/ln/github/mk),
        # а НЕ из вакансии. Для сценариев про источник контакта фикстура задаёт source_type.
        src_type = vacancy_info.get("source_type")
        self._accounts = accounts if accounts is not None else ([{"type": src_type}] if src_type else [])
        self._cid: Optional[str] = None

    def start(self) -> str:
        self._cid = self._engine.create_thread(
            self._vacancy_info, self._recruiter, self._candidate, self._accounts
        )
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
        """Снимок хода для отчёта: решение Аналитика + компактное состояние."""
        decision = self._engine.last_decision
        st = self._engine.last_state
        state_snap = None
        if st:
            state_snap = {
                "salary": st.get("salary"),
                "format_check": st.get("format_check"),
                "city": st.get("candidate_city"),
                "questions": {q["key"]: q["status"] for q in st.get("questions", [])},
                "counters": dict(st.get("counters", {})),
            }
        return {"decision": decision, "state": state_snap}

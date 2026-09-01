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
from .interviewer import PolicyInterviewer, ScreeningInterviewer
from .policy.observation import snapshot as observation_snapshot
from .store import InMemoryStateStore


# Какой движок поднимать по умолчанию: "split" (действующий) | "policy" (новая архитектура).
# Ставится раннером один раз из флага --engine; параметр конструктора её перекрывает.
DEFAULT_ENGINE = "split"


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
        engine: Optional[str] = None,
    ) -> None:
        engine = engine or DEFAULT_ENGINE
        store = InMemoryStateStore()
        if engine == "policy":
            # Новая архитектура: Наблюдатель + чистое ядро + гарды (docs/screening_split/
            # rearchitecture.html). Трасса остаётся совместимой — `last_decision` собирается из
            # TurnPlan, поэтому инварианты слоя A работают без правок фикстур.
            from .policy.engine import PolicyEngine
            from .policy.observer import ScreeningObserver
            self._engine = PolicyEngine(store, ScreeningObserver(analyzer_client),
                                        PolicyInterviewer(interviewer_spec, client), client)
        else:
            analyzer = ScreeningAnalyzer(analyzer_client)
            self._engine = ScreeningSplitEngine(store, analyzer,
                                                ScreeningInterviewer(interviewer_spec, client), client)
        self._engine_kind = engine
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
        """Снимок хода для отчёта: решение Аналитика + компактное состояние + зарплатный разбор.

        `salary` в трассе — это то, что код СДЕЛАЛ с суммой на этом ходе (годность claim, пересчёт,
        вердикт, effect). Без него инварианты слоя A по зарплате пришлось бы выводить косвенно, по
        script_key и state, и «сумму не распознали» было бы не отличить от «распознали, но не отсеяли».
        """
        decision = self._engine.last_decision
        st = self._engine.last_state
        state_snap = None
        if st:
            state_snap = {
                "salary": st.get("salary"),
                "format_check": st.get("format_check"),
                "city": st.get("candidate_city"),
                # Поле кода, а не модели: по нему видно, какой вход правила R6 сработал.
                "relocation_ready": st.get("relocation_ready"),
                "city_check": st.get("city_check"),
                "relocation_check": st.get("relocation_check"),
                "questions": {q["key"]: q["status"] for q in st.get("questions", [])},
                "counters": dict(st.get("counters", {})),
            }
        trace = {"decision": decision, "state": state_snap, "salary": self._engine.last_salary}
        if self._engine_kind == "policy":
            # Что натворили гарды и что модель увидела — иначе «гард вырезал» неотличимо от
            # «Интервьюер так и написал», а отброшенный сигнал — от неувиденного.
            trace["guard_trips"] = list(getattr(self._engine, "last_guard_trips", []) or [])
            plan = getattr(self._engine, "last_plan", None)
            if plan is not None:
                trace["audit"] = plan.audit
                trace["rule"] = plan.rule
            obs = getattr(self._engine, "last_observation", None)
            if obs is not None:
                trace["observation"] = observation_snapshot(obs)
        return trace

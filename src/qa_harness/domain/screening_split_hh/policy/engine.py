"""Тонкий оркестратор поверх чистого ядра — HH-канал. Всё решение в `core.decide`, здесь ввод-вывод.

Дельта к TG-оркестратору:

- **нет ленивой миграции.** В TG она чинит документы Mongo, заведённые старым движком (вырезает вилку
  из контекста, пишет типизированный `salary_band`). В hh переносить нечего: боевой split-движок в
  `eggplant-api` не запускался ни разу, а там, где он появится, состояние живёт в Postgres и правка
  схемы обязана быть явной Alembic-ревизией, а не ленивой (план, hh-контур, п. 3);
- **нет `contact_source`**: ни в контексте, ни в подстановках скриптов;
- **вилка приходит от канала** типизированной (`{min, max, currency}`) — то самое место, где сегодня
  hh сравнивает тенге как рубли (P11): здесь `currency` наконец читается.

Вызовов LLM за ход: **0** (диалог закрыт) · **1** (ход-скрипт) · **2** (ход с вопросом).
"""

from typing import Any, Optional

from qa_harness.core import accumulate_usage, blank_usage
from qa_harness.domain.screening_split import salary as salary_mod
from qa_harness.domain.screening_split.errors import AssistantError
from qa_harness.domain.screening_split.policy.engine import _compat_decision
from qa_harness.domain.screening_split.policy.guards import GuardSpec, _urls, apply_guards

from .. import context as ctx_mod
from .. import state as state_model
from . import reasons
from .context import allowed_formats_of, build_observer_context, has_geo_restriction
from .core import DecideContext, decide
from .observation import Observation


class PolicyResult:
    """Результат хода. По смыслу совместим с `..engine.ConversationResult`."""

    __slots__ = ("response", "conversation_end")

    def __init__(self, response: Optional[str], conversation_end: bool) -> None:
        self.response = response
        self.conversation_end = conversation_end


class PolicyEngine:
    def __init__(self, store: Any, observer: Any, interviewer: Any, client: Any, *,
                 defensive_guards: bool = True) -> None:
        self.store = store
        self.observer = observer
        self.interviewer = interviewer
        self._client = client
        self._defensive = defensive_guards
        self._recruiter_name = ""
        self._candidate_name = ""
        self._vacancy: dict = {}
        # Снимок хода для QA-трассы. В проде не нужен; поведение от него не зависит.
        self.last_plan: Any = None
        self.last_decision: Optional[dict] = None
        self.last_state: Optional[dict] = None
        self.last_usage: dict = blank_usage()
        self.last_salary: Optional[dict] = None
        self.last_guard_trips: list = []
        self.last_observation: Optional[Observation] = None

    # ── старт ────────────────────────────────────────────────────────────────

    def create_thread(self, vacancy_info: dict, recruiter_name: str, candidate_name: str) -> str:
        # Контекст БЕЗ вилки: секрет не кладут туда, где его потом запрещают (П4).
        context = build_observer_context(recruiter_name, candidate_name, vacancy_info)
        seed = ctx_mod.build_interviewer_seed(recruiter_name, candidate_name)
        self._recruiter_name, self._candidate_name = recruiter_name, candidate_name

        # Conversation заводится ради ИДЕНТИФИКАТОРА: её id — ключ диалога. Историю в неё больше
        # никто не пишет и не читает — Интервьюер stateless.
        conversation = self._client.conversations.create(
            items=[{"type": "message", "role": "assistant", "content": seed}]
        )
        conversation_id = conversation.id

        state = state_model.init_state(allowed_formats_of(vacancy_info),
                                       vacancy_info.get("questions", "") or "")
        self.store.create(
            conversation_id, "policy",
            state=state, context=context,
            location=vacancy_info.get("location", "") or "",
            salary_band={"min": vacancy_info.get("min_salary"),
                         "max": vacancy_info.get("max_salary"),
                         "currency": vacancy_info.get("salary_currency", "RUB")},
        )
        self._vacancy = vacancy_info
        return conversation_id

    # ── ход ──────────────────────────────────────────────────────────────────

    def add_message_and_run(self, conversation_id: Any, message: str) -> PolicyResult:
        self.last_plan = None
        self.last_decision = None
        self.last_state = None
        self.last_usage = blank_usage()
        self.last_salary = None
        self.last_guard_trips = []
        self.last_observation = None

        doc = self.store.load(conversation_id)
        if not doc:
            return PolicyResult(reasons.render("REPLY_FALLBACK"), False)
        if doc.get("finished"):
            return PolicyResult(None, True)

        state = doc["state"]
        band = doc.get("salary_band") or {}
        ctx = DecideContext(
            band_min=band.get("min"), band_max=band.get("max"),
            band_currency=band.get("currency") or "RUB",
            location=doc.get("location", ""),
            has_geo_restriction=has_geo_restriction(self._vacancy),
        )

        observation, failed = self._observe(doc.get("context", ""), state, message)
        plan = decide(state, observation, message, ctx, analyzer_failed=failed)

        text = self._speak(plan, message, ctx, state.get("last_sent") or "")

        # Что реально ушло кандидату: у Интервьюера истории нет, а дословно повторённый переспрос
        # выглядел бы поломкой.
        if plan.kind == "ask" and text:
            plan.state_next["last_sent"] = text

        self.last_plan = plan
        self.last_observation = observation
        self.last_state = plan.state_next
        self.last_decision = _compat_decision(plan)
        self._log_salary(conversation_id, observation, plan)

        self.store.save_state(conversation_id, plan.state_next, finished=plan.end)
        return PolicyResult(text or None, plan.end)

    # ── внутреннее ───────────────────────────────────────────────────────────

    def _observe(self, context: str, state: dict, message: str) -> tuple[Observation, bool]:
        """Наблюдение либо признак сбоя. Сбой — это R2 (`REPLY_FALLBACK`), а не отказ кандидату."""
        try:
            observation = self.observer.run(context, state, message)
        except AssistantError:
            accumulate_usage(self.last_usage, self.observer.last_usage)
            return Observation(), True
        accumulate_usage(self.last_usage, self.observer.last_usage)
        return observation, False

    def _speak(self, plan: Any, message: str, ctx: DecideContext, last_sent: str) -> str:
        """Скрипт — из реестра; вопрос — через Интервьюера, затем шлюз гардов."""
        if plan.kind == "silent":
            return ""
        if plan.kind == "script":
            return reasons.render(plan.reason_code, city=ctx.location) or ""

        seed = ctx_mod.build_interviewer_seed(self._recruiter_name, self._candidate_name)
        try:
            raw, usage = self.interviewer.run(plan.instruction, message,
                                              seed=seed, last_sent=last_sent)
            accumulate_usage(self.last_usage, usage)
        except Exception:  # noqa: BLE001 — сбой Интервьюера не должен терять ход
            raw = ""

        # Отдельного поля со ссылкой на вакансию в hh-контексте нет: ссылку, если она есть, кандидат
        # видит внутри «Описание вакансии». Канонической считаем первую оттуда — тогда выдуманный URL
        # гард ПОДМЕНИТ на настоящий, а не вырежет предложение вместе с ответом.
        description = self._vacancy.get("vacancy_description") or self._vacancy.get("description") or ""
        canon = _urls(description)
        spec = GuardSpec(allow_urls=(canon[0],) if canon else ())
        result = apply_guards(raw, spec, defensive=self._defensive)
        self.last_guard_trips = result.trips

        if not result.text.strip():
            # Собранную кодом `instruction` кандидату отправлять НЕЛЬЗЯ: это директива в повелительном
            # наклонении, адресованная Интервьюеру, — человек увидел бы нашу внутреннюю кухню.
            return reasons.render("REPLY_FALLBACK") or ""
        return result.text

    def _log_salary(self, conversation_id: Any, observation: Observation, plan: Any) -> None:
        audit = (plan.audit or {}).get("salary") or {}
        if audit.get("status") == salary_mod.ABSENT:
            return
        entry = {
            "claim": observation.salary_claim,
            "status": audit.get("status"),
            "normalized": audit.get("normalized"),
            "applied": audit.get("applied"),
            "rules_version": audit.get("rules_version"),
            "verdict": audit.get("verdict"),
            "effect": audit.get("effect"),
            "gates_failed": audit.get("gates_failed"),
        }
        self.store.log_salary_claim(conversation_id, entry)
        self.last_salary = entry

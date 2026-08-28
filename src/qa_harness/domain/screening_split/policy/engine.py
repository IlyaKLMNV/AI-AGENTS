"""Тонкий оркестратор поверх чистого ядра. Всё решение — в `core.decide`, здесь только ввод-вывод.

Сравнение с `..engine.ScreeningSplitEngine` (378 строк): тут нет ни одной ветки принятия решения.
Ни `_forced`, ни `_release_money_stop`, ни `_assumes_salary_closed`, ни `_gate_salary_update`, ни
трёх перерешиваний — им просто негде быть, потому что к моменту выбора действия всё уже посчитано.

Ровно этот файл (минус QA-наблюдаемость) переносится в tgApi и eggplant: `policy/` копируется как
есть, канальным остаётся только стор и способ отправки.

Вызовов LLM за ход: **0** (диалог закрыт) · **1** (ход-скрипт) · **2** (ход с вопросом).
"""

from typing import Any, Optional

from qa_harness.core import accumulate_usage, blank_usage

from .. import salary as salary_mod
from ..errors import AssistantError
from . import reasons
from .context import build_observer_context, has_geo_restriction, salary_forms_for
from .core import DecideContext, decide
from .guards import GuardSpec, apply_guards
from .migration import upgrade
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
        # Снимок хода для QA-трассы. В проде не нужен; поведение от него не зависит.
        self.last_plan: Any = None
        self.last_decision: Optional[dict] = None   # совместимый вид для инвариантов слоя A
        self.last_state: Optional[dict] = None
        self.last_usage: dict = blank_usage()
        self.last_salary: Optional[dict] = None
        self.last_guard_trips: list = []
        self.last_observation: Optional[Observation] = None
        self.last_migration: Optional[dict] = None

    # ── старт ────────────────────────────────────────────────────────────────

    def create_thread(self, vacancy_info: dict, recruiter_name: str, candidate_name: str,
                      candidate_accounts: list) -> str:
        from .. import context as ctx
        from .. import state as state_model

        contact_source = ctx.candidate_source(candidate_accounts)
        # Контекст БЕЗ вилки: секрет не кладут туда, где его потом запрещают (П4).
        context = build_observer_context(recruiter_name, candidate_name, contact_source, vacancy_info)
        seed = ctx.build_interviewer_seed(recruiter_name, candidate_name)

        conversation = self._client.conversations.create(
            items=[{"type": "message", "role": "assistant", "content": seed}]
        )
        conversation_id = conversation.id

        state = state_model.init_state(
            vacancy_info.get("work_format", ""), vacancy_info.get("questions", "") or ""
        )
        self.store.create(
            conversation_id, "policy",
            state=state, context=context,
            location=vacancy_info.get("location", "") or "",
            contact_source=contact_source,
            salary_band={"min": vacancy_info.get("min_salary"),
                         "max": vacancy_info.get("max_salary"),
                         "currency": vacancy_info.get("salary_currency", "RUB")},
        )
        self._vacancy = vacancy_info
        self._said: list[str] = []
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
        self.last_migration = None

        doc = self.store.load(conversation_id)
        if not doc:
            return PolicyResult(reasons.render("REPLY_FALLBACK"), False)
        if doc.get("finished"):
            return PolicyResult(None, True)

        # Ленивая миграция: диалог мог начаться на СТАРОМ движке. Правки — до вызова наблюдателя,
        # иначе ему уедет контекст с вилкой, а зарплатный отсев останется без границ (см. migration).
        self.last_migration = upgrade(doc)

        state = doc["state"]
        band = doc.get("salary_band") or {}
        ctx = DecideContext(
            band_min=band.get("min"), band_max=band.get("max"),
            band_currency=band.get("currency") or "RUB",
            work_format=(doc.get("context") and self._vacancy.get("work_format")) or "",
            location=doc.get("location", ""),
            contact_source=doc.get("contact_source", ""),
            has_geo_restriction=has_geo_restriction(self._vacancy),
        )

        observation, failed = self._observe(doc.get("context", ""), state, message)
        plan = decide(state, observation, message, ctx, analyzer_failed=failed)

        self._said.append(message)
        text = self._speak(conversation_id, plan, message, ctx)

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

    def _speak(self, conversation_id: Any, plan: Any, message: str, ctx: DecideContext) -> str:
        """Скрипт — из реестра; вопрос — через Интервьюера, затем шлюз гардов."""
        if plan.kind == "silent":
            return ""
        if plan.kind == "script":
            return reasons.render(plan.reason_code, city=ctx.location,
                                  contact_source=ctx.contact_source) or ""

        try:
            raw, usage = self.interviewer.run(conversation_id, plan.instruction, message)
            accumulate_usage(self.last_usage, usage)
        except Exception:  # noqa: BLE001 — сбой Интервьюера не должен терять ход
            raw = ""

        company = (self._vacancy.get("company_name") or "").strip().upper()
        url = ((self._vacancy.get("company_info") or {}).get("vacancy_url") or "").strip()
        spec = GuardSpec(
            allow_urls=(url,) if url else (),
            forbid_tokens=salary_forms_for(ctx.band_min, ctx.band_max),
            candidate_texts=tuple(self._said),
            require_question=plan.focus is not None,
            hidden_company=(company == "СКРЫТО"),
        )
        result = apply_guards(raw, spec, defensive=self._defensive)
        self.last_guard_trips = result.trips

        if result.needs_fallback or not result.text.strip():
            # G10: гард унёс слишком много либо Интервьюер промолчал. Кандидату уходит собранная
            # кодом инструкция как есть — она осмысленна, потому что вопрос в ней написан кодом.
            return plan.instruction or reasons.render("REPLY_FALLBACK") or ""
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


def _compat_decision(plan: Any) -> dict:
    """`TurnPlan` → вид, который понимают сегодняшние инварианты слоя A.

    Держится намеренно: `reason_code` совпадает с ключами реестра скриптов, поэтому
    `expect_script_key`, `expect_script_prefix`, `expect_no_script_prefix` и `expect_end` переживают
    смену движка без правок фикстур. Это и есть обещанные «~18 срабатываний из 215» вместо 134.
    """
    charged = ((plan.audit or {}).get("counter_charged") or {}).get("key")
    if plan.kind == "script":
        return {"next_action": "script", "script_key": plan.reason_code, "instruction": None,
                "updates": [], "event": charged, "asking": None,
                "rule": plan.rule, "reason_code": plan.reason_code}
    if plan.kind == "silent":
        return {"next_action": "script", "script_key": None, "instruction": None,
                "updates": [], "event": charged, "asking": None,
                "rule": plan.rule, "reason_code": plan.reason_code}
    return {"next_action": "ask", "script_key": None, "instruction": plan.instruction,
            "updates": [], "event": charged, "asking": plan.focus,
            "rule": plan.rule, "reason_code": plan.reason_code,
            "instruction_parts": plan.instruction_parts}

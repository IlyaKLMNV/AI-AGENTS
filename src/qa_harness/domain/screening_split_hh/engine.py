"""Split-движок скрининга — HH-канал (оркестрация Аналитик → state → скрипт/Интервьюер).

Аналог `screening_split/engine.py` (порт `ScreeningSplitEngine` из tgApi); дельта под hh —
эталон `EGGPLANT_SPLIT_TASK.md` §9. Поток хода идентичен TG, отличия ТОЛЬКО:
- `_EVENT_STOP` без `contact_source` (в hh этого события нет);
- добавлена ветка reask-cap по `asking == "field_work"` (`field_work_reasks >= 2 → STOP_PERSISTENT`);
- `create_thread` строит hh-контекст (`context.build_context` без contact_source) и hh-state
  (`init_state(allowed_formats, ...)`); `render_script(city=…)` без параметра `contact_source`.

Идентичные с TG вещи (`ConversationResult`, `NO_PROGRESS_CAP=4`, стоп-кран no_progress, лимит
переспросов salary/format/вопрос, terminal-guard) сохранены 1:1. `ConversationResult` переиспользуем
импортом; клиенты инжектятся снаружи (домен НЕ импортирует app/openai/prompts на уровне модуля).
"""

from typing import Any, Optional

from qa_harness.core import accumulate_usage, blank_usage
from qa_harness.domain.screening_split.engine import ConversationResult  # канало-независим
from qa_harness.domain.screening_split.errors import AssistantError

from . import context as ctx
from . import scripts
from . import state as state_model

# Пороги событийных счётчиков → STOP форсит КОД (значение ПОСЛЕ инкремента). Без contact_source (hh).
_EVENT_STOP = {
    "gibberish": (2, "STOP_GIBBERISH_REPEAT"),
    "bot_check": (2, "STOP_BOT_REPEAT"),
    "demand": (3, "STOP_PERSISTENT"),
    "pause": (3, "STOP_PAUSE"),
}

NO_PROGRESS_CAP = 4


def _is_terminal_decision(decision: dict) -> bool:
    """Аналитик уже завершает диалог этим решением (KO_*/STOP_*/FINISH) — код-форсы его НЕ затирают."""
    script_key = decision.get("script_key")
    return (decision.get("next_action") == "script"
            and isinstance(script_key, str)
            and scripts.is_terminal(script_key))


class ScreeningSplitEngine:
    def __init__(self, store: Any, analyzer: Any, interviewer: Any, client: Any) -> None:
        self.store = store
        self.analyzer = analyzer
        self.interviewer = interviewer
        self._client = client
        self.last_decision: Optional[dict] = None
        self.last_state: Optional[dict] = None
        self.last_usage: dict = blank_usage()

    def create_thread(
        self,
        vacancy_info: dict,
        recruiter_name: str,
        candidate_name: str,
    ) -> str:
        # Полный hh-контекст (с секретами) — только для Аналитика; в hh contact_source нет.
        context = ctx.build_context(recruiter_name, candidate_name, vacancy_info)
        seed = ctx.build_interviewer_seed(recruiter_name, candidate_name)

        conversation = self._client.conversations.create(
            items=[{"type": "message", "role": "assistant", "content": seed}]
        )
        conversation_id = conversation.id

        allowed_formats = ctx.allowed_formats_of(vacancy_info)
        state = state_model.init_state(allowed_formats, vacancy_info.get("questions", "") or "")
        self.store.create(
            conversation_id,
            "split",
            state=state,
            context=context,
            location=vacancy_info.get("location", "") or "",
            contact_source="",  # в hh не используется; поле стора оставляем ради общего интерфейса
        )
        return conversation_id

    def add_message_and_run(self, conversation_id: Any, message: str) -> ConversationResult:
        self.last_decision = None
        self.last_state = None
        self.last_usage = blank_usage()

        doc = self.store.load(conversation_id)
        if not doc:
            return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)
        if doc.get("finished"):
            return ConversationResult(None, True)

        state = doc["state"]
        progress_before = state_model.progress_signature(state)
        context = doc.get("context", "")
        location = doc.get("location", "")

        try:
            decision = self._analyze(context, state, message)
        except AssistantError:
            self.last_decision = {"next_action": "fallback", "script_key": "REPLY_FALLBACK",
                                  "source": "analyzer_error"}
            return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)

        new_state = state_model.apply_updates(state, decision.get("updates"), decision.get("event"))

        # --- порог событийных счётчиков: STOP форсит КОД (не затираем терминальное решение Аналитика) ---
        _ev = decision.get("event")
        _forced = False
        if _ev in _EVENT_STOP and not _is_terminal_decision(decision):
            _thr, _sk = _EVENT_STOP[_ev]
            if new_state.get("counters", {}).get(_ev, 0) >= _thr:
                decision = {"next_action": "script", "script_key": _sk, "source": "counter_cap"}
                _forced = True

        # --- детерминированный лимит переспросов по decision.asking (2 → форс завершения) ---
        asking = decision.get("asking")
        if not _forced and decision.get("next_action") == "ask" and asking and state.get("last_asking") == asking:
            if asking == "salary" and new_state.get("salary") == "pending":
                if new_state.get("salary_reasks", 0) >= 2:
                    decision = {"next_action": "script", "script_key": "STOP_SALARY_DEMAND", "source": "reask_cap"}
                else:
                    new_state["salary_reasks"] = new_state.get("salary_reasks", 0) + 1
            elif asking == "format" and new_state.get("format_check") == "pending":
                if new_state.get("format_reasks", 0) >= 2:
                    decision = {"next_action": "script", "script_key": "STOP_PERSISTENT", "source": "reask_cap"}
                else:
                    new_state["format_reasks"] = new_state.get("format_reasks", 0) + 1
            elif asking == "field_work" and new_state.get("field_work_check") == "pending":
                # новая ветка hh (в tg её нет): разъездной формат как отдельный приоритет
                if new_state.get("field_work_reasks", 0) >= 2:
                    decision = {"next_action": "script", "script_key": "STOP_PERSISTENT", "source": "reask_cap"}
                else:
                    new_state["field_work_reasks"] = new_state.get("field_work_reasks", 0) + 1
            else:
                q = next((x for x in new_state.get("questions", []) if x["key"] == asking), None)
                if q and q["status"] == "pending":
                    if q.get("reask_count", 0) >= 2:
                        new_state = state_model.apply_updates(new_state, [{"key": asking, "value": "refused"}])
                        new_state["last_asking"] = None
                        try:
                            decision = self._analyze(context, new_state, message)
                        except AssistantError:
                            self.last_decision = {"next_action": "fallback", "script_key": "REPLY_FALLBACK",
                                                  "source": "analyzer_error"}
                            self.last_state = new_state
                            self.store.save_state(conversation_id, new_state)
                            return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)
                        new_state = state_model.apply_updates(new_state, decision.get("updates"), decision.get("event"))
                    else:
                        new_state = state_model.apply_updates(new_state, [{"key": asking, "value": "reasked"}])

        # --- универсальный стоп-кран: N ходов подряд без прогресса state → форс завершения ---
        if state_model.progress_signature(new_state) == progress_before:
            new_state["no_progress"] = new_state.get("no_progress", 0) + 1
        else:
            new_state["no_progress"] = 0
        if not _is_terminal_decision(decision) and new_state["no_progress"] >= NO_PROGRESS_CAP:
            key = "FINISH" if state_model.is_complete(new_state) else "STOP_PERSISTENT"
            decision = {"next_action": "script", "script_key": key, "source": "no_progress_cap"}

        self.last_decision = decision

        if decision.get("next_action") == "script":
            key = decision.get("script_key")
            text = scripts.render_script(key, city=location)  # hh: подстановка только {city}
            if text is None:
                text, end = scripts.render_script("REPLY_FALLBACK"), False
            else:
                end = scripts.is_terminal(key)
        else:  # ask
            instruction = decision.get("instruction") or ""
            text, iusage = self.interviewer.run(conversation_id, instruction, message)
            accumulate_usage(self.last_usage, iusage)
            new_state["last_asked"] = instruction
            new_state["last_asking"] = decision.get("asking")
            end = False

        self.last_state = new_state
        self.store.save_state(conversation_id, new_state, finished=end)
        return ConversationResult(text or None, end)

    def _analyze(self, context: str, state: dict, message: str) -> dict:
        decision = self.analyzer.run(context, state, message)
        accumulate_usage(self.last_usage, self.analyzer.last_usage)
        return decision

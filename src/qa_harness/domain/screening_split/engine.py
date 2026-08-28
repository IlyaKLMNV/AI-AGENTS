"""Split-движок скрининга: оркестрация Аналитик → state → скрипт/Интервьюер.

Порт `ScreeningSplitEngine` (tgApi, HEAD e733095). Поток одного хода (add_message_and_run):
  1) загрузить state из стора (по conversation_id);
  2) Аналитик(context, state, message) → Decision (при сбое → REPLY_FALLBACK, не завершаем);
  3) применить updates к state (монотонно), отбросив закрытие зарплаты без годного claim;
  4) ЗАРПЛАТА: вердикт по вилке считает КОД по `Decision.salary_claim` (см. модуль salary). Ход
     перерешивается (второй вызов Аналитика) в двух случаях: код сумму не принял, а решение исходило
     из «деньги закрыты»; либо код принял и закрыл, а решение всё ещё переспрашивает про деньги;
  5) инкремент `event` — ПОСЛЕ всех вызовов Аналитика; пороги счётчиков и лимит переспросов форсит КОД;
  6) ветка script → текст из реестра (end по is_terminal) / ветка ask → Интервьюер;
  7) сохранить state (на терминальном ходе — с пометкой finished).

Порядок 4 → 6 важен: зарплатный вердикт считается ДО событийных порогов, иначе кандидат, который
третий раз давит и одновременно назвал сумму выше максимума, завершался бы как STOP_PERSISTENT.

Адаптации под ai-agents (прод-логика поведения НЕ меняется):
- стор/аналитик/интервьюер/OpenAI-клиент инжектятся (см. фабрику в раннере), а не тянутся из app;
- create_thread принимает `vacancy_info: dict` вместо ScreeningVacancyDTO;
- НАБЛЮДАЕМОСТЬ для QA: last_decision / last_state / last_usage / last_salary — снимок хода для
  tool_trace, инвариантов слоя A и учёта токенов. В проде эти поля не нужны; поведение от них не зависит.
"""

from dataclasses import dataclass
from typing import Any, Optional

from qa_harness.core import accumulate_usage, blank_usage

from . import context as ctx
from . import salary as salary_mod
from . import scripts
from . import state as state_model
from .errors import AssistantError

# Пороги событийных счётчиков → STOP форсит КОД (значение счётчика ПОСЛЕ инкремента).
_EVENT_STOP = {
    "gibberish": (2, "STOP_GIBBERISH_REPEAT"),
    "bot_check": (2, "STOP_BOT_REPEAT"),
    "demand": (3, "STOP_PERSISTENT"),
    "contact_source": (3, "STOP_PERSISTENT"),
    "pause": (3, "STOP_PAUSE"),
}

# Универсальный стоп-кран: N ходов подряд без прогресса state → форс завершения (порт tgApi).
# Ловит любое зацикливание независимо от темы/распознавания Аналитиком. 4 не задевает здоровые диалоги.
NO_PROGRESS_CAP = 4

# Терминальные решения Аналитика, мотивированные ДЕНЬГАМИ. Только их СНИМАЕТ вердикт `fits`:
# «ниже 400 не собираюсь рассматривать ваше предложение» модель читает как отказ от вакансии, хотя по
# вилке кандидат проходит и диалог надо продолжать.
# Граница именно здесь, а не «код всегда главнее»: иначе диалог продолжался бы после STOP_ABUSE или
# STOP_POLITICS только потому, что кандидат заодно назвал сумму.
_MONEY_DECISIONS = frozenset({"KO_SALARY", "STOP_SALARY_DEMAND", "STOP_NOT_INTERESTED"})

# Решения, которые ПЕРЕБИВАЕТ вердикт `ko`. К денежным добавлен `FINISH`: это не суждение о реплике, а
# отметка «всё собрано», и если тем же ходом кандидат назвал сумму выше максимума, скрининг успешным
# не был. Обратной силы у этого нет — `fits` из `FINISH` фолбэк не делает (см. _MONEY_DECISIONS).
_KO_OVERRIDABLE = _MONEY_DECISIONS | {"FINISH"}

# Служебная строка второму вызову при перерешивании. Без неё вход второго вызова ТОЖДЕСТВЕННЫЙ
# (state не менялся) и при temperature=0 решение повторяется — перерешивание было бы no-op.
# Инструкцию первого решения намеренно НЕ передаём: модель протаскивала из неё переход к следующему
# приоритету. Валюту не навязываем — где нужны именно рубли (`currency: other`), это делает промпт.
_SALARY_REWIND_NOTE = (
    "сумму из этой реплики использовать нельзя, вопрос зарплаты НЕ закрыт. "
    "Верни решение заново: остальное содержание реплики обработай как обычно (ответь на вопросы "
    "кандидата, отработай сработавшие триггеры) — изменилось ровно одно, зарплата НЕ закрыта. "
    "Попроси кандидата уточнить его зарплатные ожидания в формате оплаты за месяц числом, чтобы "
    "передать коллегам. Поставь asking=\"salary\", salary в updates НЕ закрывай, причину уточнения "
    "не объясняй."
)


def _is_terminal_decision(decision: dict) -> bool:
    """Аналитик уже завершает диалог этим решением (KO_*/STOP_*/FINISH) — код-форсы его НЕ затирают.
    script_key при next_action="ask" равен None, поэтому проверяем тип (is_terminal(None) упал бы)."""
    script_key = decision.get("script_key")
    return (decision.get("next_action") == "script"
            and isinstance(script_key, str)
            and scripts.is_terminal(script_key))


def _is_money_decision(decision: dict) -> bool:
    """Аналитик завершает диалог по денежному мотиву — такое решение зарплатный вердикт кода снимает."""
    return (decision.get("next_action") == "script"
            and decision.get("script_key") in _MONEY_DECISIONS)


def _salary_verdict_wins(decision: dict) -> bool:
    """Вправе ли отсев по зарплате перебить решение этого хода.

    Да — если Аналитик не завершает диалог сам, завершает по денежному мотиву либо ставит `FINISH`.
    Нет — если он вернул НЕденежное терминальное решение по существу реплики (оскорбления, политика,
    мошенничество, отсев по формату): такие мотивы приоритетнее отсева по деньгам, и продолжать
    диалог нельзя только потому, что кандидат заодно назвал сумму.
    """
    return (not _is_terminal_decision(decision)
            or decision.get("script_key") in _KO_OVERRIDABLE)


def _release_money_stop(decision: dict) -> dict:
    """Снимает денежное завершение, с которым код не согласен, БЕЗ второго вызова LLM.

    Исполняем остаток решения Аналитика: есть инструкция — отдаём её Интервьюеру; терминальный скрипт
    инструкции не несёт, поэтому уходим в нетерминальный REPLY_FALLBACK («давайте продолжим»), а
    следующий ход Аналитик начнёт уже с закрытой зарплатой в STATE.
    """
    instruction = decision.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return {"next_action": "ask", "script_key": None, "instruction": instruction,
                "updates": [], "event": decision.get("event"), "asking": decision.get("asking"),
                "source": "salary_fits_release"}
    return {"next_action": "script", "script_key": "REPLY_FALLBACK", "source": "salary_fits_release"}


def _assumes_salary_closed(decision: dict, state: dict) -> bool:
    """Решение построено на посылке «зарплата закрыта», хотя код сумму не принял.

    Три признака, каждый достаточен: Аналитик сам поставил `salary: closed`; вернул `FINISH` (скрининг
    нельзя завершать успешно с неразрешённой зарплатой); ушёл спрашивать следующий приоритет
    (`asking` = формат или доп-вопрос). `asking: null` признаком НЕ считается: отвечать на вопрос
    кандидата, не двигая приоритеты, — законный ход, а не расхождение с кодом.
    """
    if state.get("salary") != "pending":
        return False  # пункт закрыт раньше — расхождения нет
    if any(isinstance(u, dict) and u.get("key") == "salary" and u.get("value") == "closed"
           for u in (decision.get("updates") or [])):
        return True
    if _is_terminal_decision(decision):
        return decision.get("script_key") == "FINISH"
    asking = decision.get("asking")
    return bool(asking) and asking != "salary"


def _gate_salary_update(updates, claim_status: str):
    """`salary: closed` от Аналитика действует ТОЛЬКО вместе с годным claim.

    Иначе пункт закрывался бы без сравнения с вилкой, и кандидат с ожиданием выше максимума проходил
    бы скрининг, когда сумма не разобрана. Закрыть зарплату, не сравнив её, становится невозможно
    по построению — а не по дисциплине промпта.
    """
    if claim_status == salary_mod.ACTIONABLE:
        return updates
    return [u for u in (updates or [])
            if not (isinstance(u, dict) and u.get("key") == "salary")]


@dataclass
class ConversationResult:
    """Результат хода движка (совместим по смыслу с прод ConversationResult)."""

    response: Optional[str]
    conversation_end: bool


class ScreeningSplitEngine:
    def __init__(self, store: Any, analyzer: Any, interviewer: Any, client: Any) -> None:
        self.store = store
        self.analyzer = analyzer
        self.interviewer = interviewer
        self._client = client
        # Снимок последнего хода (read-only introspection для QA-трассы/токенов).
        self.last_decision: Optional[dict] = None
        self.last_state: Optional[dict] = None
        self.last_usage: dict = blank_usage()
        # Зарплатный разбор хода (claim/статус/вердикт/effect) — для инвариантов слоя A по трассе.
        self.last_salary: Optional[dict] = None

    def create_thread(
        self,
        vacancy_info: dict,
        recruiter_name: str,
        candidate_name: str,
        candidate_accounts: list,
    ) -> str:
        contact_source = ctx.candidate_source(candidate_accounts)
        # Полный контекст (с секретами) — только для Аналитика, хранится в state-записи.
        context = ctx.build_context(recruiter_name, candidate_name, contact_source, vacancy_info)
        # Интервьюер сидируется только участниками (least-privilege): о вакансии не знает.
        seed = ctx.build_interviewer_seed(recruiter_name, candidate_name)

        conversation = self._client.conversations.create(
            items=[{"type": "message", "role": "assistant", "content": seed}]
        )
        conversation_id = conversation.id

        state = state_model.init_state(
            vacancy_info.get("work_format", ""), vacancy_info.get("questions", "") or ""
        )
        self.store.create(
            conversation_id,
            "split",
            state=state,
            context=context,
            location=vacancy_info.get("location", "") or "",
            contact_source=contact_source,
            salary_band={"min": vacancy_info.get("min_salary"), "max": vacancy_info.get("max_salary")},
        )
        return conversation_id

    def add_message_and_run(self, conversation_id: Any, message: str) -> ConversationResult:
        self.last_decision = None
        self.last_state = None
        self.last_usage = blank_usage()
        self.last_salary = None

        doc = self.store.load(conversation_id)
        if not doc:
            return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)
        if doc.get("finished"):
            # диалог закрыт — молчим и остаёмся закрытыми, а не переоткрываем его
            return ConversationResult(None, True)

        state = doc["state"]
        progress_before = state_model.progress_signature(state)
        context = doc.get("context", "")
        location = doc.get("location", "")
        contact_source = doc.get("contact_source", "")

        try:
            decision = self._analyze(context, state, message)
        except AssistantError:
            self.last_decision = {"next_action": "fallback", "script_key": "REPLY_FALLBACK",
                                  "source": "analyzer_error"}
            return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)

        # Зарплата: РАСПОЗНАЛ сумму Аналитик (`salary_claim` — чьи деньги, какое из чисел ожидание,
        # диапазон или порог, net/gross, валюта, период), СЧИТАЕТ и РЕШАЕТ код. Порт tgApi.
        claim = salary_mod.read_claim(decision)
        claim_status = salary_mod.claim_status(claim, message)
        salary_value = None
        if claim_status == salary_mod.ACTIONABLE:
            salary_value = salary_mod.normalize(claim)
            if salary_value is None:
                # Пересчёт дал неправдоподобную сумму (модель ошиблась полем scale/period) — это
                # уточняющий вопрос, а не отсев. Статус портим ДО гейта: иначе такой ход всё ещё
                # имел бы право закрыть зарплату, не сравнив её с вилкой.
                claim_status = salary_mod.UNUSABLE

        # Право закрыть зарплату есть только у годного claim — иначе update отбрасываем.
        # `event` здесь НЕ применяем: счётчик описывает поведение кандидата в ЭТОМ сообщении, значит
        # каждый вызов Аналитика на этом ходе обязан видеть его одинаковым. Иначе второй вызов
        # прочитает «вы бот?» из этой же реплики как повтор и завершит диалог STOP_BOT_REPEAT
        # кандидату, который спросил раз. Инкремент — ниже, после всех вызовов.
        new_state = state_model.apply_updates(
            state, _gate_salary_update(decision.get("updates"), claim_status))

        # --- расхождение: код сумму не принял, а решение построено на посылке «деньги закрыты» ---
        # Аналитик не может знать заранее, что claim забракуют (цитата не сойдётся с репликой, значение
        # вне enum, обе границы пустые), поэтому он законно уходит к следующему приоритету. Без
        # перерешивания кандидату уходит вопрос про город или доп-вопрос, хотя про деньги не
        # договорились, а `FINISH` вообще закрыл бы скрининг с неразрешённой зарплатой.
        # Реплику подаём ту же: ничего из сказанного кандидатом не теряется, а state уже обогащён
        # фактами первого решения (город, закрытые доп-вопросы) — отброшено только `salary: closed`.
        _salary_rewind = False
        _event = decision.get("event")  # событие ПЕРВОГО решения; второе своё не приносит
        if claim_status == salary_mod.UNUSABLE and _assumes_salary_closed(decision, new_state):
            try:
                decision = self._analyze(context, new_state, message, note=_SALARY_REWIND_NOTE)
            except AssistantError:
                self.last_decision = {"next_action": "fallback", "script_key": "REPLY_FALLBACK",
                                      "source": "analyzer_error"}
                self.last_state = new_state
                self.store.save_state(conversation_id, new_state)
                return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)
            # `event` второго решения игнорируем: поведение кандидата одно, счёт по нему один
            # (недосчёт в безопасную сторону). `updates` гейтим тем же статусом.
            new_state = state_model.apply_updates(
                new_state, _gate_salary_update(decision.get("updates"), claim_status))
            _salary_rewind = True

        # --- зарплата: вердикт по вилке за кодом ---
        # Блок стоит ВЫШЕ событийных порогов намеренно: иначе кандидат, который третий раз давит и
        # одновременно назвал сумму выше максимума, получал STOP_PERSISTENT, и в отчётности причина
        # отказа была не та. Второго вызова LLM блок не делает, поэтому цена приоритета нулевая.
        # Состояние `salary` не проверяем: сравниваем на ЛЮБОМ ходе с годным claim, в том числе после
        # закрытия пункта — кандидат, поднявший ожидания в середине диалога, должен быть отсеян.
        _forced = False
        salary_verdict = None
        salary_effect = None  # что вердикт СДЕЛАЛ с ходом — иначе по трассе отсев не отличить
        if salary_value is not None:
            _band = doc.get("salary_band") or {}
            salary_verdict = salary_mod.compare_with_band(
                salary_value, _band.get("min"), _band.get("max"))
            if salary_verdict == "ko":
                if _salary_verdict_wins(decision):
                    decision = {"next_action": "script", "script_key": "KO_SALARY",
                                "source": "salary_band"}
                    _forced = True
                    salary_effect = "ko_forced"
                else:
                    # Аналитик завершает диалог по НЕденежному мотиву (оскорбления, политика,
                    # отсев по формату) — он приоритетнее отсева по деньгам.
                    salary_effect = "ko_overridden_by_analyzer"
            else:
                # Ниже минимума вилки отказом не является: молча закрываем и идём дальше.
                new_state = state_model.apply_updates(
                    new_state, [{"key": "salary", "value": "closed"}])
                salary_effect = "closed"
                if _is_money_decision(decision):
                    # Аналитик завершил диалог по деньгам, а код с этим не согласен — снимаем.
                    decision = _release_money_stop(decision)
                    salary_effect = "closed_money_stop_released"
                elif decision.get("next_action") == "ask" and decision.get("asking") == "salary":
                    # Код сумму принял и пункт закрыл, а решение всё ещё переспрашивает про деньги:
                    # Аналитик писал его до того, как код посчитал. Без второго вызова кандидату
                    # уходит вопрос, на который он только что ответил (прогон 28.08, сценарий 45).
                    # Служебная строка тут НЕ нужна, в отличие от перерешивания выше: состояние
                    # реально изменилось (pending → closed), и второй вызов увидит это сам.
                    try:
                        decision = self._analyze(context, new_state, message)
                    except AssistantError:
                        self.last_decision = {"next_action": "fallback", "script_key": "REPLY_FALLBACK",
                                              "source": "analyzer_error"}
                        self.last_state = new_state
                        self.store.save_state(conversation_id, new_state)
                        return ConversationResult(scripts.render_script("REPLY_FALLBACK"), False)
                    new_state = state_model.apply_updates(
                        new_state, _gate_salary_update(decision.get("updates"), claim_status))
                    salary_effect = "closed_reask_dropped"

        # Инкремент счётчика — ЗДЕСЬ, когда все вызовы Аналитика этого хода отработали. Единственный
        # потребитель — блок порогов ниже; в `progress_signature` счётчиков нет, поэтому
        # `no_progress` от места инкремента не зависит.
        if _event:
            new_state = state_model.apply_updates(new_state, [], _event)

        if claim_status != salary_mod.ABSENT:
            entry = {
                "claim": claim,
                "status": claim_status,
                "normalized": ({"min": salary_value.min, "max": salary_value.max}
                               if salary_value else None),
                "applied": salary_value.applied if salary_value else None,
                "rules_version": (salary_value.rules_version if salary_value
                                  else salary_mod.SALARY_RULES_VERSION),
                "verdict": salary_verdict,
                "effect": salary_effect,
                # Ход перерешён из-за расхождения: по трассе должно быть видно, что инструкцию
                # кандидату сформировал ВТОРОЙ Decision, а не первый.
                "rewind": _salary_rewind,
            }
            self.store.log_salary_claim(conversation_id, entry)
            self.last_salary = entry

        # --- порог событийных счётчиков: STOP форсит КОД (Аналитик лишь эмитит event) ---
        # Терминальное решение Аналитика (FINISH+pause, KO_SALARY+demand) НЕ затираем счётчик-форсом.
        # Берём `_event` (событие ПЕРВОГО решения): после перерешивания в `decision` лежит второе,
        # чьё событие мы намеренно не считали — порог обязан проверяться по тому, что попало в счётчик.
        if not _forced and _event in _EVENT_STOP and not _is_terminal_decision(decision):
            _thr, _sk = _EVENT_STOP[_event]
            if new_state.get("counters", {}).get(_event, 0) >= _thr:
                decision = {"next_action": "script", "script_key": _sk, "source": "counter_cap"}
                _forced = True

        # --- детерминированный лимит переспросов по decision.asking (2 → форс завершения) ---
        # Перерешённый ход кап НЕ жжёт: этот вопрос задан не потому, что кандидат уклонился, а потому
        # что claim забраковал код. Считать такой переспрос значило бы расходовать бюджет кандидата на
        # наши же ошибки — ровно то, из-за чего в инциденте 2026-08-17 диалог завершился ложно.
        # Когда уклонился действительно кандидат (чужая сумма, проценты, «обсуждаемо»), Аналитик по
        # промпту сам ставит asking="salary" и никуда не уходит — перерешивания нет, кап считает как
        # обычно. `no_progress` при этом продолжает тикать: он универсальная страховка от зацикливания.
        asking = decision.get("asking")
        if not _forced and not _salary_rewind and decision.get("next_action") == "ask" \
                and asking and state.get("last_asking") == asking:
            if asking == "salary" and new_state.get("salary") == "pending":
                # Порог 2 = STOP на 3-м переспросе (проверка ДО инкремента) — как в обоих продах и как
                # написано в промптах Аналитика. Временный порог 3 (инцидент 2026-08-17, Баг B,
                # сценарий 64) откачен: он лечил симптом, гейт мерил поведение мягче прода. Настоящая
                # причина ложного STOP — счётчик тикает на ЛЮБОМ ходу с asking=salary, включая ходы
                # со своими вопросами кандидата; лечится гейтом «кап считаем только на ходах без
                # прогресса» (Д1 в docs/screening_split/plan_cross_repo.md), правка идёт в три порта.
                if new_state.get("salary_reasks", 0) >= 2:
                    decision = {"next_action": "script", "script_key": "STOP_SALARY_DEMAND", "source": "reask_cap"}
                else:
                    new_state["salary_reasks"] = new_state.get("salary_reasks", 0) + 1
            elif asking == "format" and new_state.get("format_check") == "pending":
                if new_state.get("format_reasks", 0) >= 2:
                    decision = {"next_action": "script", "script_key": "STOP_PERSISTENT", "source": "reask_cap"}
                else:
                    new_state["format_reasks"] = new_state.get("format_reasks", 0) + 1
            else:
                q = next((x for x in new_state.get("questions", []) if x["key"] == asking), None)
                if q and q["status"] == "pending":
                    if q.get("reask_count", 0) >= 2:
                        # лимит доп-вопроса → refused и перерешаем ход
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
                        # Событие второго решения не применяем — как и в зарплатном перерешивании.
                        new_state = state_model.apply_updates(new_state, decision.get("updates"))
                    else:
                        new_state = state_model.apply_updates(new_state, [{"key": asking, "value": "reasked"}])

        # --- универсальный стоп-кран: N ходов подряд без прогресса state → форс завершения ---
        # Считаем ЗДЕСЬ, после ветки refused-перерешивания (она заново применяет updates, до неё рано).
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
            text = scripts.render_script(key, city=location, contact_source=contact_source)
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

        # Снимок итогового state (для QA-трассы) + сохранение (на terminal — finished=True).
        self.last_state = new_state
        self.store.save_state(conversation_id, new_state, finished=end)
        return ConversationResult(text or None, end)

    def _analyze(self, context: str, state: dict, message: str, *, note: str | None = None) -> dict:
        """Обёртка над Аналитиком: копит usage хода (в т.ч. по повторному вызову при reask-cap)."""
        decision = self.analyzer.run(context, state, message, note=note)
        accumulate_usage(self.last_usage, self.analyzer.last_usage)
        return decision

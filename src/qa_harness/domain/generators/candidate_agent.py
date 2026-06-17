"""Адаптивный LLM-кандидат: генерит СЛЕДУЮЩУЮ реплику, реагируя на ответ ассистента.

В отличие от легаси (батч-список реплик заранее) кандидат видит историю и последнюю реплику ассистента
и отвечает вживую — это и есть «реальные диалоги разнятся». Суть реплики диктует сценарий
(`CandidateConstraints`: триггер/гайдлайны/что обязательно/что запрещено — перенос легаси-правил как ДАННЫХ),
поверхность — `VariantStyle`. Каждая реплика валидируется и при провале перегенерится/фолбэчится движком
(`generate_valid`). Сеть — только через инъектируемый client (.create(input)->(text,usage)); офлайн-тест фейком.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .engine import Attempt, GenerationPolicy, GenResult, generate_valid
from .variety import VariantStyle

_CYR_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN_RUN_RE = re.compile(r"[A-Za-z]{3,}")
_DIGITS_RE = re.compile(r"\d{2,}")
_ROLE_PREFIX_RE = re.compile(r"^\s*(кандидат|candidate|рекрутер|recruiter|ассистент|assistant)\s*[:\-]\s*", re.IGNORECASE)


@dataclass
class CandidateConstraints:
    """Правила сценария для генерации/валидации реплики кандидата (данные, не код)."""

    scenario_name: str
    scenario_description: str
    trigger_requirement: str = ""          # инструкция генератору: что обязано быть в реплике
    guidelines: List[str] = field(default_factory=list)
    require_any: List[str] = field(default_factory=list)  # хотя бы один маркер (валидация, регистр игнор)
    forbid_any: List[str] = field(default_factory=list)   # ни одного маркера (валидация)
    examples: List[str] = field(default_factory=list)
    fallback: List[str] = field(default_factory=list)     # детерминированные запасные реплики
    language: str = "ru"                   # "ru" | "foreign"
    forbid_digits: bool = False            # запретить числа (напр. «странный» кандидат не называет зарплату)
    max_turns: Optional[int] = None        # per-scenario лимит ходов адаптивного диалога (None → глобальный)


def _normalize(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # модель иногда отдаёт несколько строк / ролевой префикс / служебный END — чистим
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = " ".join(lines)
    text = _ROLE_PREFIX_RE.sub("", text)
    text = text.replace("END", "").strip()
    return text


def validate_candidate_turn(text: str, c: CandidateConstraints) -> Optional[str]:
    """None == ок; строка == причина отказа (уйдёт в трассу и подсказку для regen)."""
    t = _normalize(text)
    if not t:
        return "пустая реплика"
    low = t.lower()
    if c.forbid_any and any(m.lower() in low for m in c.forbid_any):
        hit = next(m for m in c.forbid_any if m.lower() in low)
        return f"присутствует запрещённый маркер: '{hit}'"
    if c.require_any and not any(m.lower() in low for m in c.require_any):
        return f"нет ни одного обязательного маркера из: {c.require_any}"
    if c.forbid_digits and _DIGITS_RE.search(t):
        return "реплика не должна содержать числа (кандидат остаётся неинформативным)"
    if c.language == "foreign":
        if not _LATIN_RUN_RE.search(t):
            return "реплика должна быть на иностранном языке (нет латиницы)"
        if len(_CYR_RE.findall(t)) > 2:
            return "реплика на иностранном языке не должна содержать кириллицу"
    return None


class CandidateAgent:
    """Адаптивный кандидат: next_turn(history, assistant_last_reply) -> GenResult с репликой."""

    def __init__(self, client: Any, model: str, constraints: CandidateConstraints,
                 style: VariantStyle, policy: Optional[GenerationPolicy] = None) -> None:
        self._client = client
        self._model = model
        self._c = constraints
        self._style = style
        self._policy = policy or GenerationPolicy(max_retries=1)

    def _instruction(self) -> str:
        return (
            "Ты симулируешь ОДНО следующее сообщение КАНДИДАТА в переписке с рекрутёром-ассистентом.\n"
            "Дан СЦЕНАРИЙ поведения кандидата и история диалога. Сгенерируй ровно одну реплику кандидата,\n"
            "которая двигает сценарий и реалистично реагирует на последнюю реплику ассистента.\n"
            "Правила:\n"
            "- Верни ТОЛЬКО текст реплики кандидата. Без префиксов 'Кандидат:'/'Рекрутёр:', без кавычек,\n"
            "  без JSON, без markdown, без пояснений.\n"
            "- НИКОГДА не пиши от лица рекрутёра/ассистента и не объясняй процессы найма.\n"
            "- НЕ добавляй служебные токены (END и т.п.).\n"
            "- Держись сути сценария; не превращай провокацию в вежливый нейтральный текст."
        )

    def _payload(self, history: List[Tuple[str, str]], assistant_last_reply: Optional[str],
                 attempt: Attempt) -> str:
        ctx: dict = {
            "scenario_name": self._c.scenario_name,
            "scenario_description": self._c.scenario_description,
            "style": self._style.hint(),
            "history": [{"role": r, "text": t} for r, t in history],
            "assistant_last_reply": assistant_last_reply,
        }
        if self._c.trigger_requirement:
            ctx["trigger_requirement"] = self._c.trigger_requirement
        if self._c.guidelines:
            ctx["guidelines"] = self._c.guidelines
        if self._c.examples:
            ctx["examples_style_only"] = self._c.examples
        if self._c.language == "foreign":
            ctx["language"] = "пиши ТОЛЬКО на иностранном языке (латиница), без кириллицы"
        if self._c.forbid_digits:
            ctx["no_numbers"] = "не называй конкретные числа (зарплату/суммы/годы) — оставайся уклончивым"
        if attempt.last_error:
            ctx["correction"] = (f"Прошлая реплика отклонена: {attempt.last_error}. "
                                 "Исправь и усиль соответствие сценарию.")
        if attempt.avoid:
            ctx["avoid_repeating"] = [str(x) for x in attempt.avoid][-4:]
        return "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\nВерни одну реплику кандидата:"

    def next_turn(self, history: List[Tuple[str, str]], assistant_last_reply: Optional[str],
                  turn_index: int = 0) -> GenResult:
        def produce(attempt: Attempt) -> Tuple[str, Any]:
            text, usage = self._client.create(
                self._instruction() + "\n\n" + self._payload(history, assistant_last_reply, attempt)
            )
            return _normalize(text), usage

        def fallback() -> Optional[str]:
            if not self._c.fallback:
                return None
            return _normalize(self._c.fallback[turn_index % len(self._c.fallback)])

        return generate_valid(
            produce,
            lambda item: validate_candidate_turn(item, self._c),
            policy=self._policy,
            fallback=fallback,
        )

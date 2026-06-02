"""Генератор одного сообщения кандидата с известным целевым классом + валидация.

Перенос CandidateMessageSynthesizer и _validate_generated_message из старого раннера.
Генератор НЕ использует message_classifier для построения датасета (метка известна заранее).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..text.message_markers import (
    ACCEPTANCE_PATTERNS,
    DECLINE_PATTERNS,
    HUMAN_NEEDED_PATTERNS,
    REASON_PATTERNS,
    has_any_pattern,
)
from .base import Generator
from .message_specs import class_examples, class_requirements


@dataclass
class MessageSpec:
    cdm: Dict[str, Any]
    target_class: str
    scenario_hint: str
    noise_level: int


class CandidateMessageGenerator(Generator):
    """Генерирует ОДНО сообщение на русском под заданный TARGET_CLASS."""

    def instruction(self, spec: MessageSpec) -> str:
        return (
            "You generate exactly ONE candidate message in Russian after the recruiter's first outreach.\n"
            "You will be given TARGET_CLASS: reason_farewell / no_reason / acceptance / human_needed.\n"
            "Generate one message so that it is unambiguous and easy to classify into TARGET_CLASS.\n\n"
            "Global rules:\n"
            "- Return only the candidate message text.\n"
            "- No JSON, no quotes, no markdown, no explanations.\n"
            "- Keep the message realistic and concise.\n"
            "- Avoid class ambiguity.\n"
            "- Follow the class-specific hard requirements exactly.\n"
        )

    def payload(self, spec: MessageSpec) -> str:
        vacancy = spec.cdm.get("vacancy") or {}
        candidate = spec.cdm.get("candidate") or {}
        noise_desc = ["low", "medium", "high"][min(max(spec.noise_level, 0), 2)]

        ctx = {
            "TARGET_CLASS": spec.target_class,
            "SCENARIO_HINT": spec.scenario_hint,
            "noise_level": noise_desc,
            "vacancy": {
                "title": vacancy.get("title"),
                "company_name": vacancy.get("company_name"),
                "company_description": vacancy.get("company_description") or vacancy.get("firm_description"),
                "responsibilities": vacancy.get("responsibilities"),
                "work_format": vacancy.get("work_format"),
                "location": vacancy.get("location"),
                "salary_range_from": vacancy.get("salary_range_from"),
                "salary_range_to": vacancy.get("salary_range_to"),
                "salary": vacancy.get("salary"),
                "stack": vacancy.get("vacancy_stack") or vacancy.get("stack"),
            },
            "candidate": {
                "candidate_name": candidate.get("candidate_name"),
                "candidate_job_list": candidate.get("candidate_job_list"),
                "candidate_skills": candidate.get("candidate_skills"),
            },
        }

        return (
            "CONTEXT_JSON:\n"
            f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
            "INSTRUCTIONS:\n"
            f"1) TARGET_CLASS = {spec.target_class}\n"
            f"2) SCENARIO_HINT = {spec.scenario_hint}\n"
            "3) Use vacancy context only to make the message realistic.\n"
            "4) Do not invent a different class than requested.\n"
            f"5) {class_requirements(spec.target_class)}\n"
            f"6) {class_examples(spec.target_class)}\n"
            "7) Return exactly one message in Russian.\n"
        )

    def parse(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("message generator returned empty message")
        if "\n" in text.strip():
            text = " ".join(x.strip() for x in text.splitlines() if x.strip()).strip()
        return text


def validate_candidate_message(target_class: str, message: str) -> Optional[str]:
    """Проверить, что сгенерированное сообщение соответствует классу. None == ок."""
    text = message.strip()
    if not text:
        return "generated empty message"

    has_decline = has_any_pattern(text, DECLINE_PATTERNS)
    has_reason = has_any_pattern(text, REASON_PATTERNS)
    has_acceptance = has_any_pattern(text, ACCEPTANCE_PATTERNS) or "?" in text
    has_human_needed = has_any_pattern(text, HUMAN_NEEDED_PATTERNS)

    if target_class == "reason_farewell":
        if not has_decline:
            return "reason_farewell message has no explicit refusal"
        if not has_reason:
            return "reason_farewell message has no clear reason"
        return None

    if target_class == "no_reason":
        if not has_decline:
            return "no_reason message has no explicit refusal"
        if has_reason:
            return "no_reason message leaks a reason"
        return None

    if target_class == "acceptance":
        if has_decline:
            return "acceptance message contains refusal markers"
        if has_human_needed:
            return "acceptance message contains human_needed markers"
        if not has_acceptance:
            return "acceptance message lacks clear interest or relevant vacancy question"
        return None

    if target_class == "human_needed":
        if has_decline:
            return "human_needed message looks like a refusal instead of escalation"
        if has_acceptance and not has_human_needed:
            return "human_needed message looks like a normal acceptance/clarification"
        if not has_human_needed:
            return "human_needed message lacks clear escalation markers"
        return None

    return None

"""Генератор профиля кандидата для теста sourcing_assistant БЕЗ backend.

Тест sourcing — это КОНТРАКТ вывода (массив 1:1 к requirements, форма {requirement,comment,passed}),
семантики нет. Backend нужен лишь чтобы НАЙТИ живых кандидатов. Для вариативной генерации мы генерим
профиль кандидата LLM-ом (а requirements засеваем из словаря) — backend-зависимость снимается полностью.
Профиль — в формате входа промпта (`{about, skills[], positions[]}`, как build_candidate_profile).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from qa_harness.core import safe_json_loads

from .base import Generator


@dataclass
class CandidateProfileSpec:
    domain: str
    requirements: List[str]   # засеянные требования — часть их профиль может покрывать (для разнообразия passed)
    noise_level: int = 1


class CandidateProfileGenerator(Generator):
    """Генерит реалистичный профиль кандидата JSON {about, skills[], positions[]} под домен."""

    def instruction(self, spec: CandidateProfileSpec) -> str:
        return (
            "Ты генерируешь реалистичный профиль IT-кандидата для российского рынка. Верни СТРОГО JSON-объект "
            "без markdown:\n"
            '{"about": str, "skills": [{"skill": str}], "positions": [{"name": str, "pos": str, '
            '"description": str, "rangeStr": str, "current": bool}]}\n'
            "Требования: всё на русском (названия технологий латиницей); about — 1-3 предложения о кандидате; "
            "skills — 4-8 навыков; positions — 1-2 места работы с описанием. Кандидат должен покрывать ЧАСТЬ "
            "перечисленных требований (не обязательно все) — это нормально."
        )

    def payload(self, spec: CandidateProfileSpec) -> str:
        import json
        noise = ["лаконично", "обычно", "подробно"][min(max(spec.noise_level, 0), 2)]
        ctx = {"domain": spec.domain, "requirements_for_reference": spec.requirements, "style": noise}
        return "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\nВерни только JSON-профиль кандидата:"

    def parse(self, text: str, spec: CandidateProfileSpec) -> Dict[str, Any]:
        data, _err = safe_json_loads(text or "", lenient=True)
        if not isinstance(data, dict):
            raise ValueError("profile generator did not return a JSON object")
        about = str(data.get("about") or "").strip()
        skills = data.get("skills")
        if not about:
            raise ValueError("profile has empty about")
        if not isinstance(skills, list) or not skills:
            raise ValueError("profile has no skills")
        positions = data.get("positions") if isinstance(data.get("positions"), list) else []
        return {"about": about, "skills": skills, "positions": positions}

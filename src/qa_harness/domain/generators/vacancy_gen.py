"""Генератор варьируемой IT-вакансии (входной контекст) для CDM-раннеров (first_touch и т.п.).

Producer Группы 3: вместо диалога генерим ВХОДНОЙ КОНТЕКСТ — реалистичную вакансию (название/компания/
описание/обязанности/стек/вилка). Раннер собирает из неё payload и выводит ожидания (для first_touch —
expected_facts из самого input, поэтому ground truth получается автоматически). Валидация в parse:
обязательные поля непусты — иначе тест бессмыслен.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from qa_harness.core import safe_json_loads

from .base import Generator

DOMAINS = (
    "backend (Python)", "frontend (React/TypeScript)", "DevOps/SRE", "data engineering",
    "mobile (iOS/Swift)", "QA automation", "ML engineering", "fullstack (Node/React)",
)
SENIORITIES = ("junior", "middle", "senior", "lead")
_REQUIRED = ("vacancy_name", "company_description", "vacancy_responsibilities")


@dataclass
class VacancySpec:
    domain_hint: str
    seniority_hint: str
    noise_level: int = 1


class VacancyGenerator(Generator):
    """Генерит реалистичную IT-вакансию (РФ-рынок) в JSON под domain/seniority."""

    def instruction(self, spec: VacancySpec) -> str:
        return (
            "Ты генерируешь реалистичную IT-вакансию для российского рынка. Верни СТРОГО JSON-объект "
            "без markdown и пояснений со следующими ключами:\n"
            '{"vacancy_name": str, "hiring_company_name": str, "company_description": str, '
            '"vacancy_responsibilities": str, "vacancy_stack": str, "salary_range": str}\n'
            "Требования: всё на русском; vacancy_name/company_description/vacancy_responsibilities — "
            "непустые и осмысленные; vacancy_stack — перечисление технологий; salary_range — вилка "
            "(напр. '200000-280000 руб') или пустая строка, если не указывается.\n"
            "ВАЖНО: company_description НЕ должен содержать НАЗВАНИЕ компании (оно только в hiring_company_name) "
            "— опиши сферу/продукт/команду без упоминания имени бренда."
        )

    def payload(self, spec: VacancySpec) -> str:
        noise = ["лаконично", "обычно", "подробно/с шумом"][min(max(spec.noise_level, 0), 2)]
        ctx = {"domain": spec.domain_hint, "seniority": spec.seniority_hint, "style": noise}
        return "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\nВерни только JSON-объект вакансии:"

    def parse(self, text: str, spec: VacancySpec) -> Dict[str, Any]:
        data, _err = safe_json_loads(text or "", lenient=True)
        if not isinstance(data, dict):
            raise ValueError("vacancy generator did not return a JSON object")
        out = {k: ("" if data.get(k) is None else str(data.get(k)).strip()) for k in
               ("vacancy_name", "hiring_company_name", "company_description",
                "vacancy_responsibilities", "vacancy_stack", "salary_range")}
        missing = [k for k in _REQUIRED if not out.get(k)]
        if missing:
            raise ValueError(f"vacancy missing required fields: {missing}")
        return out

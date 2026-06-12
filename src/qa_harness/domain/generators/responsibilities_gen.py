"""Seeded-генератор текста обязанностей с ИЗВЕСТНЫМИ терминами для теста responsibilities_parser.

Producer Группы 3: чтобы ground truth был детерминирован, генерацию ЗАСЕВАЕМ — задаём core-термины
(их парсер ОБЯЗАН извлечь → expect) и soft-навыки (их извлекать НЕ должен → forbid). Генератор пишет
текст вакансии, прямо упоминая core-технологии. Валидация в parse: все core-термины присутствуют в
тексте (иначе expect недостижим). Тот же приём «контролируем метку через засев», что у autofill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List

from .base import Generator

# Технологический словарь по доменам — источник core-терминов (засев).
TECH_VOCAB = {
    "backend": ["Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Kafka", "gRPC", "SQLAlchemy"],
    "frontend": ["React", "TypeScript", "Redux", "Webpack", "Next.js", "GraphQL"],
    "devops": ["Docker", "Kubernetes", "Terraform", "Ansible", "Prometheus", "GitLab CI", "Helm"],
    "data": ["Spark", "Airflow", "ClickHouse", "dbt", "Kafka", "Hadoop"],
    "ml": ["PyTorch", "TensorFlow", "scikit-learn", "MLflow", "pandas", "NumPy"],
}
# Личные качества — шум, который парсер НЕ должен брать в ключевые технические термины (forbid).
SOFT_NOISE = ["коммуникабельность", "ответственность", "работа в команде", "стрессоустойчивость", "инициативность"]


@dataclass
class ResponsibilitiesSpec:
    domain: str
    core_terms: List[str]          # технологии, которые парсер обязан извлечь (засев → expect)
    soft_terms: List[str] = field(default_factory=list)  # личные качества (засев → forbid)
    noise_level: int = 1


class ResponsibilitiesGenerator(Generator):
    """Генерит текст вакансии (обязанности/требования), прямо упоминая core-технологии."""

    def instruction(self, spec: ResponsibilitiesSpec) -> str:
        return (
            "Ты пишешь раздел вакансии «обязанности и требования» на русском (4-6 пунктов, обычный текст, "
            "без markdown и пояснений вне текста).\n"
            "ОБЯЗАТЕЛЬНО прямо и явно упомяни как требуемые технологии ВСЕ из списка core. Личные качества "
            "из списка soft упомяни вскользь (1 фраза). Не добавляй число технологий сверх разумного."
        )

    def payload(self, spec: ResponsibilitiesSpec) -> str:
        noise = ["лаконично", "обычно", "подробно"][min(max(spec.noise_level, 0), 2)]
        ctx = {"domain": spec.domain, "core": spec.core_terms, "soft": spec.soft_terms, "style": noise}
        return "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\nВерни только текст вакансии:"

    def parse(self, text: str, spec: ResponsibilitiesSpec) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("пустой текст вакансии")
        low = text.lower()
        missing = [t for t in spec.core_terms if t.lower() not in low]
        if missing:
            raise ValueError(f"в тексте отсутствуют core-термины: {missing}")
        return text

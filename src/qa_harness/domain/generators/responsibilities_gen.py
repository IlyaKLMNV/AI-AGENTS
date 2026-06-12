"""Seeded-генератор текста вакансии с ИЗВЕСТНЫМИ терминами для responsibilities_parser / one_line.

Чтобы ground truth был детерминирован, генерацию ЗАСЕВАЕМ и РАЗДЕЛЯЕМ секции:
- core_terms — ОБЯЗАТЕЛЬНЫЕ требования (промпт обязан вернуть как требования → expect);
- nice_to_have — «будет плюсом» (НЕ должны попасть в обязательные → forbid);
- conditions — условия (ДМС/график — НЕ требования → forbid);
- soft_terms — личные качества (НЕ технические требования → forbid).
Валидация в parse: все core-термины присутствуют в тексте (иначе expect недостижим). Приём «контролируем
метку через засев», как у autofill. nice_to_have/conditions опциональны (one_line переиспользует генератор).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List

from .base import Generator

# Технологический словарь по доменам — источник core/nice-to-have терминов (засев).
TECH_VOCAB = {
    "backend": ["Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Kafka", "gRPC", "SQLAlchemy"],
    "frontend": ["React", "TypeScript", "Redux", "Webpack", "Next.js", "GraphQL"],
    "devops": ["Docker", "Kubernetes", "Terraform", "Ansible", "Prometheus", "GitLab CI", "Helm"],
    "data": ["Spark", "Airflow", "ClickHouse", "dbt", "Kafka", "Hadoop"],
    "ml": ["PyTorch", "TensorFlow", "scikit-learn", "MLflow", "pandas", "NumPy"],
}
# Личные качества — НЕ технические требования (forbid).
SOFT_NOISE = ["коммуникабельность", "ответственность", "работа в команде", "стрессоустойчивость", "инициативность"]
# Условия найма — НЕ требования к кандидату (forbid).
CONDITIONS_NOISE = ["ДМС", "гибкий график", "удалённая работа", "корпоративный спорт", "обучение за счёт компании"]


@dataclass
class ResponsibilitiesSpec:
    domain: str
    core_terms: List[str]                              # обязательные требования (засев → expect)
    soft_terms: List[str] = field(default_factory=list)        # личные качества (засев → forbid)
    nice_to_have: List[str] = field(default_factory=list)      # «будет плюсом» (засев → forbid)
    conditions: List[str] = field(default_factory=list)        # условия найма (засев → forbid)
    noise_level: int = 1


class ResponsibilitiesGenerator(Generator):
    """Генерит текст вакансии с РАЗДЕЛЁННЫМИ секциями (обязательное / плюс / условия / soft)."""

    def instruction(self, spec: ResponsibilitiesSpec) -> str:
        return (
            "Ты пишешь текст вакансии на русском (обычный текст, без markdown и пояснений вне текста).\n"
            "ЯВНО раздели секции (используй заголовки внутри текста):\n"
            "- «Требования:» — перечисли ОБЯЗАТЕЛЬНЫЕ технологии из списка core (именно как обязательные).\n"
            "- «Будет плюсом:» — технологии из списка nice_to_have (если список непуст) как ЖЕЛАТЕЛЬНЫЕ, НЕ обязательные.\n"
            "- «Условия:» — пункты из списка conditions (если непуст).\n"
            "- Личные качества из soft упомяни вскользь.\n"
            "Не добавляй технологий сверх перечисленных."
        )

    def payload(self, spec: ResponsibilitiesSpec) -> str:
        noise = ["лаконично", "обычно", "подробно"][min(max(spec.noise_level, 0), 2)]
        ctx = {"domain": spec.domain, "core_обязательные": spec.core_terms,
               "nice_to_have_будет_плюсом": spec.nice_to_have, "conditions_условия": spec.conditions,
               "soft_личные_качества": spec.soft_terms, "style": noise}
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

"""Контролируемый генератор резюме для теста sourcing_assistant (известные positive/negative навыки).

Новая задача sourcing — консервативная 0/1-оценка, поэтому generate должен быть РАЗМЕЧЕН: задаём positive
навыки (должны быть в резюме → ожидаем passed=1) и negative (НЕ должны быть → ожидаем passed=0). Резюме
пишет LLM для натуральности, но после генерации валидируем: все must_mention присутствуют, ни одного
must_not_mention нет (иначе метки врут). decoys (смежные технологии) кладём в must_mention, но НЕ в
требования — проверяем, что смежное не засчитывается.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from .base import Generator


@dataclass
class ResumeSpec:
    domain: str
    must_mention: List[str]                      # positive-навыки + decoys (обязаны быть в резюме)
    must_not_mention: List[str] = field(default_factory=list)  # negative-навыки (НЕ должны быть в резюме)
    noise_level: int = 1


class ResumeGenerator(Generator):
    """Генерит текст резюме кандидата, упоминая must_mention и НЕ упоминая must_not_mention."""

    def instruction(self, spec: ResumeSpec) -> str:
        return (
            "Ты пишешь реалистичные ДАННЫЕ кандидата на русском (резюме + краткая анкета), обычный текст без markdown.\n"
            "ОБЯЗАТЕЛЬНО явно упомяни КАЖДЫЙ пункт из must_mention (навык, готовность, город, оборудование — как есть).\n"
            "КАТЕГОРИЧЕСКИ НЕ упоминай ничего из must_not_mention (ни прямо, ни намёком).\n"
            "Структура: 2-3 предложения про опыт + строка «Навыки: …» + при необходимости короткая «Анкета: …» "
            "(готовность/город/оборудование). Не выдумывай технологии сверх must_mention."
        )

    def payload(self, spec: ResumeSpec) -> str:
        noise = ["кратко", "обычно", "подробно"][min(max(spec.noise_level, 0), 2)]
        ctx = {"domain": spec.domain, "must_mention": spec.must_mention,
               "must_not_mention": spec.must_not_mention, "style": noise}
        return "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\nВерни только текст резюме:"

    def parse(self, text: str, spec: ResumeSpec) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("пустое резюме")
        low = text.lower()
        missing = [t for t in spec.must_mention if t.lower() not in low]
        if missing:
            raise ValueError(f"в резюме отсутствуют must_mention: {missing}")
        leaked = [t for t in spec.must_not_mention if t.lower() in low]
        if leaked:
            raise ValueError(f"в резюме просочились must_not_mention: {leaked}")
        return text

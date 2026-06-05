"""LLM-судья фактов для first_touch (перенос evaluate_message из легаси).

Судья (ModelClient — отдельная модель, НЕ промпт-под-тестом) получает expected_facts +
allowed_context_facts + сгенерированное сообщение и возвращает:
- facts_present: по каждому ожидаемому факту — упомянут ли он (true/false, с учётом перефразирования);
- hallucinated_facts: фактические утверждения про вакансию/условия, которых нет во входных фактах;
- question_present: есть ли вопросительный CTA.
Это первый LLM-судья в харнессе; для дешёвого/офлайн-прогона есть эвристика в checks.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from qa_harness.core import safe_json_loads

EVAL_INSTRUCTION = (
    "Ты строгий QA-ревьюер первого сообщения рекрутера.\n"
    "Даны:\n"
    "- expected_facts: факты вакансии (с чем сравниваем наличие фактов в тексте)\n"
    "- allowed_context_facts: допустимые контекстные факты (источник кандидата, причина контакта), "
    "их НЕ считать галлюцинациями\n"
    "- generated_message: текст сообщения\n\n"
    "Задача:\n"
    "1) facts_present: для каждого ключа из expected_facts определить true/false. "
    "true только если факт явно упомянут или ясно перефразирован.\n"
    "2) hallucinated_facts: список фактических утверждений про вакансию/условия, которых нет в "
    "expected_facts и allowed_context_facts. Не добавляй общие рекрутерские фразы "
    "('заинтересовал ваш опыт' и т.п.), приветствия и вежливые вводные. Сомневаешься — НЕ добавляй.\n"
    "3) question_present: есть ли в сообщении вопросительный CTA.\n\n"
    "Верни строго JSON: "
    '{"facts_present": {"company_name": true, ...}, "hallucinated_facts": ["..."], '
    '"question_present": true, "comment": "кратко на русском"}'
)


@dataclass
class FactVerdict:
    facts_present: Dict[str, bool]
    hallucinated_facts: List[str]
    question_present: bool
    comment: str


class FactJudge:
    """LLM-судья наличия/выдуманности фактов. model_client — qa_harness.core.llm_client.ModelClient."""

    def __init__(self, model_client: Any) -> None:
        self._client = model_client

    def evaluate(
        self,
        expected_facts: Dict[str, str],
        allowed_context_facts: Dict[str, str],
        message: str,
    ) -> Tuple[FactVerdict, Any]:
        payload = json.dumps(
            {
                "instruction": EVAL_INSTRUCTION,
                "expected_facts": expected_facts,
                "allowed_context_facts": allowed_context_facts,
                "generated_message": message,
            },
            ensure_ascii=False,
        )
        raw, usage = self._client.create(payload)
        # core.safe_json_loads возвращает (obj, err); lenient=True выдёргивает JSON из текста/фенсов
        data, _err = safe_json_loads(raw, lenient=True)

        present_raw = data.get("facts_present") if isinstance(data, dict) else None
        facts_present = {
            k: bool(present_raw.get(k)) if isinstance(present_raw, dict) else False
            for k in expected_facts
        }
        hallucinated: List[str] = []
        question_present = "?" in (message or "")
        comment = ""
        if isinstance(data, dict):
            h = data.get("hallucinated_facts") or []
            if not isinstance(h, list):
                h = [str(h)]
            hallucinated = [str(x).strip() for x in h if str(x).strip()]
            if "question_present" in data:
                question_present = bool(data.get("question_present"))
            comment = str(data.get("comment") or "").strip()
        else:
            comment = "judge_parse_error"

        return FactVerdict(facts_present, hallucinated, question_present, comment), usage

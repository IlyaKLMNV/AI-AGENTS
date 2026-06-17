"""LLM-судья мероприятия для first_touch_event (перенос evaluate_message).

Сверяет сгенерированное приглашение с фиксированным эталоном (reference.py): missing_facts (из
required_fact_keys), hallucinated_facts, forbidden_claims. Бросает ValueError, если судья вернул не-объект
(раннер трактует это как инфра-ошибку, а не как провал качества).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Tuple

from qa_harness.core import safe_json_loads

from .reference import EVAL_INSTRUCTION, EXPECTED_FACT_KEYS, FORBIDDEN_DETAILS, REFERENCE_FACTS


@dataclass
class EventVerdict:
    missing_facts: List[str]
    hallucinated_facts: List[str]
    forbidden_claims: List[str]
    comment: str


class EventJudge:
    def __init__(self, model_client: Any) -> None:
        self._client = model_client

    def evaluate(self, message: str) -> Tuple[EventVerdict, Any]:
        payload = json.dumps(
            {
                "instruction": EVAL_INSTRUCTION,
                "required_fact_keys": EXPECTED_FACT_KEYS,
                "reference_facts": REFERENCE_FACTS,
                "forbidden_details": FORBIDDEN_DETAILS,
                "message": message,
            },
            ensure_ascii=False,
        )
        raw, usage = self._client.create(payload)
        data, _err = safe_json_loads(raw, lenient=True)
        if not isinstance(data, dict):
            raise ValueError("event judge did not return a JSON object")

        def _read_list(key: str) -> List[str]:
            value = data.get(key) or []
            if not isinstance(value, list):
                value = [value]
            return [str(x).strip() for x in value if str(x).strip()]

        verdict = EventVerdict(
            missing_facts=[k for k in _read_list("missing_facts") if k in EXPECTED_FACT_KEYS],
            hallucinated_facts=_read_list("hallucinated_facts"),
            forbidden_claims=_read_list("forbidden_claims"),
            comment=str(data.get("comment") or "").strip(),
        )
        return verdict, usage

"""LLM-судья сценариев: соответствует ли поведение screening_assistant ожидаемому (expected_behavior).

Заменяет ~4000 строк hardcoded-эвристик легаси: судья (ModelClient) получает сценарий (название/описание/
ожидаемое поведение) + транскрипт разговора и решает passed + violations + comment. Бросает на не-объект
(раннер трактует как инфра).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from qa_harness.core import safe_json_loads

EVAL_INSTRUCTION = (
    "Ты строгий QA-ревьюер скрининг-ассистента рекрутинга. Дан СЦЕНАРИЙ (название, описание и ОЖИДАЕМОЕ "
    "поведение модели согласно промпту) и ТРАНСКРИПТ разговора ассистента с кандидатом.\n"
    "Задача: решить, отработал ли ассистент так, как описано в expected_behavior.\n"
    "ВАЖНО про завершение диалога: служебный токен END в реплику кандидату не выводится. Если ассистент "
    "завершил диалог, его реплика помечена маркером «⟦ДИАЛОГ ЗАВЕРШЁН⟧» в конце. Считай это выполнением "
    "требований вида «завершить диалог END» / «прекратить общение». Отсутствие маркера = диалог НЕ завершён.\n"
    "passed=true только если поведение ассистента соответствует ожидаемому по СУТИ. Стиль, тон, "
    "формулировки и порядок не важны. violations — кратко чем поведение разошлось с ожидаемым.\n"
    'Верни строго JSON: {"passed": true|false, "violations": ["..."], "comment": "кратко на русском"}'
)

END_MARKER = "⟦ДИАЛОГ ЗАВЕРШЁН⟧"


@dataclass
class ScenarioVerdict:
    passed: bool
    violations: List[str] = field(default_factory=list)
    comment: str = ""


class ScenarioJudge:
    def __init__(self, model_client: Any) -> None:
        self._client = model_client

    def evaluate(self, scenario: Any, transcript_text: str) -> Tuple[ScenarioVerdict, Any]:
        payload = json.dumps(
            {
                "instruction": EVAL_INSTRUCTION,
                "scenario": {
                    "name": scenario.name,
                    "description": scenario.description,
                    "expected_behavior": scenario.expected_behavior,
                },
                "transcript": transcript_text,
            },
            ensure_ascii=False,
        )
        raw, usage = self._client.create(payload)
        data, _err = safe_json_loads(raw, lenient=True)
        if not isinstance(data, dict):
            raise ValueError("scenario judge did not return a JSON object")
        violations_raw = data.get("violations")
        violations = (
            [str(x).strip() for x in violations_raw if str(x).strip()]
            if isinstance(violations_raw, list) else []
        )
        return ScenarioVerdict(bool(data.get("passed")), violations, str(data.get("comment") or "").strip()), usage

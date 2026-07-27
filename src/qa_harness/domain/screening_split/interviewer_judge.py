"""LLM-судья Интервьюера (слой B, семантическая часть).

Интервьюер получает внутреннюю `instruction` Аналитика и должен сформулировать РОВНО ОДНО
сообщение кандидату, передав ТОЛЬКО смысл инструкции: без добавленных своих вопросов, не
отвечая за кандидата, не раскрывая внутренних процессов/секретов, в тёплом деловом тоне.

Судит ходы, где говорил Интервьюер (next_action=ask): пары (instruction, message). Утечку
КОНКРЕТНОГО числа вилки/ссылки ловит детерминированный leak_scan (checks.py) — здесь семантика:
верность инструкции, отсутствие добавленных вопросов, реплик за кандидата, служебного текста.
Детерминированные проверки (leak) и этот судья вместе образуют слой B (ошибка Интервьюера).
"""

import json
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from qa_harness.core import safe_json_loads

EVAL_INSTRUCTION = (
    "Ты строгий QA-ревьюер РОЛИ «Интервьюер» в раздельном скрининге. Интервьюер — «рот»: он получает "
    "служебную ВНУТРЕННЮЮ инструкцию (что сказать) и должен сформулировать кандидату РОВНО ОДНО "
    "естественное сообщение, передав ТОЛЬКО смысл инструкции.\n"
    "Дан список ходов: на каждом — internal_instruction (что велено сказать) и message (что Интервьюер "
    "реально написал кандидату). Оцени КАЖДЫЙ ход и верни общий вердикт.\n"
    "НАРУШЕНИЯ (passed=false, если есть хоть одно существенное):\n"
    "- добавил СВОИ вопросы, которых нет в инструкции (светские, оценочные, лишние уточнения);\n"
    "- не выполнил инструкцию (проигнорировал заданный вопрос/факт);\n"
    "- сам ответил за кандидата («Я в Москве», «ориентируюсь на 250») или процитировал его текст;\n"
    "- раскрыл внутренние процессы/результаты проверки («вы нам подходите», «локация подходит»), или "
    "написал слово END / попрощался (завершение — не его задача);\n"
    "- вывел служебный контекст (строки «Ваше имя:», «Должность:») или технический формат (remote/office).\n"
    "НЕ нарушение: тёплый нейтральный тон, «Поняла, спасибо», переформулировка инструкции своими словами, "
    "ответ на несколько вопросов кандидата в одном сообщении (если это требовалось инструкцией).\n"
    "КРИТИЧЕСКИ ВАЖНО — ты оцениваешь ТОЛЬКО ВЕРНОСТЬ ПЕРЕДАЧИ инструкции, а НЕ её уместность. "
    "Какие вопросы задавать, когда и в каком порядке (зарплата, город, пауза, доп-вопросы) — решает "
    "Аналитик, и это уже в internal_instruction. Если Интервьюер точно (в т.ч. ДОСЛОВНО) передал то, что "
    "велено — это PASS, даже если тебе кажется, что вопрос «преждевременный» или инструкция неоптимальна. "
    "НЕ штрафуй за содержание/уместность самой инструкции — только за искажение, добавление или пропуск.\n"
    "violations — кратко, с номером хода. Верни строго JSON: "
    '{"passed": true|false, "violations": ["..."], "comment": "кратко на русском"}'
)


@dataclass
class InterviewerVerdict:
    passed: bool
    violations: List[str] = field(default_factory=list)
    comment: str = ""


class InterviewerJudge:
    def __init__(self, model_client: Any) -> None:
        self._client = model_client

    def evaluate(self, pairs: List[dict]) -> Tuple[InterviewerVerdict, Any]:
        """pairs = [{"turn": n, "instruction": str, "message": str}, ...] по ask-ходам."""
        payload = json.dumps({"instruction": EVAL_INSTRUCTION, "turns": pairs}, ensure_ascii=False)
        raw, usage = self._client.create(payload)
        data, _err = safe_json_loads(raw, lenient=True)
        if not isinstance(data, dict):
            raise ValueError("interviewer judge did not return a JSON object")
        violations_raw = data.get("violations")
        violations = ([str(x).strip() for x in violations_raw if str(x).strip()]
                      if isinstance(violations_raw, list) else [])
        return InterviewerVerdict(bool(data.get("passed")), violations, str(data.get("comment") or "").strip()), usage

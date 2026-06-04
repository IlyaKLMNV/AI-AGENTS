"""Учёт токенов OpenAI Responses API.

Единый источник usage-триады, продублированной в 10+ старых раннерах
(`_blank_usage`/`_extract_usage_numbers`/`_accumulate_usage`). Поведение перенесено
дословно ради парити со старым кодом: те же ключи, те же fallback-имена полей,
тот же вывод total = input + output, когда total отсутствует.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Имена полей токенов, которые встречаются в разных версиях SDK / в dict-usage.
_INPUT_FIELDS = ("input_tokens", "prompt_tokens", "input_token_count")
_OUTPUT_FIELDS = ("output_tokens", "completion_tokens", "output_token_count")
_TOTAL_FIELDS = ("total_tokens", "token_count")


def blank_usage() -> Dict[str, int]:
    """Пустой usage-bucket."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _first(source: Any, fields: Tuple[str, ...], *, is_dict: bool) -> Any:
    for name in fields:
        val = source.get(name) if is_dict else getattr(source, name, None)
        if val:
            return val
    return None


def extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    """Вернуть (input, output, total) из usage-объекта SDK или dict.

    Принимает None, dict или объект SDK. Если total отсутствует — считает его
    как input + output. Никогда не бросает.
    """
    if not usage:
        return 0, 0, 0

    is_dict = isinstance(usage, dict)
    it = _first(usage, _INPUT_FIELDS, is_dict=is_dict) or 0
    ot = _first(usage, _OUTPUT_FIELDS, is_dict=is_dict) or 0
    tt = _first(usage, _TOTAL_FIELDS, is_dict=is_dict)

    if tt is None:
        tt = (it or 0) + (ot or 0)

    return int(it or 0), int(ot or 0), int(tt or 0)


def accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    """Прибавить usage к bucket (мутирует bucket на месте)."""
    it, ot, tt = extract_usage_numbers(usage)
    bucket["input_tokens"] += it
    bucket["output_tokens"] += ot
    bucket["total_tokens"] += tt


def usage_total(bucket: Dict[str, int]) -> Dict[str, int]:
    """Нормализованный вид для отчёта: {input, output, total}."""
    return {
        "input": int(bucket.get("input_tokens", 0)),
        "output": int(bucket.get("output_tokens", 0)),
        "total": int(bucket.get("total_tokens", 0)),
    }

"""Единый контракт парсинга JSON из ответов LLM.

Объединяет две разошедшиеся реализации старого кода:
- `extractor_agent_runner.safe_json_loads` -> возвращает (obj, err);
- `first_touch_runner._safe_json_loads` -> бросает, но умеет выдёргивать JSON из текста.

Здесь:
- `safe_json_loads(s)` по умолчанию повторяет поведение extractor (строгий парс, (obj, err));
- `lenient=True` добавляет выдёргивание подстроки {…}/[…] (поведение first_touch);
- `expect_json_object()` — единый строгий гард: и харнесс, и прод-обёртки должны
  бросать одинаково на не-объекте (см. docs/REFACTOR_PLAN.md P0, риск jsonio).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple


def extract_json_substring(text: str) -> Optional[str]:
    """Выдернуть первый блок {…} или […] из текста (или None)."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    return None


def safe_json_loads(s: str, *, lenient: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """Распарсить JSON, не бросая. Возвращает (obj, None) или (None, error_str).

    lenient=False (по умолчанию) — строгий парс, дословно как у extractor_agent.
    lenient=True — при неудаче пытается выдернуть подстроку {…}/[…] и распарсить её.
    """
    try:
        return json.loads(s), None
    except Exception as e:  # noqa: BLE001 — намеренно широкий, как в исходниках
        if lenient:
            extracted = extract_json_substring(s or "")
            if extracted:
                try:
                    return json.loads(extracted), None
                except Exception as e2:  # noqa: BLE001
                    return None, str(e2)
        return None, str(e)


def expect_json_object(obj: Any, err: Optional[str] = None) -> dict:
    """Строгий гард: вернуть dict либо бросить ValueError/TypeError.

    Единая точка, чтобы харнесс и прод-обёртки реагировали на «не-объект» одинаково.
    """
    if err:
        raise ValueError(f"Failed to parse JSON: {err}")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object, got {type(obj).__name__}")
    return obj

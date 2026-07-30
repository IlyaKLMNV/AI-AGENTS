"""Step1: вызов OpenAI Responses API (через сырой HTTP — как в legacy extractor).

requests.post к /responses (а не SDK), потому что extractor так и ходил; это важно для
будущей кассетной сверки на транспортном уровне. Возвращает (text, usage, error).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

from qa_harness.core.jsonio import safe_json_loads


def parse_extractor_json(text: str) -> Tuple[Optional[dict], str]:
    """Распарсить вывод step1 строго, с явным различением «грязного» вывода.

    Возвращает (obj, status):
    - "ok"      — голый валидный JSON-объект (как и требует промпт);
    - "dirty"   — JSON удалось вытащить из обёртки/текста (промпт нарушил «только JSON»);
    - "invalid" — распарсить не удалось вовсе.
    """
    obj, err = safe_json_loads(text or "")  # строгий парс
    if err is None and isinstance(obj, dict):
        return obj, "ok"
    obj2, err2 = safe_json_loads(text or "", lenient=True)  # выдернуть подстроку {…}
    if err2 is None and isinstance(obj2, dict):
        return obj2, "dirty"
    return None, "invalid"


@dataclass
class PromptCfg:
    prompt_id: Optional[str]
    prompt_version: Optional[str]
    model: str


def extract_response_text(resp: Dict[str, Any]) -> str:
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"]
    out = resp.get("output")
    if isinstance(out, list):
        chunks = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        chunks.append(c["text"])
        if chunks:
            return "\n".join(chunks).strip()
    return ""


def _blank() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def call_openai_step1(
    api_key: str,
    prompt_cfg: PromptCfg,
    user_input: str,
    timeout_s: int = 60,
) -> Tuple[Optional[str], Dict[str, int], Optional[str]]:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if not prompt_cfg.prompt_id:
        return None, _blank(), "step1_prompt_id_missing"

    payload: Dict[str, Any] = {"model": prompt_cfg.model, "input": user_input, "prompt": {"id": prompt_cfg.prompt_id}}
    if prompt_cfg.prompt_version:
        payload["prompt"]["version"] = str(prompt_cfg.prompt_version)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except Exception as e:  # noqa: BLE001
        return None, _blank(), str(e)

    if r.status_code >= 400:
        return None, _blank(), f"openai_http_{r.status_code}:{r.text[:500]}"

    try:
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return None, _blank(), f"openai_bad_json:{e}"

    usage_raw = data.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("input_tokens") or 0),
        "output_tokens": int(usage_raw.get("output_tokens") or 0),
        "total_tokens": int(usage_raw.get("total_tokens") or 0),
    }
    text = extract_response_text(data)
    if not text:
        return None, usage, "openai_empty_output_text"
    return text.strip(), usage, None

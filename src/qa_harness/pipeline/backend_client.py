"""Step3: вызов backend /site/searchBool + классификация ошибок.

Перенос из extractor_agent_runner. 400 'Positions or skills or keys must be set'
классифицируется как insufficient_search_terms (не падение step3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

INSUFFICIENT_MSG_RE = re.compile(r"Positions or skills or keys must be set", re.IGNORECASE)
AUTH_MSG_RE = re.compile(r"wrong token|unauthorized|forbidden|invalid token", re.IGNORECASE)


@dataclass
class BackendCfg:
    base_url: str
    step3_path: str = "/site/searchBool"
    token_in_body: bool = False
    timeout_s: int = 30
    retries: int = 1
    require_count: bool = True


def classify_step3_error(status: int, body_text: str) -> str:
    if status == 400 and INSUFFICIENT_MSG_RE.search(body_text or ""):
        return "insufficient_search_terms"
    if status in (401, 403) or AUTH_MSG_RE.search(body_text or ""):
        return "auth_error"
    return "http_error"


def call_backend_search_bool(
    backend: BackendCfg,
    token: str,
    payload: Dict[str, Any],
) -> Tuple[str, int, int, Optional[int], Optional[str], Optional[Dict[str, Any]]]:
    """Вернуть (kind, status, attempts, count, error, json). kind: success|insufficient_search_terms|...."""
    url = backend.base_url.rstrip("/") + backend.step3_path
    attempts = 0
    last_status = 0
    last_kind = "http_error"
    last_err: Optional[str] = None
    last_json: Optional[Dict[str, Any]] = None

    req_payload = dict(payload)
    headers = {"Content-Type": "application/json"}
    if backend.token_in_body:
        req_payload["token"] = token
    else:
        headers["X-Auth-Token"] = token

    for _ in range(max(1, backend.retries + 1)):
        attempts += 1
        try:
            # allow_redirects=False: search API не должен уводить на страницу логина;
            # 3xx -> /auth значит «не авторизован», и это надо репортить честно, а не идти по
            # редиректу и потом падать в bad_json на HTML-странице логина.
            r = requests.post(url, headers=headers, json=req_payload, timeout=backend.timeout_s, allow_redirects=False)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            continue

        status = r.status_code
        last_status = status
        text = r.text or ""
        if 300 <= status < 400:
            loc = r.headers.get("Location") or r.headers.get("location") or ""
            return "auth_redirect", status, attempts, None, f"redirect_to:{loc}", None
        if status >= 400:
            kind = classify_step3_error(status, text)
            last_kind = kind
            if kind == "insufficient_search_terms":
                return "insufficient_search_terms", status, attempts, None, None, None
            last_err = text[:800]
            continue

        try:
            data = r.json()
            last_json = data if isinstance(data, dict) else None
        except Exception as e:  # noqa: BLE001
            return "bad_json", status, attempts, None, f"backend_bad_json:{e}", None

        if backend.require_count and (not isinstance(last_json, dict) or "count" not in last_json):
            return "missing_count", status, attempts, None, "backend_missing_count", last_json

        count = last_json.get("count") if isinstance(last_json, dict) else None
        return "success", status, attempts, (count if isinstance(count, int) else None), None, last_json

    return last_kind, last_status, attempts, None, last_err, last_json

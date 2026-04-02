from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI


ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "first_touch_event"

DEFAULT_PROMPT_ID = "pmpt_69ce8e02fc4081958ec897a73fa07961066f8c3e46cdc060"
DEFAULT_PROMPT_VERSION = "1"
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
DEFAULT_NAMES = [
    "Анна",
    "Мария",
    "Екатерина",
    "Алексей",
    "Илья",
    "Олег",
    "Наталья",
    "Дмитрий",
    "Сергей",
]
DEFAULT_MIN_UNIQUE_BODIES = 2
DEFAULT_MIN_UNIQUE_QUESTIONS = 2

EXPECTED_FACT_KEYS = [
    "event_vk_jt_go",
    "audience_backend_engineers",
    "date_4_april",
    "city_moscow",
    "venue_skylight",
    "program_optimization_cases",
    "program_architecture_tasks",
    "program_networking_party",
    "nikita_discussion",
    "nikita_role_vk_api_infra",
    "nikita_program_committee_golangconf",
    "registration_link_offer",
]

REFERENCE_FACTS = {
    "event_name": "VK JT Go",
    "event_type": "офлайн-митап",
    "audience": "бэкенд-инженеры",
    "date": "4 апреля",
    "city": "Москва",
    "venue": "офис Skylight",
    "program": [
        "разбор реальных кейсов оптимизации из практики компании",
        "решение архитектурных задач",
        "нетворкинг-вечеринка",
    ],
    "separate_block": {
        "topic": "обсуждение инженерных новостей",
        "speaker": "Никита Галушко",
        "speaker_role": "ведущий разработчик API-инфраструктуры ВКонтакте",
        "speaker_additional_role": "член программного комитета GolangConf",
    },
    "cta": "в конце должен быть короткий доброжелательный вопрос про ссылку на регистрацию",
}

FORBIDDEN_DETAILS = [
    "нельзя добавлять время мероприятия",
    "нельзя добавлять стоимость, бонусы, подарки или дополнительные активности",
    "нельзя писать, что автор сообщения сам будет на мероприятии или является его участником",
    "нельзя добавлять других спикеров, темы или секции, которых нет в эталоне",
    "нельзя делать выводы о кандидате, его опыте, интересах или специализации",
    "нельзя добавлять неподтвержденный день недели",
]


@dataclass
class JudgeResult:
    missing_facts: List[str]
    hallucinated_facts: List[str]
    forbidden_claims: List[str]
    comment: str


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def ensure_dirs(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input_token_count") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output_token_count") or 0
        total_tokens = usage.get("total_tokens") or usage.get("token_count")
    else:
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)

    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    input_tokens, output_tokens, total_tokens = _extract_usage_numbers(usage)
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens


def _normalize_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_for_signature(text: str) -> str:
    normalized = _normalize_text(text).lower()
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized, flags=re.UNICODE)
    return normalized.strip()


def _extract_json_substring(text: str) -> Optional[str]:
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


def _safe_json_loads(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty json text")
    try:
        return json.loads(raw)
    except Exception:
        extracted = _extract_json_substring(raw)
        if not extracted:
            raise
        return json.loads(extracted)


def _extract_numbers(text: str) -> List[int]:
    numbers: List[int] = []
    for raw in re.findall(r"\d+(?:[\s\u00A0]\d+)*", text or ""):
        cleaned = re.sub(r"[\s\u00A0]+", "", raw)
        try:
            numbers.append(int(cleaned))
        except ValueError:
            continue
    return numbers


def _extra_numbers(text: str) -> List[str]:
    allowed_numbers = {4}
    found = sorted(n for n in set(_extract_numbers(text)) if n not in allowed_numbers)
    return [f"extra_numbers:{found}"] if found else []


def _expected_greeting(candidate_name: str) -> str:
    name = (candidate_name or "").strip()
    if name:
        return f"{name}, здравствуйте!"
    return "Здравствуйте!"


def _greeting_ok(message: str, candidate_name: str) -> bool:
    text = _normalize_text(message)
    expected = _expected_greeting(candidate_name)
    return text.startswith(expected)


def _strip_greeting(message: str, candidate_name: str) -> str:
    text = _normalize_text(message)
    expected = _expected_greeting(candidate_name)
    if text.startswith(expected):
        return text[len(expected) :].lstrip()

    lines = [line.strip() for line in text.split("\n")]
    if lines and lines[0] == expected:
        return "\n".join(lines[1:]).strip()

    return text


def _last_question(message: str) -> str:
    text = re.sub(r"\s+", " ", _normalize_text(message).replace("\n", " "), flags=re.UNICODE).strip()
    if not text or not text.endswith("?"):
        return ""

    prefix = text[:-1]
    boundary = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
    if boundary >= 0:
        return text[boundary + 1 :].strip()
    return text


def _final_question_ok(message: str) -> bool:
    text = _normalize_text(message)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False

    last_line = lines[-1]
    if not last_line.endswith("?"):
        return False

    question = _last_question(text)
    if not question:
        return False

    word_count = len(re.findall(r"\w+", question, flags=re.UNICODE))
    if word_count > 14:
        return False

    lowered = question.lower()
    return bool(re.search(r"(ссыл|регистрац)", lowered))


def _judge_instruction() -> str:
    return (
        "Ты строгий QA-ревьюер короткого первого сообщения с приглашением на мероприятие.\n"
        "Проверь сообщение только относительно эталонных фактов ниже.\n"
        "Перефразировки допустимы. Если смысл сохранён, это НЕ ошибка.\n\n"
        "Нужно вернуть строго JSON-объект со схемой:\n"
        "{"
        '"missing_facts": ["..."],'
        '"hallucinated_facts": ["..."],'
        '"forbidden_claims": ["..."],'
        '"comment": "краткий комментарий на русском"'
        "}\n\n"
        "Правила оценки:\n"
        "1) missing_facts: какие обязательные смыслы отсутствуют. Используй только ключи из списка required_fact_keys.\n"
        "2) hallucinated_facts: фактические детали, которых нет в эталоне.\n"
        "3) forbidden_claims: только грубые нарушения, например время, стоимость, бонусы, новые спикеры,\n"
        "   утверждение, что автор сам будет на мероприятии, выводы о кандидате, неподтвержденный день недели.\n"
        "4) Не считай ошибкой изменения тона, длины фраз, порядка внутри близких смысловых блоков и формулировки CTA,\n"
        "   если смысл ссылки на регистрацию сохранён.\n"
    )


def _judge_payload(message: str) -> str:
    payload = {
        "instruction": _judge_instruction(),
        "required_fact_keys": EXPECTED_FACT_KEYS,
        "reference_facts": REFERENCE_FACTS,
        "forbidden_details": FORBIDDEN_DETAILS,
        "message": message,
    }
    return json.dumps(payload, ensure_ascii=False)


def evaluate_message(client: OpenAI, eval_model: str, message: str) -> Tuple[JudgeResult, Any]:
    resp = client.responses.create(
        model=eval_model,
        input=_judge_payload(message),
        text={"format": {"type": "text"}},
    )
    raw = _normalize_text(getattr(resp, "output_text", "") or "")
    data = _safe_json_loads(raw)

    if not isinstance(data, dict):
        raise ValueError("judge did not return JSON object")

    def _read_list(key: str) -> List[str]:
        value = data.get(key) or []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    result = JudgeResult(
        missing_facts=[item for item in _read_list("missing_facts") if item in EXPECTED_FACT_KEYS],
        hallucinated_facts=_read_list("hallucinated_facts"),
        forbidden_claims=_read_list("forbidden_claims"),
        comment=str(data.get("comment") or "").strip(),
    )
    return result, getattr(resp, "usage", None)


def _resolve_prompt_cfg(
    cfg: Dict[str, Any],
    prompt_id_override: Optional[str],
    prompt_version_override: Optional[str],
) -> Tuple[str, str]:
    block = cfg.get("first_touch_event_invite") if isinstance(cfg.get("first_touch_event_invite"), dict) else {}

    prompt_id = (
        (prompt_id_override or "").strip()
        or (os.getenv("FIRST_TOUCH_EVENT_PROMPT_ID") or "").strip()
        or (str(block.get("prompt_id")) if block.get("prompt_id") else "")
        or DEFAULT_PROMPT_ID
    )
    prompt_version = (
        (prompt_version_override or "").strip()
        or (os.getenv("FIRST_TOUCH_EVENT_PROMPT_VERSION") or "").strip()
        or (str(block.get("prompt_version")) if block.get("prompt_version") else "")
        or DEFAULT_PROMPT_VERSION
    )
    return prompt_id, prompt_version


def _resolve_names(raw_names: str, include_empty_name: bool) -> List[str]:
    names = [part.strip() for part in (raw_names or "").split(",") if part.strip()]
    if not names:
        names = list(DEFAULT_NAMES)
    if include_empty_name:
        return [""] + names
    return names


def _display_candidate_name(candidate_name: str) -> str:
    return candidate_name if candidate_name else "<EMPTY>"


def _duplicate_groups(cases: List[Dict[str, Any]], key_name: str, sample_name: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        if "result" not in case:
            continue
        signature = str((case.get("meta") or {}).get(key_name) or "").strip()
        if not signature:
            continue
        groups.setdefault(signature, []).append(case)

    items: List[Dict[str, Any]] = []
    for signature, grouped_cases in groups.items():
        if len(grouped_cases) <= 1:
            continue
        items.append(
            {
                "count": len(grouped_cases),
                "sample": str((grouped_cases[0].get("meta") or {}).get(sample_name) or ""),
                "case_ids": [str(case.get("case_id")) for case in grouped_cases],
                "candidate_names": [_display_candidate_name(str(case.get("candidate_name") or "")) for case in grouped_cases],
            }
        )
    items.sort(key=lambda item: (-int(item["count"]), str(item["sample"])))
    return items


def _status_digest(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed_count = 0
    failed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for case in cases:
        if "error" in case:
            errors.append(
                {
                    "case_id": case.get("case_id"),
                    "candidate_name": _display_candidate_name(str(case.get("candidate_name") or "")),
                    "error": case.get("error"),
                }
            )
            continue

        result = case.get("result") or {}
        if result.get("passed"):
            passed_count += 1
            continue

        failed.append(
            {
                "case_id": case.get("case_id"),
                "candidate_name": _display_candidate_name(str(case.get("candidate_name") or "")),
                "fail_reasons": result.get("fail_reasons") or [],
            }
        )

    return {
        "passed_count": passed_count,
        "failed_count": len(failed),
        "error_count": len(errors),
        "failed": failed,
        "errors": errors,
    }


def _compute_summary(
    cases: List[Dict[str, Any]],
    min_unique_bodies: int,
    min_unique_questions: int,
) -> Dict[str, Any]:
    success_cases = [case for case in cases if "result" in case]
    total = len(success_cases)

    passed_count = sum(1 for case in success_cases if case.get("result", {}).get("passed"))
    greeting_failures = sum(1 for case in success_cases if not case.get("result", {}).get("greeting_ok"))
    final_question_failures = sum(1 for case in success_cases if not case.get("result", {}).get("final_question_ok"))
    hallucination_cases = sum(1 for case in success_cases if case.get("result", {}).get("hallucinated_facts"))
    missing_facts_cases = sum(1 for case in success_cases if case.get("result", {}).get("missing_facts"))
    forbidden_claim_cases = sum(1 for case in success_cases if case.get("result", {}).get("forbidden_claims"))
    extra_number_cases = sum(1 for case in success_cases if case.get("result", {}).get("extra_numbers"))

    full_signatures = {
        str((case.get("meta") or {}).get("full_signature") or "")
        for case in success_cases
        if str((case.get("meta") or {}).get("full_signature") or "")
    }
    body_signatures = {
        str((case.get("meta") or {}).get("body_signature") or "")
        for case in success_cases
        if str((case.get("meta") or {}).get("body_signature") or "")
    }
    question_signatures = {
        str((case.get("meta") or {}).get("last_question_signature") or "")
        for case in success_cases
        if str((case.get("meta") or {}).get("last_question_signature") or "")
    }

    unique_body_messages = len(body_signatures)
    unique_last_questions = len(question_signatures)
    variability_passed = unique_body_messages >= min_unique_bodies and unique_last_questions >= min_unique_questions

    return {
        "total_cases": total,
        "pass_rate": round((passed_count / total) if total else 0.0, 4),
        "passed_cases": passed_count,
        "failed_cases": total - passed_count,
        "issue_counts": {
            "bad_greeting": greeting_failures,
            "bad_final_question": final_question_failures,
            "missing_facts": missing_facts_cases,
            "hallucinations": hallucination_cases,
            "forbidden_claims": forbidden_claim_cases,
            "extra_numbers": extra_number_cases,
        },
        "variability": {
            "passed": variability_passed,
            "unique_full_messages": len(full_signatures),
            "unique_body_messages": unique_body_messages,
            "unique_last_questions": unique_last_questions,
            "min_unique_bodies_required": min_unique_bodies,
            "min_unique_questions_required": min_unique_questions,
            "duplicate_body_groups": _duplicate_groups(
                success_cases,
                key_name="body_signature",
                sample_name="body_without_greeting",
            ),
            "duplicate_question_groups": _duplicate_groups(
                success_cases,
                key_name="last_question_signature",
                sample_name="last_question",
            ),
        },
    }


def _report_definitions(min_unique_bodies: int, min_unique_questions: int) -> Dict[str, Any]:
    return {
        "scope": [
            "приглашение на офлайн-митап VK JT Go для бэкенд-инженеров",
            "4 апреля, Москва, офис Skylight",
            "в программе: кейсы оптимизации, архитектурные задачи, нетворкинг-вечеринка",
            "отдельно: инженерные новости с Никитой Галушко",
            "последняя фраза: короткий вопрос про ссылку на регистрацию",
        ],
        "pass_criteria": [
            "greeting_ok=true",
            "final_question_ok=true",
            "missing_facts is empty",
            "hallucinated_facts is empty",
            "forbidden_claims is empty",
            "extra_numbers is empty",
        ],
        "forbidden_examples": [
            "время мероприятия",
            "стоимость, бонусы, подарки, лишние активности",
            "новые спикеры или темы",
            "выводы о кандидате",
            "неподтвержденный день недели",
        ],
        "variability_rules": {
            "body_comparison": "сравнивается текст без приветствия",
            "min_unique_bodies_required": min_unique_bodies,
            "min_unique_questions_required": min_unique_questions,
        },
    }


def _compact_issues(result: Dict[str, Any]) -> Dict[str, Any]:
    issues: Dict[str, Any] = {}
    for key in ("fail_reasons", "missing_facts", "hallucinated_facts", "forbidden_claims", "extra_numbers"):
        value = result.get(key) or []
        if value:
            issues[key] = value
    return issues


def _compact_case(case: Dict[str, Any]) -> Dict[str, Any]:
    candidate_name = str(case.get("candidate_name") or "")
    compact: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "candidate_name": _display_candidate_name(candidate_name),
        "repeat_index": case.get("repeat_index"),
    }

    if "error" in case:
        compact["passed"] = False
        compact["error"] = case.get("error")
        return compact

    result = case.get("result") or {}
    meta = case.get("meta") or {}

    compact.update(
        {
            "passed": bool(result.get("passed")),
            "message": case.get("message") or "",
            "last_question": meta.get("last_question") or "",
            "checks": {
                "greeting_ok": bool(result.get("greeting_ok")),
                "final_question_ok": bool(result.get("final_question_ok")),
            },
            "judge_comment": meta.get("judge_comment") or "",
        }
    )

    issues = _compact_issues(result)
    if issues:
        compact["issues"] = issues

    return compact


def run_first_touch_event_suite(
    prompt_id: str,
    prompt_version: str,
    eval_model: str,
    names: List[str],
    repeats_per_name: int,
    min_unique_bodies: int,
    min_unique_questions: int,
    out_dir: pathlib.Path,
    quiet: bool,
) -> pathlib.Path:
    _load_dotenv(ROOT / ".env")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set and was not found in .env")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    client = OpenAI(api_key=api_key)
    prompt: Dict[str, Any] = {"id": prompt_id}
    if prompt_version:
        prompt["version"] = str(prompt_version)

    usage = {
        "generation": _blank_usage(),
        "evaluation": _blank_usage(),
        "total": _blank_usage(),
    }

    cases: List[Dict[str, Any]] = []
    total_cases = len(names) * repeats_per_name
    current_case = 0

    _log(
        quiet,
        "[init] "
        f"prompt_id={prompt_id} prompt_version={prompt_version} eval_model={eval_model} "
        f"names={len(names)} repeats_per_name={repeats_per_name}",
    )

    for candidate_name in names:
        for repeat_index in range(1, repeats_per_name + 1):
            current_case += 1
            case_id = f"case_{current_case:03d}"
            display_name = _display_candidate_name(candidate_name)
            _log(quiet, f"[run] {case_id}/{total_cases:03d} candidate_name={display_name!r} repeat={repeat_index}")

            try:
                payload = {"candidate_name": candidate_name}
                generation_response = client.responses.create(
                    prompt=prompt,
                    input=json.dumps(payload, ensure_ascii=False),
                    text={"format": {"type": "text"}},
                )
                _accumulate_usage(usage["generation"], getattr(generation_response, "usage", None))

                message = _normalize_text(getattr(generation_response, "output_text", "") or "")
                if not message:
                    raise ValueError("prompt returned empty message")

                judge_result, judge_usage = evaluate_message(client=client, eval_model=eval_model, message=message)
                _accumulate_usage(usage["evaluation"], judge_usage)

                greeting_ok = _greeting_ok(message, candidate_name)
                final_question_ok = _final_question_ok(message)
                extra_numbers = _extra_numbers(message)

                fail_reasons: List[str] = []
                if not greeting_ok:
                    fail_reasons.append("bad_greeting")
                if not final_question_ok:
                    fail_reasons.append("bad_final_question")
                if judge_result.missing_facts:
                    fail_reasons.append("missing_facts")
                if judge_result.hallucinated_facts:
                    fail_reasons.append("hallucinated_facts")
                if judge_result.forbidden_claims:
                    fail_reasons.append("forbidden_claims")
                if extra_numbers:
                    fail_reasons.append("extra_numbers")

                body_without_greeting = _strip_greeting(message, candidate_name)
                last_question = _last_question(message)

                cases.append(
                    {
                        "case_id": case_id,
                        "candidate_name": candidate_name,
                        "repeat_index": repeat_index,
                        "message": message,
                        "result": {
                            "passed": not fail_reasons,
                            "greeting_ok": greeting_ok,
                            "final_question_ok": final_question_ok,
                            "missing_facts": judge_result.missing_facts,
                            "hallucinated_facts": judge_result.hallucinated_facts,
                            "forbidden_claims": judge_result.forbidden_claims,
                            "extra_numbers": extra_numbers,
                            "fail_reasons": fail_reasons,
                        },
                        "meta": {
                            "judge_comment": judge_result.comment,
                            "full_signature": _normalize_for_signature(message),
                            "body_signature": _normalize_for_signature(body_without_greeting),
                            "last_question": last_question,
                            "last_question_signature": _normalize_for_signature(last_question),
                            "body_without_greeting": body_without_greeting,
                        },
                    }
                )
            except Exception as exc:
                cases.append(
                    {
                        "case_id": case_id,
                        "candidate_name": candidate_name,
                        "repeat_index": repeat_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                _log(quiet, f"[warn] {case_id} failed: {type(exc).__name__}: {exc}")

    _accumulate_usage(usage["total"], usage["generation"])
    _accumulate_usage(usage["total"], usage["evaluation"])

    finished_at = datetime.datetime.now()
    summary = _compute_summary(
        cases=cases,
        min_unique_bodies=min_unique_bodies,
        min_unique_questions=min_unique_questions,
    )

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "prompt": {
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
        },
        "eval_model": eval_model,
        "definitions": _report_definitions(
            min_unique_bodies=min_unique_bodies,
            min_unique_questions=min_unique_questions,
        ),
        "usage": usage,
        "status": _status_digest(cases),
        "summary": summary,
        "cases": [_compact_case(case) for case in cases],
    }

    ensure_dirs(out_dir)
    out_path = out_dir / f"first_touch_event_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(quiet, f"[done] report saved: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple runner for event-invite first-touch prompt with variability and hallucination checks."
    )
    parser.add_argument("--prompt-id", type=str, default="", help="Override prompt_id")
    parser.add_argument("--prompt-version", type=str, default="", help="Override prompt_version")
    parser.add_argument("--eval-model", type=str, default=DEFAULT_EVAL_MODEL, help="Judge model")
    parser.add_argument(
        "--names",
        type=str,
        default="",
        help="Comma-separated candidate names. Defaults to an internal pool of Russian names.",
    )
    parser.add_argument(
        "--repeats-per-name",
        type=int,
        default=1,
        help="How many generations to request for each candidate_name.",
    )
    parser.add_argument(
        "--include-empty-name",
        dest="include_empty_name",
        action="store_true",
        help="Include one case with empty candidate_name (default).",
    )
    parser.add_argument(
        "--no-include-empty-name",
        dest="include_empty_name",
        action="store_false",
        help="Do not include the empty candidate_name case.",
    )
    parser.set_defaults(include_empty_name=True)
    parser.add_argument(
        "--min-unique-bodies",
        type=int,
        default=DEFAULT_MIN_UNIQUE_BODIES,
        help="Minimum unique message bodies required for variability_passed.",
    )
    parser.add_argument(
        "--min-unique-questions",
        type=int,
        default=DEFAULT_MIN_UNIQUE_QUESTIONS,
        help="Minimum unique final registration questions required for variability_passed.",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPORTS_DIR, help="Output directory for report JSON")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.repeats_per_name <= 0:
        raise SystemExit("--repeats-per-name must be > 0")
    if args.min_unique_bodies < 0:
        raise SystemExit("--min-unique-bodies must be >= 0")
    if args.min_unique_questions < 0:
        raise SystemExit("--min-unique-questions must be >= 0")

    _load_dotenv(ROOT / ".env")
    cfg = load_yaml(CFG_PATH) if CFG_PATH.exists() else {}
    prompt_id, prompt_version = _resolve_prompt_cfg(
        cfg=cfg or {},
        prompt_id_override=args.prompt_id,
        prompt_version_override=args.prompt_version,
    )
    names = _resolve_names(raw_names=args.names, include_empty_name=bool(args.include_empty_name))

    run_first_touch_event_suite(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        eval_model=args.eval_model,
        names=names,
        repeats_per_name=int(args.repeats_per_name),
        min_unique_bodies=int(args.min_unique_bodies),
        min_unique_questions=int(args.min_unique_questions),
        out_dir=args.out_dir,
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()

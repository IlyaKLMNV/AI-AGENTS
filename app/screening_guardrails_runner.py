from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI
from screeningAssistant.screeningAss import (
    AssistantError,
    Assistants as ScreeningAssistants,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "screening_guardrails"

GEN_MODEL = "gpt-4.1-mini"
EVAL_MODEL = "gpt-4.1"

DEFAULT_CONVERSATIONS = 50
DEFAULT_TURNS_PER_CONVERSATION = 4

DEFAULT_VACANCY_INFO: Dict[str, Any] = {
    "title": "Python Backend Developer",
    "company_name": "ExampleSoft",
    "responsibilities": "Поддержка и развитие микросервисов, интеграции с продуктами.",
    "work_format": "remote",
    "location": "Москва",
    "min_salary": 200000,
    "max_salary": 280000,
    "company_info": {
        "firm_description": "Продуктовая команда, развивающая b2b-платформу.",
        "vacancy_url": "https://example.com/vacancies/python-backend",
    },
    "questions": (
        "- Расскажите про опыт с Python и современными фреймворками?\n"
        "- Какой уровень дохода и формат работы рассматриваете?\n"
        "- Какие сервисы поддерживали под высокой нагрузкой?\n"
        "- Как часто используете SQL и для каких задач?"
    ),
}


# ---------------- utils ----------------

def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_token_count")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_token_count")
            or 0
        )
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


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    inp, out, total = _extract_usage_numbers(usage)
    bucket["input_tokens"] += inp
    bucket["output_tokens"] += out
    bucket["total_tokens"] += total


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("END", "")
    return s.strip()


def _extract_json_substring(text: str) -> str | None:
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
    text = (text or "").strip()
    if not text:
        raise ValueError("empty json text")

    try:
        return json.loads(text)
    except Exception:
        extracted = _extract_json_substring(text)
        if not extracted:
            raise
        return json.loads(extracted)


# ---------------- heuristics fallback ----------------

STOP_WORDS = {
    "и", "а", "но", "или", "ли", "же", "то", "это", "в", "на", "по", "за", "к", "с", "у", "о",
    "вы", "вас", "вам", "мы", "я", "он", "она", "они", "оно", "что", "как", "где", "когда",
    "пожалуйста", "спасибо", "поняла", "понял", "отлично", "хотела", "хотел", "уточнить",
}


def _tokenize_for_similarity(text: str) -> List[str]:
    t = _normalize_text(text).lower()
    t = re.sub(r"[^a-zа-я0-9\s]", " ", t, flags=re.IGNORECASE)
    parts = [p for p in t.split() if p and p not in STOP_WORDS]
    parts = [p for p in parts if len(p) >= 3]
    return parts


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _extract_question_segments(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []

    chunks: List[str] = []
    parts = t.split("?")
    for p in parts[:-1]:
        seg = (p.strip() + "?").strip()
        if seg:
            chunks.append(seg)

    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "?" in line:
            continue
        if re.search(
            r"\b(подскажите|уточните|расскажите|могли бы|можете|какой|какая|какие|сколько|где|готовы ли)\b",
            line,
            re.I,
        ):
            chunks.append(line)

    uniq: List[str] = []
    seen = set()
    for c in chunks:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


TOPIC_KEYWORDS = {
    "location": ["город", "локац", "где вы", "где сейчас", "находит", "прожив", "москва", "питер", "спб"],
    "salary": ["зарплат", "вилка", "оклад", "доход", "компенсац", "ожидан", "сумм", "руб", "тыс", "₽"],
    "relocation": ["переезд", "переехать", "готовы ли", "готовность", "релокац", "офис", "гибрид"],
}


def _keyword_topic_hits(text: str) -> Dict[str, int]:
    t = _normalize_text(text).lower()
    hits: Dict[str, int] = {k: 0 for k in TOPIC_KEYWORDS}
    for topic, keys in TOPIC_KEYWORDS.items():
        for kw in keys:
            if kw in t:
                hits[topic] += 1
    return hits


def heuristic_repeated_questions(text: str) -> Tuple[bool, str, List[str]]:
    segments = _extract_question_segments(text)
    if len(segments) <= 1:
        return False, "heuristic: <=1 question segment", []

    toks = [_tokenize_for_similarity(s) for s in segments]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            sim = _jaccard(toks[i], toks[j])
            if sim >= 0.72:
                return True, f"heuristic: similar questions (jaccard={sim:.2f})", ["other"]

    topic_counts = {k: 0 for k in TOPIC_KEYWORDS}
    for seg in segments:
        hits = _keyword_topic_hits(seg)
        for topic, v in hits.items():
            if v > 0:
                topic_counts[topic] += 1

    repeated_topics = [t for t, c in topic_counts.items() if c >= 2]
    if repeated_topics:
        return True, f"heuristic: repeated topic(s): {', '.join(repeated_topics)}", repeated_topics

    return False, "heuristic: no clear repetition", []


def heuristic_self_answer(text: str) -> Tuple[bool, str]:
    t = _normalize_text(text)
    low = t.lower()
    if "кандидат:" in low or "candidate:" in low:
        return True, "heuristic: role label candidate:"
    if "?" in t:
        after_q = t.split("?", 1)[-1].strip()
        if re.search(r"^(да|нет|ага|конечно|у меня)\b", after_q, re.I):
            return True, "heuristic: answer-like text after question"
    return False, "heuristic: no self-answer signal"


# ---------------- candidate multi-turn generation ----------------

def _seed_pool() -> List[str]:
    return [
        "Добрый день\n230_000₽\nМосква",
        "Москва, 240 000 на руки. Готов ответить на вопросы по опыту.",
        "Город Москва. Ожидания 250 000. Давайте пройдемся по вопросам.",
        "Москва. 220k net. Могу рассказать про последний проект.",
        "Москва, 260_000₽. Готов обсудить стек и опыт.",
        "Опыт около года в ML, до этого учился и делал пет-проекты.",
        "Python использую ежедневно. SQL тоже часто: джойны и оконные функции.",
        "По нагрузке: были сервисы 100 RPS и 2-3k RPS, использовали кэш, очереди, мониторинг.",
        "Могу кратко описать последний проект: API, интеграции, очереди, мониторинг.",
        "Если нужно, отвечу по пунктам и кратко.",
    ]


def _generation_prompt_multi_turn(conversations: int, turns_per_conversation: int) -> str:
    seed = _seed_pool()
    lines = [
        f"Сгенерируй {conversations} независимых диалогов кандидата (multi-turn).",
        f"В каждом диалоге должно быть ровно {turns_per_conversation} сообщений кандидата.",
        "Каждый диалог это массив строк. Итоговый ответ это JSON массив диалогов.",
        "Цель: кандидатскими ответами довести рекрутера до задавания вопросов по квалификации.",
        "Ограничения:",
        "- Не пиши про политику, национальности, оскорбления.",
        "- Не обвиняй в спаме и не проси удалить контакт.",
        "- Чаще указывай Москва и 200-280k.",
        "Примеры фраз кандидата (используй как стиль, не копируй дословно):",
    ]
    for i, s in enumerate(seed, start=1):
        lines.append(f"{i}) {s}")
    lines.append('Верни только JSON формата: [["msg1","msg2",...],[...]] без пояснений и без markdown.')
    return "\n".join(lines)


def _fallback_conversations(conversations: int, turns_per_conversation: int) -> List[List[str]]:
    rnd = random.Random(42)
    pool = _seed_pool()

    dialogs: List[List[str]] = []
    for _ in range(conversations):
        first = rnd.choice([
            "Добрый день\n230_000₽\nМосква",
            "Москва, 240_000₽ на руки",
            "Г. Москва, 250 000 net",
        ])
        turns = [_normalize_text(first)]
        while len(turns) < turns_per_conversation:
            msg = rnd.choice(pool)
            if msg in turns:
                continue
            turns.append(_normalize_text(msg))
        dialogs.append(turns)
    return dialogs


def generate_candidate_conversations(
    client: OpenAI,
    conversations: int,
    turns_per_conversation: int,
    usage_bucket: Dict[str, int],
) -> List[List[str]]:
    prompt = _generation_prompt_multi_turn(conversations, turns_per_conversation)

    dialogs: List[List[str]] = []
    try:
        response = client.responses.create(model=GEN_MODEL, input=prompt)
        _accumulate_usage(usage_bucket, getattr(response, "usage", None))
        text = _normalize_text(getattr(response, "output_text", "") or "")
        data = _safe_json_loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    turns = [_normalize_text(str(x)) for x in item if str(x).strip()]
                    if len(turns) == turns_per_conversation:
                        dialogs.append(turns)
    except Exception:
        dialogs = []

    if len(dialogs) < conversations:
        fb = _fallback_conversations(conversations, turns_per_conversation)
        dialogs.extend(fb[len(dialogs):])

    return dialogs[:conversations]


# ---------------- assistant runner (multi-turn) ----------------

def create_screening_assistant(cfg: Dict[str, Any]) -> ScreeningAssistants:
    sa_cfg = _component_cfg(cfg, "screening_assistant")
    prompt_id = sa_cfg.get("prompt_id")
    prompt_version = sa_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError("screening_assistant.prompt_id is not set in model.yaml")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    return ScreeningAssistants(
        api_key=api_key,
        vacancy_info=DEFAULT_VACANCY_INFO,
        recruiter_name="Майя",
        candidate_name="Алексей",
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )


def run_assistant_multi_turn(
    assistant: ScreeningAssistants,
    candidate_turns: List[str],
) -> List[Dict[str, Any]]:
    conversation_id = assistant.create_thread()
    turns_out: List[Dict[str, Any]] = []

    for idx, cand_msg in enumerate(candidate_turns, start=1):
        result = assistant.add_message_and_run(conversation_id, cand_msg)
        if result is None:
            raise AssistantError("assistant returned no result")

        reply_text = _normalize_text(result.response or "")
        turns_out.append(
            {
                "turn_index": idx,
                "candidate_message": _normalize_text(cand_msg),
                "assistant_reply": reply_text,
                "assistant_conversation_end": bool(getattr(result, "conversation_end", False)),
                "usage": getattr(result, "usage", None),
            }
        )

        if getattr(result, "conversation_end", False):
            break

    return turns_out


# ---------------- evaluation ----------------

def _eval_instruction() -> str:
    return (
        "Ты проверяешь одно сообщение ассистента (рекрутера) на два нарушения:\n"
        "\n"
        "A) self_answer = true, если ассистент в своем сообщении пишет за кандидата или продолжает диалог за обе стороны.\n"
        "Примеры:\n"
        "- после вопросов вставляет ответ кандидата ('Да, был опыт...', 'Python использую ежедневно...')\n"
        "- использует метки ролей ('Кандидат:', 'Candidate:')\n"
        "- явно отвечает вместо кандидата на свой вопрос.\n"
        "\n"
        "B) repeated_questions = true, если ассистент в одном сообщении повторяет по смыслу один и тот же вопрос.\n"
        "Это не про 'много разных вопросов'. Это про повтор одного и того же:\n"
        "- спрашивает про город/локацию несколько раз разными формулировками,\n"
        "- спрашивает про зарплату несколько раз разными формулировками,\n"
        "- спрашивает про переезд/готовность несколько раз,\n"
        "- или повторяет один и тот же вопрос по опыту (например, прогноз спроса/рекомендации) 2-3 раза.\n"
        "\n"
        "Важно:\n"
        "- 2-3 РАЗНЫХ вопроса (город + зарплата + стек) это нормально.\n"
        "- Нарушение только если есть повтор одного и того же вопроса по смыслу.\n"
        "\n"
        "Верни строго JSON объекта:\n"
        '{'
        '"self_answer": true|false, '
        '"repeated_questions": true|false, '
        '"repeated_topics": ["location"|"salary"|"relocation"|"experience"|"other"], '
        '"comment": "кратко почему"'
        '}'
    )


def _eval_payload(candidate_message: str, assistant_reply: str) -> str:
    payload = {
        "instruction": _eval_instruction(),
        "candidate_message": candidate_message,
        "assistant_reply": assistant_reply,
    }
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class EvalResult:
    self_answer: bool
    repeated_questions: bool
    repeated_topics: List[str]
    comment: str
    used_heuristics: bool


def evaluate_reply(
    client: OpenAI,
    candidate_message: str,
    assistant_reply: str,
    usage_bucket: Dict[str, int],
) -> EvalResult:
    payload = _eval_payload(candidate_message, assistant_reply)

    raw = ""
    try:
        response = client.responses.create(model=EVAL_MODEL, input=payload)
        _accumulate_usage(usage_bucket, getattr(response, "usage", None))
        raw = _normalize_text(getattr(response, "output_text", "") or "")
        data = _safe_json_loads(raw)

        self_answer = bool(data.get("self_answer"))
        repeated_questions = bool(data.get("repeated_questions"))
        repeated_topics = data.get("repeated_topics") or []
        if not isinstance(repeated_topics, list):
            repeated_topics = []
        repeated_topics = [str(x) for x in repeated_topics][:10]
        comment = str(data.get("comment") or "").strip()

        return EvalResult(
            self_answer=self_answer,
            repeated_questions=repeated_questions,
            repeated_topics=repeated_topics,
            comment=comment,
            used_heuristics=False,
        )
    except Exception:
        sa, sa_reason = heuristic_self_answer(assistant_reply)
        rq, rq_reason, topics = heuristic_repeated_questions(assistant_reply)
        comment = f"Eval parse failed, heuristics used. self_answer={sa_reason}; repeated={rq_reason}; raw={raw[:120]}"
        return EvalResult(
            self_answer=sa,
            repeated_questions=rq,
            repeated_topics=topics or (["other"] if rq else []),
            comment=comment,
            used_heuristics=True,
        )


# ---------------- reporting ----------------

def _build_conversation_level_summary(full_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summary на уровне ДИАЛОГОВ, как ты просил.
    """
    conversations = full_report.get("conversations", [])
    total = len(conversations)

    conv_self = 0
    conv_rep = 0
    conv_any = 0

    for conv in conversations:
        has_self = False
        has_rep = False
        for t in conv.get("turns", []):
            flags = t.get("flags") or {}
            if flags.get("self_answer"):
                has_self = True
            if flags.get("repeated_questions"):
                has_rep = True
        if has_self:
            conv_self += 1
        if has_rep:
            conv_rep += 1
        if has_self or has_rep:
            conv_any += 1

    pct = (conv_any / total * 100.0) if total else 0.0

    return {
        "conversations_total": total,
        "conversations_with_self_answer": conv_self,
        "conversations_with_repeated_questions": conv_rep,
        "conversations_with_any_violation": conv_any,
        "problem_conversations_pct": pct,
    }


def _make_console_summary(summary: Dict[str, Any]) -> str:
    return (
        "=== GUARDRAILS SUMMARY (conversation-level) ===\n"
        f"conversations_total: {summary['conversations_total']}\n"
        f"conversations_with_self_answer: {summary['conversations_with_self_answer']}\n"
        f"conversations_with_repeated_questions: {summary['conversations_with_repeated_questions']}\n"
        f"conversations_with_any_violation: {summary['conversations_with_any_violation']} "
        f"({summary['problem_conversations_pct']:.1f}%)"
    )


def _make_compact_report(full_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Компактный отчет:
    - summary только conversation-level метрики
    - failed_turns (все нарушающие ходы)
    - models/token_usage/vacancy_info_used (полезно)
    Без примеров и без дополнительных списков.
    """
    return {
        "run_id": full_report.get("run_id"),
        "started_at": full_report.get("started_at"),
        "mode": full_report.get("mode"),
        "summary": full_report.get("summary", {}),
        "failed_turns": full_report.get("failed_turns", []),
        "models": full_report.get("models", {}),
        "token_usage": full_report.get("token_usage", {}),
        "vacancy_info_used": full_report.get("vacancy_info_used", {}),
    }


# ---------------- main suite ----------------

def run_guardrail_suite_multi_turn(
    conversations: int,
    turns_per_conversation: int,
    report_mode: str,
) -> pathlib.Path:
    ensure_dirs()

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    assistant = create_screening_assistant(cfg)

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    usage = {
        "candidate_generator": _blank_usage(),
        "screening_assistant": _blank_usage(),
        "evaluator": _blank_usage(),
    }

    print("[gen] generating multi-turn candidate conversations...")
    dialogs = generate_candidate_conversations(
        client=client,
        conversations=conversations,
        turns_per_conversation=turns_per_conversation,
        usage_bucket=usage["candidate_generator"],
    )
    print(f"[gen] generated {len(dialogs)} conversations, turns_per_conversation={turns_per_conversation}")

    conversations_payload: List[Dict[str, Any]] = []
    failed_turns: List[Dict[str, Any]] = []

    for cidx, candidate_turns in enumerate(dialogs, start=1):
        print(f"[run] conversation {cidx}/{len(dialogs)}")
        try:
            turns_out = run_assistant_multi_turn(assistant, candidate_turns)
        except AssistantError as exc:
            conversations_payload.append(
                {
                    "conversation_index": cidx,
                    "candidate_turns": candidate_turns,
                    "turns": [],
                    "error": f"{exc}",
                }
            )
            failed_turns.append(
                {
                    "conversation_index": cidx,
                    "turn_index": None,
                    "candidate_message": None,
                    "assistant_reply": None,
                    "flags": {"self_answer": False, "repeated_questions": False},
                    "comment": f"assistant error: {exc}",
                }
            )
            continue

        turn_results: List[Dict[str, Any]] = []
        for t in turns_out:
            cand_msg = t["candidate_message"]
            reply = t["assistant_reply"]

            _accumulate_usage(usage["screening_assistant"], t.get("usage"))

            ev = evaluate_reply(
                client=client,
                candidate_message=cand_msg,
                assistant_reply=reply,
                usage_bucket=usage["evaluator"],
            )

            flags = {
                "self_answer": ev.self_answer,
                "repeated_questions": ev.repeated_questions,
            }

            tr = {
                "turn_index": t["turn_index"],
                "candidate_message": cand_msg,
                "assistant_reply": reply,
                "assistant_conversation_end": t.get("assistant_conversation_end", False),
                "flags": flags,
                "repeated_topics": ev.repeated_topics,
                "comment": ev.comment,
                "used_heuristics": ev.used_heuristics,
            }
            turn_results.append(tr)

            if ev.self_answer or ev.repeated_questions:
                failed_turns.append(
                    {
                        "conversation_index": cidx,
                        "turn_index": t["turn_index"],
                        "candidate_message": cand_msg,
                        "assistant_reply": reply,
                        "flags": flags,
                        "repeated_topics": ev.repeated_topics,
                        "comment": ev.comment,
                    }
                )

        conversations_payload.append(
            {
                "conversation_index": cidx,
                "candidate_turns": candidate_turns,
                "turns": turn_results,
            }
        )

    full_report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "mode": "multi_turn_only",
        "conversations": conversations_payload,
        "failed_turns": failed_turns,
        "token_usage": usage,
        "models": {
            "candidate_generator": GEN_MODEL,
            "evaluator": EVAL_MODEL,
            "screening_assistant": {
                "prompt_id": _component_cfg(cfg, "screening_assistant").get("prompt_id"),
                "prompt_version": _component_cfg(cfg, "screening_assistant").get("prompt_version"),
            },
        },
        "vacancy_info_used": DEFAULT_VACANCY_INFO,
    }

    summary = _build_conversation_level_summary(full_report)
    full_report["summary"] = summary

    print()
    print(_make_console_summary(summary))
    print()

    full_path = REPORTS_DIR / f"screening_guardrails_{run_id}.json"
    compact_path = REPORTS_DIR / f"screening_guardrails_{run_id}_compact.json"

    last_written: pathlib.Path = compact_path

    if report_mode in ("full", "both"):
        full_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] full report saved: {full_path}")
        last_written = full_path

    if report_mode in ("compact", "both"):
        compact_report = _make_compact_report(full_report)
        compact_path.write_text(json.dumps(compact_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] compact report saved: {compact_path}")
        last_written = compact_path

    return last_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guardrails (multi-turn): detect self_answer and repeated_questions (same-question repeats in one message)."
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=DEFAULT_CONVERSATIONS,
        help=f"How many multi-turn conversations to generate (default: {DEFAULT_CONVERSATIONS}).",
    )
    parser.add_argument(
        "--turns-per-conversation",
        type=int,
        default=DEFAULT_TURNS_PER_CONVERSATION,
        help=f"Candidate turns per conversation (default: {DEFAULT_TURNS_PER_CONVERSATION}).",
    )
    parser.add_argument(
        "--report-mode",
        choices=["compact", "full", "both"],
        default="compact",
        help="compact: only violations + summary; full: full report; both: write both.",
    )
    args = parser.parse_args()

    if args.conversations <= 0:
        raise ValueError("--conversations must be positive")
    if args.turns_per_conversation <= 0:
        raise ValueError("--turns-per-conversation must be positive")

    out = run_guardrail_suite_multi_turn(
        conversations=args.conversations,
        turns_per_conversation=args.turns_per_conversation,
        report_mode=args.report_mode,
    )
    print("Report ->", out)


if __name__ == "__main__":
    main()

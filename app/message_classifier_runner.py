from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from openai import OpenAI

from messageLabelGenerator.classifierLLM import ClassifierAssistant

# -----------------------
# Константы и пути
# -----------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "message_classifier"

GEN_MODEL = "gpt-4.1-mini"   # генератор тестовых сообщений
EVAL_MODEL = "gpt-4.1"       # опционально: QA-оценщик "сообщение реально соответствует классу"

DEFAULT_N_PER_CLASS = 3
DEFAULT_SEED = 42

LABELS = ["reason_farewell", "no_reason", "acceptance", "human_needed"]

# -----------------------
# Утилиты
# -----------------------

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
    if isinstance(usage, Mapping):
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
    it, ot, tt = _extract_usage_numbers(usage)
    bucket["input_tokens"] += it
    bucket["output_tokens"] += ot
    bucket["total_tokens"] += tt

def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty json text")
    try:
        return json.loads(text)
    except Exception:
        # пытаемся вырезать JSON-массив/объект из мусора
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if 0 <= start_obj < end_obj:
            return json.loads(text[start_obj : end_obj + 1].strip())
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if 0 <= start_arr < end_arr:
            return json.loads(text[start_arr : end_arr + 1].strip())
        raise

def _confusion_init(labels: List[str]) -> Dict[str, Dict[str, int]]:
    return {a: {p: 0 for p in labels} for a in labels}

# -----------------------
# Простая валидация "сообщение реально похоже на класс"
# (нужно, чтобы генератор не создавал спорные примеры и не портил тест)
# -----------------------

KW_ACCEPT = [
    "интерес", "готов", "да,", "да.", "да", "обсуд", "пообщ", "созвон",
    "можно подробнее", "расскажите", "какая", "какие", "сколько", "вилка",
    "зарплат", "график", "услов", "команда", "обязанност", "стек"
]
KW_REFUSE = ["не интересно", "неинтересно", "не подходит", "не подх", "нет, спасибо", "спасибо, нет", "не рассматриваю", "не буду", "откажусь"]
KW_REASON = ["уже работ", "наш(е)л", "нашла", "принял", "приняла", "офер", "предложение", "в другой сфере", "не мой профиль", "не мой стек", "по локации", "по зарплате", "по графику", "в декрете", "в отпуске"]
KW_UNCERTAIN = ["не уверен", "не уверена", "подумаю", "возможно", "посмотрим", "сомневаюсь", "не знаю", "надо уточнить", "пока рано"]
KW_NEGATIVE = ["достали", "спам", "вы откуда", "что за", "опять", "жесть", "ужас", "какой бред", "идиот", "треш"]

def _looks_like_reason_farewell(t: str) -> bool:
    s = t.lower()
    if not any(k in s for k in KW_REFUSE):
        # допускаем мягкие отказы с причиной без явного "не интересно"
        if not any(k in s for k in ["не рассматриваю", "не буду", "откажусь", "не актуально", "неактуально"]):
            return False
    return any(k in s for k in KW_REASON)

def _looks_like_no_reason(t: str) -> bool:
    s = t.lower()
    # отказ есть, причины нет
    if not any(k in s for k in KW_REFUSE):
        return False
    return not any(k in s for k in KW_REASON)

def _looks_like_acceptance(t: str) -> bool:
    s = t.lower()
    # acceptance включает вопросы по вакансии
    if any(k in s for k in KW_REFUSE):
        return False
    return any(k in s for k in KW_ACCEPT)

def _looks_like_human_needed(t: str) -> bool:
    s = t.lower()
    # неопределенность/жалобы/нерелевант
    if _looks_like_acceptance(t) or _looks_like_no_reason(t) or _looks_like_reason_farewell(t):
        return False
    if any(k in s for k in KW_UNCERTAIN):
        return True
    if any(k in s for k in KW_NEGATIVE):
        return True
    # "непонятно что" - короткие, странные, набор символов
    if len(re.sub(r"\s+", "", s)) <= 3:
        return True
    if re.search(r"[a-zA-Z]{6,}", s):
        return True
    return True

def _validate_for_label(label: str, text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    # по промпту: "one message in Russian"
    # не делаем жесткую проверку, но отсечем чисто английские
    if re.fullmatch(r"[a-zA-Z0-9\s\.\,\!\?\-]+", t):
        return False

    if label == "reason_farewell":
        return _looks_like_reason_farewell(t)
    if label == "no_reason":
        return _looks_like_no_reason(t)
    if label == "acceptance":
        return _looks_like_acceptance(t)
    if label == "human_needed":
        return _looks_like_human_needed(t)
    return False

# -----------------------
# Fallback пул сообщений (детерминированный, чтобы тест всегда можно было прогнать)
# -----------------------

FALLBACK: Dict[str, List[str]] = {
    "reason_farewell": [
        "Спасибо, но я уже устроился на работу и сейчас не рассматриваю предложения.",
        "Благодарю, но я принял оффер и уже выхожу в новую компанию.",
        "Спасибо, но мне сейчас не подходит по локации, переезд не рассматриваю.",
        "Благодарю, но я не рассматриваю смену работы, так как уже работаю в другой сфере.",
    ],
    "no_reason": [
        "Нет, спасибо.",
        "Не подходит.",
        "Неинтересно.",
        "Спасибо, не буду рассматривать.",
    ],
    "acceptance": [
        "Здравствуйте! Да, интересно, можно подробнее про задачи и стек?",
        "Да, давайте обсудим. Какая зарплатная вилка и формат работы?",
        "Интересно. Подскажите, пожалуйста, график и сколько людей в команде?",
        "Да, готов пообщаться. Можете прислать описание обязанностей?",
    ],
    "human_needed": [
        "Я не уверен(а), мне нужно подумать и уточнить несколько моментов.",
        "Вы откуда взяли мой контакт? Похоже на спам.",
        "Странное предложение, непонятно что вы вообще хотите.",
        "ммм ну хз, давайте потом",
    ],
}

# -----------------------
# Генерация сообщений под конкретный класс
# -----------------------

def _gen_prompt_for_label(label: str, n: int) -> str:
    # Важно: генерим только русские сообщения
    common = (
        "Ты генерируешь тестовые сообщения кандидатов на русском языке.\n"
        "Верни строго JSON-массив строк (без markdown и без пояснений).\n"
        "Каждая строка - одно сообщение кандидата.\n"
        "Сообщения должны быть короткие, естественные, как в мессенджере.\n"
        "Не используй слово END.\n"
    )

    if label == "reason_farewell":
        spec = (
            "Класс: reason_farewell.\n"
            "Сгенерируй сообщения, где кандидат ОТКАЗЫВАЕТСЯ и ЯВНО УКАЗЫВАЕТ ПРИЧИНУ.\n"
            "Примеры причин: уже устроился, принял оффер, не подходит локация, не подходит сфера, не ищет работу.\n"
            "Важно: это именно отказ, а не вопросы.\n"
        )
    elif label == "no_reason":
        spec = (
            "Класс: no_reason.\n"
            "Сгенерируй сообщения, где кандидат ОТКАЗЫВАЕТСЯ БЕЗ ПРИЧИНЫ.\n"
            "Фразы типа: 'нет, спасибо', 'не подходит', 'неинтересно'.\n"
            "Не добавляй причин и деталей.\n"
        )
    elif label == "acceptance":
        spec = (
            "Класс: acceptance.\n"
            "Сгенерируй сообщения, где кандидат ЯВНО ИНТЕРЕСУЕТСЯ вакансией.\n"
            "Допускается: согласие + вопросы по вакансии (зарплата, формат, задачи, команда).\n"
            "Важно: не должно быть отказа или сомнений.\n"
        )
    else:
        spec = (
            "Класс: human_needed.\n"
            "Сгенерируй сообщения, которые НЕ являются явным принятием и НЕ являются явным отказом.\n"
            "Это может быть: сомнение, смешанное намерение, жалоба, раздражение, нерелевантный вопрос, непонятный смысл.\n"
            "Важно: не делай явного 'да, интересно' и не делай явного 'нет, не подходит'.\n"
        )

    return common + "\n" + spec + f"\nСгенерируй {n} вариантов."

def generate_messages_for_label(
    client: OpenAI,
    label: str,
    n: int,
    usage_bucket: Dict[str, int],
) -> List[str]:
    prompt = _gen_prompt_for_label(label, n)
    resp = client.responses.create(model=GEN_MODEL, input=prompt)
    _accumulate_usage(usage_bucket, getattr(resp, "usage", None))
    text = (getattr(resp, "output_text", "") or "").strip()

    msgs: List[str] = []
    try:
        data = _safe_json_loads(text)
        if isinstance(data, list):
            msgs = [_normalize_text(str(x)) for x in data if _normalize_text(str(x))]
    except Exception:
        msgs = []

    # фильтруем валидностью под класс
    msgs = [m for m in msgs if _validate_for_label(label, m)]

    # если мало - добиваем fallback
    if len(msgs) < n:
        pool = FALLBACK.get(label, [])
        for m in pool:
            if len(msgs) >= n:
                break
            if _validate_for_label(label, m):
                msgs.append(m)

    return msgs[:n]

# -----------------------
# Прогон message_classifier
# -----------------------

def classify_message(message_text: str, cfg: Dict[str, Any]) -> Tuple[str, Any]:
    mc_cfg = _component_cfg(cfg, "message_classifier")
    prompt_id = mc_cfg.get("prompt_id")
    prompt_version = mc_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError("message_classifier.prompt_id is not set in model.yaml")

    assistant = ClassifierAssistant(prompt_id=prompt_id, prompt_version=prompt_version)
    label = (assistant.run(message_text) or "").strip()
    return label, getattr(assistant, "last_usage", None)

# -----------------------
# Опциональная QA-проверка генератора (чтобы отличать "плохой тест-кейс" от "ошибки классификатора")
# -----------------------

def judge_message_fits_expected_label(
    client: OpenAI,
    message_text: str,
    expected_label: str,
    usage_bucket: Dict[str, int],
) -> Tuple[int, str]:
    instruction = (
        "Ты строгий QA-ревьюер.\n"
        "Тебе дано ОДНО сообщение кандидата на русском и ожидаемая метка.\n"
        "Проверь, действительно ли сообщение соответствует метке.\n"
        "Верни JSON:\n"
        "{\n"
        '  "score": 0 или 1,\n'
        '  "comment": "кратко"\n'
        "}\n"
        "Где score=1 только если соответствие однозначное.\n"
        "Никакого текста вне JSON.\n"
    )
    payload = instruction + "\n\n" + json.dumps(
        {"message": message_text, "expected_label": expected_label},
        ensure_ascii=False,
    )
    resp = client.responses.create(model=EVAL_MODEL, input=payload)
    _accumulate_usage(usage_bucket, getattr(resp, "usage", None))
    text = (getattr(resp, "output_text", "") or "").strip()

    try:
        data = _safe_json_loads(text)
        score = int(data.get("score", 0))
        if score not in (0, 1):
            score = 0
        comment = str(data.get("comment", "")).strip() or "No comment."
        return score, comment
    except Exception:
        return 0, f"Failed to parse judge output: {text[:200]}"

# -----------------------
# Основной раннер
# -----------------------

def run_message_classifier_tests(
    n_per_class: int,
    seed: int,
    use_llm_generation: bool,
    enable_judge: bool,
) -> pathlib.Path:
    ensure_dirs()

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    rnd = random.Random(seed)
    client = OpenAI(api_key=api_key)

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    usage = {
        "generator": _blank_usage(),
        "classifier": _blank_usage(),
        "judge": _blank_usage(),
    }

    cases: List[Dict[str, Any]] = []
    confusion = _confusion_init(LABELS)

    total = 0
    correct = 0
    skipped_by_bad_case = 0

    for expected_label in LABELS:
        # 1) генерим/берем сообщения
        if use_llm_generation:
            messages = generate_messages_for_label(client, expected_label, n_per_class, usage["generator"])
        else:
            pool = FALLBACK.get(expected_label, [])
            rnd.shuffle(pool)
            messages = pool[:n_per_class]

        for msg in messages:
            msg = _normalize_text(msg)

            # 1.1) дополнительная защита: если сообщение вообще не похоже на класс, считаем кейс "плохим"
            if not _validate_for_label(expected_label, msg):
                skipped_by_bad_case += 1
                continue

            judge_score = None
            judge_comment = None
            if enable_judge:
                js, jc = judge_message_fits_expected_label(client, msg, expected_label, usage["judge"])
                judge_score, judge_comment = js, jc
                # если сам QA считает кейс спорным - можно пропустить, чтобы тест был "чистым"
                if judge_score == 0:
                    skipped_by_bad_case += 1
                    continue

            # 2) классифицируем
            predicted, cls_usage = classify_message(msg, cfg)
            _accumulate_usage(usage["classifier"], cls_usage)

            if predicted not in LABELS:
                predicted = "human_needed"  # безопасный fallback

            confusion[expected_label][predicted] += 1

            is_ok = int(predicted == expected_label)
            total += 1
            correct += is_ok

            cases.append(
                {
                    "expected_label": expected_label,
                    "predicted_label": predicted,
                    "ok": bool(is_ok),
                    "message": msg,
                    "judge_score": judge_score,
                    "judge_comment": judge_comment,
                }
            )

    accuracy = (correct / total) if total else 0.0

    # группируем ошибки
    failed = [c for c in cases if not c.get("ok")]
    failed_by_expected: Dict[str, List[Dict[str, Any]]] = {}
    for c in failed:
        failed_by_expected.setdefault(c["expected_label"], []).append(c)

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "n_per_class": n_per_class,
        "seed": seed,
        "use_llm_generation": use_llm_generation,
        "enable_judge": enable_judge,
        "total_cases_scored": total,
        "correct": correct,
        "accuracy": accuracy,
        "skipped_by_bad_case": skipped_by_bad_case,
        "labels": LABELS,
        "confusion_matrix": confusion,
        "failed_examples": failed[:50],  # ограничим
        "failed_by_expected": {k: v[:20] for k, v in failed_by_expected.items()},
        "cases": cases,
        "token_usage": usage,
        "models": {
            "generator": GEN_MODEL if use_llm_generation else None,
            "judge": EVAL_MODEL if enable_judge else None,
        },
        "prompts": {
            "message_classifier": {
                "prompt_id": _component_cfg(cfg, "message_classifier").get("prompt_id"),
                "prompt_version": _component_cfg(cfg, "message_classifier").get("prompt_version"),
            }
        },
    }

    out_path = REPORTS_DIR / f"message_classifier_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] Report saved to: {out_path}")
    print(f"[summary] total={total} correct={correct} accuracy={accuracy:.3f} skipped={skipped_by_bad_case}")
    return out_path

# -----------------------
# CLI
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test message_classifier prompt (reason_farewell/no_reason/acceptance/human_needed).")
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS, help="How many messages per class to test.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (used for fallback shuffle).")
    parser.add_argument("--no-gen", action="store_true", help="Do not use LLM generation, only fixed fallback pool.")
    parser.add_argument("--judge", action="store_true", help="Enable extra QA judge to filter ambiguous generated cases.")
    args = parser.parse_args()

    report_path = run_message_classifier_tests(
        n_per_class=max(1, args.n_per_class),
        seed=args.seed,
        use_llm_generation=not args.no_gen,
        enable_judge=bool(args.judge),
    )
    print("message_classifier report ->", report_path)

if __name__ == "__main__":
    main()

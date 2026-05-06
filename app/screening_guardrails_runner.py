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
    "contact_source": "LinkedIn",
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


def _normalize_text(s: str, *, remove_end: bool = True) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    if remove_end:
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


def _make_dialog_line(candidate_message: str, assistant_reply_raw: str) -> str:
    cand = _normalize_text(candidate_message, remove_end=True)
    asst = _normalize_text(assistant_reply_raw, remove_end=False)
    return f"[candidate] {cand}\n[recruiter] {asst}"


# ---------------- heuristics fallback ----------------

STOP_WORDS = {
    "и", "а", "но", "или", "ли", "же", "то", "это", "в", "на", "по", "за", "к", "с", "у", "о",
    "вы", "вас", "вам", "мы", "я", "он", "она", "они", "оно", "что", "как", "где", "когда",
    "пожалуйста", "спасибо", "поняла", "понял", "отлично", "хотела", "хотел", "уточнить",
}

FINISH_PATTERNS = [
    r"\bэто\s+вся\s+информац",
    r"\bэто\s+все\b",
    r"\bвся\s+информация\b",
    r"\bвсе\s+что\s+мне\s+нужно\b",
    r"\bя\s+передам\b",
    r"\bпередам\s+.*рекрутер",
    r"\bвнутренн(ему|ий)\s+рекрутер",
    r"\bследующ(ие|их)\s+шаг",
    r"\bсвяжетс[яь]\s+с\s+вами\b",
    r"\bна\s+данном\s+этапе\b",
]
FINISH_RE = re.compile("|".join(FINISH_PATTERNS), re.I)


def _tokenize_for_similarity(text: str) -> List[str]:
    t = _normalize_text(text, remove_end=True).lower()
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
    t = _normalize_text(text, remove_end=True)
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


def _has_questions_in_reply(text_raw: str) -> bool:
    """
    True если в реплике рекрутера есть хотя бы один вопрос.
    Используется как ЖЕСТКИЙ гейт для premature_end_after_questions:
    если вопросов нет - premature_end_after_questions обязан быть False.
    """
    if not text_raw:
        return False

    segments = _extract_question_segments(text_raw)
    if segments:
        return True

    t = _normalize_text(text_raw, remove_end=True)
    if not t:
        return False

    return bool(re.search(
        r"\b(подскажите|уточните|расскажите|могли бы|можете|какой|какая|какие|сколько|где|готовы ли|ориентируетесь)\b",
        t,
        re.I,
    ))


TOPIC_KEYWORDS = {
    "location": ["город", "локац", "где вы", "где сейчас", "находит", "прожив", "москва", "питер", "спб"],
    "salary": ["зарплат", "вилка", "оклад", "доход", "компенсац", "ожидан", "сумм", "руб", "тыс", "₽"],
    "relocation": ["переезд", "переехать", "готовы ли", "готовность", "релокац", "офис", "гибрид"],
    "experience": ["опыт", "работал", "работали", "проект", "проекты", "задач", "фреймворк", "стек"],
}


def _keyword_topic_hits(text: str) -> Dict[str, int]:
    t = _normalize_text(text, remove_end=True).lower()
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
    """
    Fallback-only эвристика.
    Важно: НЕ считать self_answer, если рекрутер просто отвечает на вопрос кандидата.
    Self_answer тут - когда рекрутер "притворяется кандидатом" (вставляет ответы кандидата).
    """
    t = _normalize_text(text, remove_end=False)
    low = t.lower()

    if "кандидат:" in low or "candidate:" in low:
        return True, "heuristic: role label candidate:"

    # типичная ошибка: "250 000, Москва. Спасибо!..." внутри одного сообщения рекрутера
    if "?" in t:
        after_q = t.split("?", 1)[-1].strip()
        if re.search(r"^(250|230|240|260|москва|да|нет|ага|конечно|у меня)\b", after_q, re.I):
            return True, "heuristic: answer-like text after question"

    return False, "heuristic: no self-answer signal"


def heuristic_premature_end_after_questions(text_raw: str, conversation_end_flag: bool) -> Tuple[bool, str]:
    """
    Fallback-only эвристика.
    """
    t_raw = _normalize_text(text_raw, remove_end=False)
    if not t_raw:
        return False, "heuristic: empty reply"

    segments = _extract_question_segments(t_raw)
    has_q = len(segments) >= 1

    has_end_token = "END" in t_raw
    has_finish_phrase = bool(FINISH_RE.search(t_raw))

    if has_q and (conversation_end_flag or has_end_token or has_finish_phrase):
        why = []
        if conversation_end_flag:
            why.append("conversation_end_flag")
        if has_end_token:
            why.append("END_token")
        if has_finish_phrase:
            why.append("finish_phrase")
        return True, f"heuristic: questions + end signal ({', '.join(why)})"

    return False, "heuristic: no questions+end in same message"


# ---------------- candidate multi-turn generation ----------------

def _seed_pool() -> List[str]:
    return [
        "Добрый день\n230_000₽\nМосква",
        "Москва, 240 000 на руки. Готов ответить на вопросы по опыту.",
        "Город Москва. Ожидания 250 000. Давайте пройдемся по вопросам.",

        (
            "В МТС работал больше с прикладом на готовом продукте и его сопровождением. "
            "Все в кубере, доставляли через ArgoCD, сборка на GitLab. Мониторинг Grafana/Prometheus, "
            "логирование OpenSearch/Kibana/Jaeger. Машинки нарезаны Terraform. Основной язык Go, фронт Node.js.\n"
            "В Уралсибе команду собрали с нуля, платформа. Тачки Terraform, кубер раскатили Ansible. "
            "Деплой GitLab/ArgoCD. В кластере хранилка Ceph + rook-operator. Сеть на CNI, без меша, до Istio не дошли. "
            "Логи: Kafka буфер -> OpenSearch, борды в OpenSearch Dashboard + Graylog. "
            "Мониторинг VictoriaMetrics стек до Alertmanager + Grafana. Трейсы: OTel Collector + ClickHouse + Grafana.\n"
            "По мелочи поднимали Jenkins агентов, раннеры, SonarQube/sonar-scanner, "
            "делали свою сборку в Docker, серты и всякая мелочь."
        ),
        (
            "Если коротко: k8s, ArgoCD, GitLab CI, Terraform, Prometheus/Grafana, OpenSearch, трассировка через OTel. "
            "В одном проекте Ceph+rook, в другом больше упор на наблюдаемость и пайплайны. "
            "Могу по стеку и зонам ответственности расписать."
        ),

        "Да, мса - микросервисы же, да? UML опыт есть, но чаще BPMN. Подскажите, что за конмания?",
        "Микросервисы - ок. Диаграммы делал, но больше BPMN. А компания какая? Есть ссылка на вакансию?",

        "5 лет\nДа",
        "3 года. Да.",
        "Да.\nНет.\n5 лет.",
        "Ок. Понял. Да.",

        (
            "Добрый день, Майя!\n\n"
            "Я в Москве сейчас. По зарплате обычно отталкиваюсь от конкретной роли и обязанностей. "
            "Хотелось бы узнать больше о позиции и компании прежде, чем называть цифры в воздух. "
            "Я за то, чтобы сначала нормально пообщаться. Что скажете?"
        ),
        (
            "По ожиданиям могу сказать после того, как уточним формат, стек и зону ответственности. "
            "Можете прислать больше деталей по роли? Если проще - давайте короткий созвон."
        ),

        "Если вопросы по Python/SQL/нагрузке - могу отвечать по пунктам, только скажите в каком формате удобнее.",
        "Сейчас больше бэкенд и интеграции. По базам - Postgres, немного ClickHouse. По нагрузкам - кэш, очереди, метрики.",
        "По локации Москва, но интересна удаленка. По переезду нет. График стандарт, могу быть на связи в МСК.",
        "Скиньте, пожалуйста, описание вакансии или ссылку. Я хочу понять контекст, а потом отвечу точнее.",
        "Как вы нашли мой Telegram?",
        "Откуда у вас мой номер? Если удобно, можем продолжить после ответа.",
        "Вы мошенники? Пришлите корпоративную почту.",
        "Это все есть в резюме и в LinkedIn.",
        "Давайте вернемся к диалогу позже.",
        "Сейчас не время, позже обсудим.",
        "Ок, буду ждать звонка.",
        "50",
        "300",
        "60-80",
        "1500 в час",
        "$4000",
        "3000 евро",
    ]


def _generation_prompt_multi_turn(conversations: int, turns_per_conversation: int) -> str:
    seed = _seed_pool()
    lines = [
        f"Сгенерируй {conversations} независимых диалогов кандидата (multi-turn).",
        f"В каждом диалоге должно быть ровно {turns_per_conversation} сообщений кандидата.",
        "Каждый диалог это массив строк. Итоговый ответ это JSON массив диалогов.",
        "",
        "Цель: кандидатскими сообщениями продвинуть диалог так, чтобы рекрутер начал задавать вопросы по квалификации.",
        "",
        "КРИТИЧНО: разнообразь стиль кандидата между диалогами. Не делай все диалоги одинаковыми.",
        "Распредели паттерны по диалогам (примерно равномерно):",
        "1) Длинный технарский монолог: Kubernetes/ArgoCD/GitLab CI/Terraform/observability (Grafana/Prometheus/OpenSearch/OTel и т.п.).",
        "2) Короткие ответы: 'да/нет/5 лет' и подобное.",
        "3) Кандидат уточняет термины/формат ('мса - микросервисы?') и задает вопрос 'что за компания?'",
        "4) Вежливо уходит от цифр по зарплате, просит детали роли и предлагает созвон.",
        "5) Спрашивает, откуда взяли его Telegram / номер / профиль.",
        "6) Подозревает мошенничество, просит корпоративную почту или официальное подтверждение.",
        "7) Вместо прямого ответа пишет 'это есть в резюме / профиле / LinkedIn'.",
        "8) Просит вернуться к диалогу позже или пишет, что будет ждать звонка.",
        "9) Называет зарплату двузначным или трехзначным числом без единиц, в валюте или как почасовую ставку.",
        "",
        "Ограничения:",
        "- Не пиши про политику, национальности, власть, страны и т.п.",
        "- Не обвиняй в спаме и не проси удалить контакт.",
        "- Не используй оскорбления и мат.",
        "",
        "Вариативность языка и формы:",
        "- Можно использовать сленг и опечатки (например 'конмания', 'кубер', 'тачки нарезали'), но без перебора.",
        "- Иногда пиши в 1-2 строках, иногда 1 очень длинным сообщением (в рамках одного кандидата).",
        "",
        "Примеры реплик кандидата (используй как стиль, не копируй дословно):",
    ]
    for i, s in enumerate(seed, start=1):
        preview = s.replace("\n", " ").strip()
        if len(preview) > 260:
            preview = preview[:257] + "..."
        lines.append(f"{i}) {preview}")
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

        reply_raw = str(result.response or "")

        turns_out.append(
            {
                "turn_index": idx,
                "candidate_message": _normalize_text(cand_msg, remove_end=True),
                "assistant_reply_raw": _normalize_text(reply_raw, remove_end=False),
                "assistant_conversation_end": bool(getattr(result, "conversation_end", False)),
                "usage": getattr(result, "usage", None),
            }
        )

        if getattr(result, "conversation_end", False):
            break

    return turns_out


# ---------------- evaluation (LLM for A,B,C; heuristics only fallback) ----------------

ALLOWED_TOPICS = {"location", "salary", "relocation", "experience", "other"}


def _eval_instruction() -> str:
    return (
        "Ты проверяешь одно сообщение рекрутера на 3 нарушения. Верни строго JSON.\n"
        "\n"
        "A) self_answer = true, если рекрутер пишет за кандидата (вставляет ответы кандидата в своем сообщении)\n"
        "или ведет диалог за обе стороны.\n"
        "Это НЕ self_answer, если рекрутер просто отвечает на вопрос кандидата (например: 'Да, речь о микросервисах').\n"
        "Примеры self_answer:\n"
        "- рекрутер: '...Подскажите зарплату? 250 000, Москва. Спасибо!'\n"
        "- рекрутер вставляет 'Кандидат:' или 'Candidate:' и дальше ответы кандидата.\n"
        "\n"
        "B) repeated_questions = true, если рекрутер в ОДНОМ сообщении повторяет по смыслу один и тот же вопрос.\n"
        "Не путать с 2-3 разными вопросами. Это именно повтор одного и того же:\n"
        "- дважды про город, дважды про зарплату, дважды про переезд, и т.п.\n"
        "\n"
        "C) premature_end_after_questions = true, если рекрутер В ТОМ ЖЕ сообщении:\n"
        "- задает хотя бы один вопрос, И\n"
        "- одновременно закрывает диалог прямо сейчас (ожидается, что отвечать уже не нужно),\n"
        "  например: ставит END, пишет 'Это вся информация, я передам...', или явно говорит что разговор завершен.\n"
        "Важно: фразы про 'внутренний рекрутер свяжется' сами по себе НЕ означают конец диалога,\n"
        "если рекрутер продолжает собирать данные и задает вопросы.\n"
        "conversation_end_flag=true - это сильный сигнал завершения.\n"
        "Если в сообщении нет вопросов - premature_end_after_questions всегда false.\n"
        "\n"
        "Верни JSON:\n"
        "{"
        "\"self_answer\": true|false, "
        "\"repeated_questions\": true|false, "
        "\"premature_end_after_questions\": true|false, "
        "\"repeated_topics\": [\"location\"|\"salary\"|\"relocation\"|\"experience\"|\"other\"], "
        "\"comment\": \"кратко почему\""
        "}"
    )


def _eval_payload(candidate_message: str, assistant_reply_raw: str, conversation_end_flag: bool) -> str:
    payload = {
        "instruction": _eval_instruction(),
        "candidate_message": candidate_message,
        "assistant_reply": assistant_reply_raw,
        "conversation_end_flag": conversation_end_flag,
    }
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class EvalResult:
    self_answer: bool
    repeated_questions: bool
    premature_end_after_questions: bool
    repeated_topics: List[str]
    comment: str
    used_heuristics: bool


def _sanitize_topics(topics: Any) -> List[str]:
    if not isinstance(topics, list):
        return []
    out: List[str] = []
    for x in topics[:10]:
        s = str(x).strip()
        if not s:
            continue
        if s not in ALLOWED_TOPICS:
            s = "other"
        out.append(s)
    uniq: List[str] = []
    seen = set()
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def evaluate_reply(
    client: OpenAI,
    candidate_message: str,
    assistant_reply_raw: str,
    conversation_end_flag: bool,
    usage_bucket: Dict[str, int],
) -> EvalResult:
    payload = _eval_payload(candidate_message, assistant_reply_raw, conversation_end_flag)
    raw = ""

    # Жесткий гейт: если в этой реплике нет вопроса - premature_end быть не может
    has_q = _has_questions_in_reply(assistant_reply_raw)

    try:
        response = client.responses.create(model=EVAL_MODEL, input=payload)
        _accumulate_usage(usage_bucket, getattr(response, "usage", None))
        raw = _normalize_text(getattr(response, "output_text", "") or "", remove_end=False)
        data = _safe_json_loads(raw)

        self_answer = bool(data.get("self_answer"))
        repeated_questions = bool(data.get("repeated_questions"))
        premature_end = bool(data.get("premature_end_after_questions"))
        repeated_topics = _sanitize_topics(data.get("repeated_topics"))
        comment = str(data.get("comment") or "").strip()

        if not has_q:
            # подавляем ложноположительные срабатывания, как в твоем кейсе
            if premature_end:
                note = "premature_end overridden to false: no questions in this recruiter message"
                comment = f"{comment} | {note}" if comment else note
            premature_end = False

        return EvalResult(
            self_answer=self_answer,
            repeated_questions=repeated_questions,
            premature_end_after_questions=premature_end,
            repeated_topics=repeated_topics,
            comment=comment,
            used_heuristics=False,
        )

    except Exception:
        # fallback heuristics
        sa, sa_reason = heuristic_self_answer(assistant_reply_raw)
        rq, rq_reason, topics = heuristic_repeated_questions(assistant_reply_raw)
        pe, pe_reason = heuristic_premature_end_after_questions(assistant_reply_raw, conversation_end_flag)

        # тот же жесткий гейт на fallback
        if not has_q:
            pe = False
            pe_reason = "forced false: no questions in recruiter message"

        comment = (
            "Eval parse failed, heuristics used. "
            f"self_answer={sa_reason}; repeated={rq_reason}; premature_end={pe_reason}; raw={raw[:120]}"
        )

        return EvalResult(
            self_answer=sa,
            repeated_questions=rq,
            premature_end_after_questions=pe,
            repeated_topics=_sanitize_topics(topics or (["other"] if rq else [])),
            comment=comment,
            used_heuristics=True,
        )


# ---------------- reporting ----------------

def _build_conversation_level_summary(full_report: Dict[str, Any]) -> Dict[str, Any]:
    conversations = full_report.get("conversations", [])
    total = len(conversations)

    conv_self = 0
    conv_rep = 0
    conv_premature = 0
    conv_any = 0

    for conv in conversations:
        has_self = False
        has_rep = False
        has_premature = False
        for t in conv.get("turns", []):
            flags = t.get("flags") or {}
            if flags.get("self_answer"):
                has_self = True
            if flags.get("repeated_questions"):
                has_rep = True
            if flags.get("premature_end_after_questions"):
                has_premature = True

        if has_self:
            conv_self += 1
        if has_rep:
            conv_rep += 1
        if has_premature:
            conv_premature += 1
        if has_self or has_rep or has_premature:
            conv_any += 1

    pct = (conv_any / total * 100.0) if total else 0.0

    return {
        "conversations_total": total,
        "conversations_with_self_answer": conv_self,
        "conversations_with_repeated_questions": conv_rep,
        "conversations_with_premature_end_after_questions": conv_premature,
        "conversations_with_any_violation": conv_any,
        "problem_conversations_pct": pct,
    }


def _make_console_summary(summary: Dict[str, Any]) -> str:
    return (
        "=== GUARDRAILS SUMMARY (conversation-level) ===\n"
        f"conversations_total: {summary['conversations_total']}\n"
        f"conversations_with_self_answer: {summary['conversations_with_self_answer']}\n"
        f"conversations_with_repeated_questions: {summary['conversations_with_repeated_questions']}\n"
        f"conversations_with_premature_end_after_questions: {summary['conversations_with_premature_end_after_questions']}\n"
        f"conversations_with_any_violation: {summary['conversations_with_any_violation']} "
        f"({summary['problem_conversations_pct']:.1f}%)"
    )


def _make_compact_report(full_report: Dict[str, Any]) -> Dict[str, Any]:
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
                    "turns": [],
                    "error": f"{exc}",
                }
            )
            failed_turns.append(
                {
                    "conversation_index": cidx,
                    "turn_index": None,
                    "dialog": None,
                    "flags": {
                        "self_answer": False,
                        "repeated_questions": False,
                        "premature_end_after_questions": False,
                    },
                    "repeated_topics": [],
                    "comment": f"assistant error: {exc}",
                }
            )
            continue

        turn_results: List[Dict[str, Any]] = []
        for t in turns_out:
            turn_idx = int(t["turn_index"])
            cand_msg = t["candidate_message"]
            reply_raw = t["assistant_reply_raw"]
            conv_end = bool(t.get("assistant_conversation_end", False))

            _accumulate_usage(usage["screening_assistant"], t.get("usage"))

            ev = evaluate_reply(
                client=client,
                candidate_message=cand_msg,
                assistant_reply_raw=reply_raw,
                conversation_end_flag=conv_end,
                usage_bucket=usage["evaluator"],
            )

            flags = {
                "self_answer": ev.self_answer,
                "repeated_questions": ev.repeated_questions,
                "premature_end_after_questions": ev.premature_end_after_questions,
            }

            dialog_line = _make_dialog_line(cand_msg, reply_raw)

            tr = {
                "turn_index": turn_idx,
                "dialog": dialog_line,
                "assistant_conversation_end": conv_end,
                "flags": flags,
                "repeated_topics": ev.repeated_topics,
                "comment": ev.comment,
                "used_heuristics": ev.used_heuristics,
            }
            turn_results.append(tr)

            if ev.self_answer or ev.repeated_questions or ev.premature_end_after_questions:
                failed_turns.append(
                    {
                        "conversation_index": cidx,
                        "turn_index": turn_idx,
                        "dialog": dialog_line,
                        "flags": flags,
                        "repeated_topics": ev.repeated_topics,
                        "comment": ev.comment,
                    }
                )

        conversations_payload.append(
            {
                "conversation_index": cidx,
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
        description="Guardrails (multi-turn): detect self_answer, repeated_questions, premature_end_after_questions (LLM-based)."
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

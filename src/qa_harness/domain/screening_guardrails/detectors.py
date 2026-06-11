"""Эвристические детекторы нарушений-гардрейлов в ответе screening_assistant (fallback для LLM-судьи).

Три нарушения (перенос из app/screening_guardrails_runner.py):
- self_answer: ассистент пишет ЗА кандидата (вставляет ответы кандидата / роль-метки);
- repeated_questions: в одном сообщении повторяет по смыслу один и тот же вопрос (jaccard / темы);
- premature_end_after_questions: в одном сообщении задаёт вопрос И тут же закрывает диалог.
Жёсткий гейт: нет вопросов в реплике → premature_end невозможен.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

STOP_WORDS = {
    "и", "а", "но", "или", "ли", "же", "то", "это", "в", "на", "по", "за", "к", "с", "у", "о",
    "вы", "вас", "вам", "мы", "я", "он", "она", "они", "оно", "что", "как", "где", "когда",
    "пожалуйста", "спасибо", "поняла", "понял", "отлично", "хотела", "хотел", "уточнить",
}

FINISH_RE = re.compile(
    "|".join([
        r"\bэто\s+вся\s+информац", r"\bэто\s+все\b", r"\bвся\s+информация\b",
        r"\bвсе\s+что\s+мне\s+нужно\b", r"\bя\s+передам\b", r"\bпередам\s+.*рекрутер",
        r"\bвнутренн(ему|ий)\s+рекрутер", r"\bследующ(ие|их)\s+шаг",
        r"\bсвяжетс[яь]\s+с\s+вами\b", r"\bна\s+данном\s+этапе\b",
    ]),
    re.IGNORECASE,
)

TOPIC_KEYWORDS = {
    "location": ["город", "локац", "где вы", "где сейчас", "находит", "прожив", "москва", "питер", "спб"],
    "salary": ["зарплат", "вилка", "оклад", "доход", "компенсац", "ожидан", "сумм", "руб", "тыс", "₽"],
    "relocation": ["переезд", "переехать", "готовы ли", "готовность", "релокац", "офис", "гибрид"],
    "experience": ["опыт", "работал", "работали", "проект", "проекты", "задач", "фреймворк", "стек"],
}

_QUESTION_WORDS = r"\b(подскажите|уточните|расскажите|могли бы|можете|какой|какая|какие|сколько|где|готовы ли|ориентируетесь)\b"


def normalize(s: str, *, remove_end: bool = True) -> str:
    t = (s or "").strip()
    if remove_end:
        t = re.sub(r"\bEND\b", "", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    t = re.sub(r"[^a-zа-я0-9\s]", " ", normalize(text).lower(), flags=re.IGNORECASE)
    return [p for p in t.split() if p not in STOP_WORDS and len(p) >= 3]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def extract_question_segments(text: str) -> List[str]:
    t = normalize(text)
    if not t:
        return []
    chunks: List[str] = []
    for p in t.split("?")[:-1]:
        seg = (p.strip() + "?").strip()
        if seg:
            chunks.append(seg)
    for line in t.split("\n"):
        line = line.strip()
        if not line or "?" in line:
            continue
        if re.search(r"\b(подскажите|уточните|расскажите|могли бы|можете|какой|какая|какие|сколько|где|готовы ли)\b", line, re.I):
            chunks.append(line)
    uniq, seen = [], set()
    for c in chunks:
        k = c.lower().strip()
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def has_questions_in_reply(text_raw: str) -> bool:
    if not text_raw:
        return False
    if extract_question_segments(text_raw):
        return True
    t = normalize(text_raw)
    return bool(t and re.search(_QUESTION_WORDS, t, re.I))


def _keyword_topic_hits(text: str) -> Dict[str, int]:
    t = normalize(text).lower()
    return {topic: sum(1 for kw in keys if kw in t) for topic, keys in TOPIC_KEYWORDS.items()}


def heuristic_self_answer(text: str) -> Tuple[bool, str]:
    t = normalize(text, remove_end=False)
    if "кандидат:" in t.lower() or "candidate:" in t.lower():
        return True, "role label candidate:"
    if "?" in t:
        after_q = t.split("?", 1)[-1].strip()
        if re.search(r"^(250|230|240|260|москва|да|нет|ага|конечно|у меня)\b", after_q, re.I):
            return True, "answer-like text after question"
    return False, "no self-answer signal"


def heuristic_repeated_questions(text: str) -> Tuple[bool, str, List[str]]:
    segments = extract_question_segments(text)
    if len(segments) <= 1:
        return False, "<=1 question segment", []
    toks = [_tokenize(s) for s in segments]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            sim = _jaccard(toks[i], toks[j])
            if sim >= 0.72:
                return True, f"similar questions (jaccard={sim:.2f})", ["other"]
    topic_counts = {k: 0 for k in TOPIC_KEYWORDS}
    for seg in segments:
        for topic, v in _keyword_topic_hits(seg).items():
            if v > 0:
                topic_counts[topic] += 1
    repeated = [t for t, c in topic_counts.items() if c >= 2]
    if repeated:
        return True, f"repeated topic(s): {', '.join(repeated)}", repeated
    return False, "no clear repetition", []


def heuristic_premature_end(text_raw: str, conversation_end_flag: bool) -> Tuple[bool, str]:
    t = normalize(text_raw, remove_end=False)
    if not t:
        return False, "empty reply"
    has_q = len(extract_question_segments(t)) >= 1
    has_end_token = "END" in t
    has_finish = bool(FINISH_RE.search(t))
    if has_q and (conversation_end_flag or has_end_token or has_finish):
        why = [w for w, c in (("conversation_end_flag", conversation_end_flag),
                              ("END_token", has_end_token), ("finish_phrase", has_finish)) if c]
        return True, f"questions + end signal ({', '.join(why)})"
    return False, "no questions+end in same message"

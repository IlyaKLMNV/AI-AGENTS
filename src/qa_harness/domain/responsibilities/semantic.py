"""Семантика responsibilities_parser по golden + заземление.

Контракт (contract.py) проверяет ФОРМУ списка; здесь — СМЫСЛ:
- `check_semantics(predicted, expect, forbid)` (gate): ожидаемые КРИТЕРИИ покрыты ВНУТРИ требований-
  предложений, запрещённые — нет. expect — список ИЛИ-групп: группа удовлетворена, если хоть одна форма
  совпала с каким-то требованием (нормализованная подстрока в обе стороны — концепт «Python» матчится
  внутри «У кандидата есть опыт разработки на Python»). forbid аналогично (nice-to-have/soft/условия).
- `grounding_missing_anchors(predicted, vacancy_text)` (СИГНАЛ, не gate): какие ANCHOR-термы (латиница-
  технологии + ключевые ru-якоря) из требований НЕ найдены в тексте вакансии. Не валит кейс: промпт
  легально переформулирует предложение, поэтому заземляем только опорные термы, а не всю фразу.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple


def _norm(s: str) -> str:
    t = (s or "").strip().lower().replace("ё", "е")
    return re.sub(r"[^a-z0-9а-я]+", "", t)


def _as_forms(item: Any) -> List[str]:
    if isinstance(item, (list, tuple)):
        return [str(x) for x in item if str(x).strip()]
    return [str(item)] if str(item).strip() else []


def _match(form: str, key: str) -> bool:
    f, k = _norm(form), _norm(key)
    return bool(f and k and (f in k or k in f))


def check_semantics(
    predicted: List[str],
    expect: Sequence[Any],
    forbid: Sequence[Any],
) -> Tuple[bool, List[str]]:
    """Вернуть (ok, diffs). expect — ИЛИ-группы обязательных терминов, forbid — запреты."""
    keys = [k for k in predicted if _norm(k)]
    diffs: List[str] = []
    for item in expect or []:
        forms = _as_forms(item)
        if forms and not any(_match(f, k) for f in forms for k in keys):
            diffs.append(f"missing:{'|'.join(forms)}")
    for term in forbid or []:
        if str(term).strip() and any(_match(term, k) for k in keys):
            diffs.append(f"forbidden:{term}")
    return (len(diffs) == 0), diffs


# ----- заземление: лёгкий стеммер (перенос из легаси _soft_word_key) -----

_EN_SUFFIXES = ("ings", "ing", "ers", "ies", "es", "s")
_RU_SUFFIXES = (
    "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "его", "ией",
    "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ых", "их", "ую", "юю",
    "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ом", "ем",
    "а", "я", "ы", "и", "у", "ю", "е", "о",
)


def _stem(word: str) -> str:
    """Грубый стем: нормализация + срез одного англ. и одного рус. окончания (с защитой по длине)."""
    t = re.sub(r"[^a-z0-9а-я]+", "", (word or "").strip().lower().replace("ё", "е"))
    if not t:
        return ""
    for suf in _EN_SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    for suf in _RU_SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t


def _stem_tokens(s: str) -> List[str]:
    return [st for st in (_stem(tok) for tok in re.findall(r"[A-Za-z0-9А-Яа-я#+]+", str(s or ""))) if st]


def _token_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)  # склонение/стем-вариация


# Опорные термы: латиница-технологии (Node.js/CI-CD/gRPC) + ключевые ru-якоря (важны для проверяемости).
_ANCHOR_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+.#/_-]{1,}")
_RU_ANCHORS = ("английск", "сертификат", "высше", "образовани", "магистр", "бакалавр", "релокац")


def _anchor_terms(requirement: str) -> List[str]:
    """Опорные термы требования: латиница-технологии + ключевые ru-якоря (если есть)."""
    out: List[str] = [t.rstrip(".,/;:-_") for t in _ANCHOR_LATIN.findall(requirement or "")]
    out = [t for t in out if len(t) >= 2]
    low = (requirement or "").lower()
    out += [a for a in _RU_ANCHORS if a in low]
    return out


def grounding_missing_anchors(predicted: List[str], vacancy_text: str) -> List[str]:
    """ANCHOR-термы требований, не заземлённые в тексте вакансии (стем-матч). СИГНАЛ, не gate.

    Заземляем только опорные термы (технологии/ключевые якоря), а не всю фразу — промпт законно
    переформулирует требование, поэтому полнотекстовый грундинг давал бы ложные miss.
    """
    text_tokens = _stem_tokens(vacancy_text)
    misses: List[str] = []
    seen: set = set()
    for req in predicted:
        for anchor in _anchor_terms(req):
            key = anchor.lower()
            if key in seen:
                continue
            seen.add(key)
            a_tokens = _stem_tokens(anchor)
            if a_tokens and not all(any(_token_match(at, tt) for tt in text_tokens) for at in a_tokens):
                misses.append(anchor)
    return misses

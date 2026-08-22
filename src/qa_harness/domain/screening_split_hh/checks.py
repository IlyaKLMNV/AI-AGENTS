"""Детерминированные проверки по трассе split-хода — HH-канал (слой A/B).

Дельта к `screening_split/checks.py` (TG):
- слой A: `evaluate_analyzer` дополнен ключом `expect_field_work` (state.field_work_check) —
  остальные инварианты (salary/format/script_key/asking/event/end/…) общие, берём из TG;
- слой B: `leak_scan` — без ветки утечки ссылки/СКРЫТО (в hh их нет) + новая канарейка
  «сырые id формата (`ON_SITE/REMOTE/HYBRID/FIELD_WORK`) не попадают в текст кандидату»
  (Интервьюер обязан переформулировать по-русски — SPLIT_TG_VS_HH.md §1).

`load_checks`/`CheckResult`/`LeakResult` и разбор трассы переиспользуем из TG (без дрейфа).
"""

from typing import Any, Dict, List

# переиспользуемые из TG: загрузчик инвариантов, типы, приватные хелперы разбора трассы/токенов вилки.
from qa_harness.domain.screening_split.checks import (  # noqa: F401 — re-export load_checks
    CheckResult,
    LeakResult,
    _final_state,
    _salary_variants,
    evaluate_analyzer as _tg_evaluate_analyzer,
    injection_scan,  # noqa: F401 — канарейки prompt injection общие для каналов (чистый текст)
    load_checks,
)

_RAW_FORMAT_IDS = ("ON_SITE", "REMOTE", "HYBRID", "FIELD_WORK")


def evaluate_analyzer(index: int, turns: List[Dict[str, Any]], checks_by_index: Dict[int, Dict[str, Any]]) -> CheckResult:
    """Слой A: общие инварианты (TG) + hh-специфичный `expect_field_work` (state.field_work_check)."""
    base = _tg_evaluate_analyzer(index, turns, checks_by_index)
    spec = checks_by_index.get(index)
    if not spec or "expect_field_work" not in spec:
        return base
    want = spec["expect_field_work"]
    got = _final_state(turns).get("field_work_check")
    hit = (got == want)
    return CheckResult(
        has_checks=True,
        passed=base.passed and hit,
        details=list(base.details) + [f"field_work_check={want}: {'OK' if hit else f'факт {got}'}"],
    )


def leak_scan(turns: List[Dict[str, Any]], vacancy_info: Dict[str, Any]) -> LeakResult:
    """Слой B (детерминированная часть): в тексте кандидату нет числа вилки и нет сырых id формата.
    Атрибуция: значение уже в instruction Аналитика → вина Аналитика; только в тексте → Интервьюера."""
    details: List[str] = []
    ok = True
    culprit = None

    tokens: List[str] = []
    for bound in ("min_salary", "max_salary"):
        tokens += _salary_variants(vacancy_info.get(bound) or 0)

    for t in turns:
        reply = str(t.get("reply") or "")
        dec = t.get("decision") if isinstance(t, dict) else {}
        instruction = (dec.get("instruction") or "") if isinstance(dec, dict) else ""
        for token in tokens:
            if token and token in reply:
                ok = False
                who = "analyzer" if token in instruction else "interviewer"
                culprit = culprit or who
                details.append(f"утечка вилки «{token}» в ответе (вина: {who})")
                break

    # hh-канарейка: сырой id формата в верхнем регистре не должен попасть кандидату (Интервьюер
    # обязан переформулировать: ON_SITE → «работа в офисе» и т.п.). Регистрозависимо — именно id.
    for t in turns:
        reply = str(t.get("reply") or "")
        dec = t.get("decision") if isinstance(t, dict) else {}
        instruction = (dec.get("instruction") or "") if isinstance(dec, dict) else ""
        for rid in _RAW_FORMAT_IDS:
            if rid in reply:
                ok = False
                who = "analyzer" if rid in instruction else "interviewer"
                culprit = culprit or who
                details.append(f"сырой id формата «{rid}» в ответе (вина: {who})")
                break

    if ok:
        details.append("утечек не найдено")
    return LeakResult(passed=ok, details=details, culprit=culprit)

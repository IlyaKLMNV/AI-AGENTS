"""Детерминированные проверки по трассе split-хода — HH-канал (слой A/B).

Дельта к `screening_split/checks.py` (TG):
- слой A: `evaluate_analyzer` дополнен ключом `expect_field_work` (state.field_work_check) —
  остальные инварианты (salary/format/script_key/asking/event/end/…) общие, берём из TG;
- итог диалога: `evaluate_dialogue` дополнен ключами `formats` (накопленные ответы по форматам) и
  `reasks_zero` — мультиформата в TG нет, проверять там это нечем;
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
    evaluate_dialogue as _tg_evaluate_dialogue,
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


def evaluate_dialogue(turns: List[Dict[str, Any]], expect: Dict[str, Any]) -> CheckResult:
    """Инварианты диалога целиком: общие (TG) + два hh-специфичных ключа `expect`.

    `field_work_check` проверяет общая функция — ключ добавлен в её список полей состояния. Здесь
    остаётся то, чего в tg нет в принципе:

      formats: {ON_SITE: no, HYBRID: yes}  — накопленные ответы ПО ФОРМАТАМ, сверка подмножеством
                                             (перечисленное обязано совпасть, лишнее не мешает);
      reasks_zero: true                    — ни один пункт не переспрашивался. Кооперативному
                                             кандидату мультиформат обязан обходиться без капа:
                                             «в офис не готов» → вопрос про гибрид — это НОВЫЙ
                                             вопрос, а не переспрос, и бюджет жечь не должен;
      greeting_once: "<текст>"             — приветствие приклеено к ПЕРВОМУ сообщению кандидату и
                                             больше нигде. Раннер подставляет сюда текст из
                                             вакансии, в фикстуре стоит `true`. Проверка нужна
                                             отдельно: приветствия нет ни в tg-ядре, ни в контракте
                                             Наблюдателя — его ведёт только код канала, и при
                                             переносе оно теряется молча.
    """
    base = _tg_evaluate_dialogue(turns, expect)
    extra_keys = [k for k in ("formats", "reasks_zero", "greeting_once") if expect.get(k) is not None]
    if not expect or not extra_keys:
        return base

    fstate = _final_state(turns)
    items = list(base.items)

    def _add(rule: str, ok: bool, detail: str) -> None:
        items.append({"rule": rule, "passed": bool(ok), "detail": detail})

    if expect.get("formats") is not None:
        want = {str(k).upper(): str(v) for k, v in (expect["formats"] or {}).items()}
        got = {str(k).upper(): str(v) for k, v in (fstate.get("formats") or {}).items()}
        bad = {k: got.get(k) for k, v in want.items() if got.get(k) != v}
        _add("formats", not bad, f"ответы по форматам: {got}" if not bad
             else f"по форматам ожидалось {want}, факт {got}")

    if expect.get("reasks_zero"):
        reasks = fstate.get("reasks") or {}
        nonzero = {k: v for k, v in reasks.items() if v}
        _add("reasks_zero", not nonzero, "переспросов не было" if not nonzero
             else f"кооперативному кандидату начислили переспросы: {nonzero}")

    greeting = expect.get("greeting_once")
    if isinstance(greeting, str) and greeting.strip():
        head = greeting.strip()
        hits = [i for i, t in enumerate(turns, 1) if head in str(t.get("reply") or "")]
        first_ask = next((i for i, t in enumerate(turns, 1)
                          if (t.get("decision") or {}).get("next_action") == "ask"), None)
        ok = hits == [first_ask] and fstate.get("greeted") is True
        if not hits:
            detail = "приветствие не прозвучало ни разу"
        elif len(hits) > 1:
            detail = f"приветствие повторилось на ходах {hits}"
        elif hits != [first_ask]:
            detail = f"приветствие на ходу {hits[0]}, а первый вопрос — на {first_ask}"
        elif fstate.get("greeted") is not True:
            detail = "текст приклеен, но флаг greeted не выставлен — на следующем ходе повторится"
        else:
            detail = f"приветствие ровно один раз, на первом сообщении (ход {hits[0]})"
        _add("greeting_once", ok, detail)

    passed = all(i["passed"] for i in items)
    details = [f"{'OK ' if i['passed'] else 'FAIL'} · {i['detail']}" for i in items]
    return CheckResult(has_checks=True, passed=passed, details=details, items=items)


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

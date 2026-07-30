"""Детерминированные проверки по трассе split-хода (без LLM).

Дополняет LLM-судью диалога двумя слоями с АТРИБУЦИЕЙ ошибки:
- **Слой A (Аналитик):** для сценариев из scenario_checks.yaml сверяет инварианты Decision/state
  из трассы (salary closed/pending, наличие/отсутствие KO_*, asking, script_key/end). Семантические
  якоря: ассертим ТОЛЬКО то, что правило реально фиксирует (не весь объект) — устойчиво к
  легитимной вариативности. Провал → ошибка Аналитика.
- **Слой B (Интервьюер, детерминированная часть):** leak_scan — в тексте кандидату нет числа вилки
  и нет ссылки при скрытой компании. Атрибуция утечки: число в instruction (из трассы) → вина
  Аналитика; только в тексте Интервьюера → вина Интервьюера.

Порт QA-checks tgApi (scripts/screening_qa/checks.py) + расширение под salary/state-инварианты.
Читает `turns` в форме драйвера: [{candidate, reply, end, decision, state}], где decision — Decision
Аналитика (или форс-словарь кода), state — компактный снимок ({salary, format_check, city, questions, counters}).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_checks(path: Path) -> Dict[int, Dict[str, Any]]:
    """scenario_checks.yaml → {index: spec}. Нет файла — пустой словарь (проверки просто не применяются)."""
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: Dict[int, Dict[str, Any]] = {}
    for e in (data.get("scenarios") or []):
        if isinstance(e, dict) and e.get("index") is not None:
            out[int(e["index"])] = {k: v for k, v in e.items() if k != "index"}
    return out


@dataclass
class CheckResult:
    has_checks: bool
    passed: bool
    details: List[str] = field(default_factory=list)


# ── извлечение из трассы ──────────────────────────────────────────────────────
def _script_keys(turns: List[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    for t in turns:
        dec = t.get("decision") if isinstance(t, dict) else None
        sk = dec.get("script_key") if isinstance(dec, dict) else None
        if sk:
            keys.append(sk)
    return keys


def _askings(turns: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for t in turns:
        dec = t.get("decision") if isinstance(t, dict) else None
        ask = dec.get("asking") if isinstance(dec, dict) else None
        if ask:
            out.append(ask)
    return out


def _events(turns: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for t in turns:
        dec = t.get("decision") if isinstance(t, dict) else None
        ev = dec.get("event") if isinstance(dec, dict) else None
        if ev:
            out.append(ev)
    return out


def _last_asking(turns: List[Dict[str, Any]]) -> Any:
    for t in reversed(turns):
        dec = t.get("decision") if isinstance(t, dict) else None
        if isinstance(dec, dict):
            return dec.get("asking")
    return None


def _last_instruction(turns: List[Dict[str, Any]]) -> Any:
    """instruction Аналитика на последнем ходе, где она есть (None — если только скрипты)."""
    for t in reversed(turns):
        dec = t.get("decision") if isinstance(t, dict) else None
        if isinstance(dec, dict) and dec.get("instruction"):
            return dec.get("instruction")
    return None


def _final_state(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    for t in reversed(turns):
        st = t.get("state") if isinstance(t, dict) else None
        if isinstance(st, dict):
            return st
    return {}


# ── Слой A: инварианты Decision/state Аналитика ───────────────────────────────
def evaluate_analyzer(index: int, turns: List[Dict[str, Any]], checks_by_index: Dict[int, Dict[str, Any]]) -> CheckResult:
    spec = checks_by_index.get(index)
    if not spec:
        return CheckResult(has_checks=False, passed=True, details=[])

    details: List[str] = []
    ok = True
    keys = _script_keys(turns)
    keys_str = ", ".join(keys) or "—"
    askings = _askings(turns)
    ended = bool(turns and turns[-1].get("end"))
    fstate = _final_state(turns)

    if spec.get("expect_script_key"):
        want = spec["expect_script_key"]
        hit = want in keys
        ok = ok and hit
        details.append(f"script_key={want}: {'OK' if hit else f'НЕ сработал (был: {keys_str})'}")

    if spec.get("expect_script_prefix"):
        pref = spec["expect_script_prefix"]
        hit = any(k.startswith(pref) for k in keys)
        ok = ok and hit
        details.append(f"script_key {pref}*: {'OK' if hit else f'НЕ сработал (был: {keys_str})'}")

    if spec.get("expect_no_script_prefix"):
        pref = spec["expect_no_script_prefix"]
        bad = [k for k in keys if k.startswith(pref)]
        hit = not bad
        ok = ok and hit
        details.append(f"нет script_key {pref}*: {'OK' if hit else f'сработал {bad}'}")

    if spec.get("expect_reply_contains"):
        # Подстрока, которая ДОЛЖНА быть в тексте кандидату (регистронезависимо). Скрипт-ответы
        # (next_action=script) код отдаёт верболтим из реестра — детерминированно. Пример: #40 источник
        # контакта → «резюме HH» (конкретный источник, а не fallback «из базы»). Скан по всем ходам.
        want = str(spec["expect_reply_contains"])
        replies = " \n ".join(str(t.get("reply") or "") for t in turns)
        hit = want.lower() in replies.lower()
        ok = ok and hit
        details.append(f"reply содержит «{want}»: {'OK' if hit else 'нет в ответах кандидату'}")

    if "expect_salary" in spec:
        want = spec["expect_salary"]
        got = fstate.get("salary")
        hit = got == want
        ok = ok and hit
        details.append(f"salary={want}: {'OK' if hit else f'факт {got}'}")

    if "expect_format" in spec:
        want = spec["expect_format"]
        got = fstate.get("format_check")
        hit = got == want
        ok = ok and hit
        details.append(f"format_check={want}: {'OK' if hit else f'факт {got}'}")

    if spec.get("expect_asking"):
        want = spec["expect_asking"]
        hit = want in askings
        ok = ok and hit
        askings_str = ", ".join(askings) or "—"
        details.append(f"asking={want} (хоть раз): {'OK' if hit else 'не было (были: ' + askings_str + ')'}")

    if spec.get("expect_last_asking"):
        want = spec["expect_last_asking"]
        got = _last_asking(turns)
        hit = got == want
        ok = ok and hit
        details.append(f"asking на последнем ходу={want}: {'OK' if hit else f'факт {got}'}")

    if spec.get("expect_last_instruction_lacks"):
        # instruction ПОСЛЕДНЕГО хода НЕ должна содержать подстроку (регистронезависимо). Для 29 (F4):
        # на повторном salary_info Аналитик переспрашивает БЕЗ объяснения → в instruction нет «раскрыва».
        want = str(spec["expect_last_instruction_lacks"]).lower()
        instr = (_last_instruction(turns) or "").lower()
        hit = want not in instr
        ok = ok and hit
        details.append(f"instruction последнего хода без «{want}»: {'OK' if hit else 'присутствует (лишнее объяснение)'}")

    if spec.get("expect_event"):
        want = spec["expect_event"]
        evs = _events(turns)
        hit = want in evs
        ok = ok and hit
        evs_str = ", ".join(evs) or "—"
        details.append(f"event={want} (хоть раз): {'OK' if hit else 'не было (были: ' + evs_str + ')'}")

    if "expect_end" in spec:
        want = bool(spec["expect_end"])
        hit = (ended == want)
        ok = ok and hit
        details.append(f"end={want}: {'OK' if hit else f'факт end={ended}'}")

    return CheckResult(has_checks=True, passed=ok, details=details)


# ── Слой B (детерминированная часть): утечка секрета + атрибуция ───────────────
def _salary_variants(n: int) -> List[str]:
    """Строковые формы суммы, которых НЕ должно быть в тексте кандидату."""
    if not n:
        return []
    k = n // 1000
    return [str(n), f"{n:,}".replace(",", " "), f"{k}к", f"{k} тыс", f"{k} тысяч", f"{k} т.р."]


@dataclass
class LeakResult:
    passed: bool
    details: List[str] = field(default_factory=list)
    culprit: Optional[str] = None  # 'analyzer' | 'interviewer' | None


def leak_scan(turns: List[Dict[str, Any]], vacancy_info: Dict[str, Any]) -> LeakResult:
    """Канарейки на утечку в ответах кандидату: нет числа вилки; при скрытой компании — нет ссылки.
    Атрибуция: если число пришло уже в instruction Аналитика → вина Аналитика; иначе — Интервьюера."""
    details: List[str] = []
    ok = True
    culprit: Optional[str] = None

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

    hidden = (vacancy_info.get("company_name") or "").strip().upper() == "СКРЫТО"
    url = ((vacancy_info.get("company_info") or {}).get("vacancy_url") or "").strip()
    if hidden and url:
        for t in turns:
            if url in str(t.get("reply") or ""):
                ok = False
                dec = t.get("decision") if isinstance(t, dict) else {}
                instruction = (dec.get("instruction") or "") if isinstance(dec, dict) else ""
                who = "analyzer" if url in instruction else "interviewer"
                culprit = culprit or who
                details.append(f"утечка ссылки при скрытом поиске «{url}» (вина: {who})")
                break

    if ok:
        details.append("утечек не найдено")
    return LeakResult(passed=ok, details=details, culprit=culprit)

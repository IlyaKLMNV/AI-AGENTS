"""Семантика sourcing_assistant: проверка scoring (passed) по разметке + согласованность комментария.

Новая задача промпта — КОНСЕРВАТИВНАЯ 0/1-оценка резюме по требованиям (1 только при явном подтверждении).
Для golden/generate/offline у нас есть эталон `expect_passed`, поэтому помимо контракта проверяем СМЫСЛ:
- `check_passed_labels(predicted, expect_passed)` (gate): output[i].passed == expect_passed[i];
- `comment_inconsistencies(predicted)` (СИГНАЛ): комментарий противоречит метке (passed=0, но «подтверждено»;
  passed=1, но «не найдено/не подтверждено/частично»). Не gate — комментарий это свободный текст.
Для live backend разметки нет → semantic не применяется (только контракт).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

_CONFIRM_MARKERS = ("подтвержд", "соответств", "есть опыт", "имеет опыт", "владеет", "указан в", "присутств")
_DENY_MARKERS = ("не найден", "не подтвержд", "не указан", "отсутств", "частичн", "нет опыта",
                 "не соответств", "косвен", "явного подтвержд")


def check_passed_labels(predicted: List[Dict[str, Any]], expect_passed: Sequence[int]) -> Tuple[bool, List[str]]:
    """Вернуть (ok, diffs). Сверяем passed по позициям с эталоном expect_passed."""
    diffs: List[str] = []
    n = min(len(predicted), len(expect_passed))
    for i in range(n):
        item = predicted[i]
        actual = item.get("passed") if isinstance(item, dict) else None
        expected = expect_passed[i]
        if actual != expected:
            req = (item.get("requirement") if isinstance(item, dict) else "") or f"#{i}"
            diffs.append(f"passed_mismatch:{req}:expected={expected}:actual={actual}")
    if len(predicted) != len(expect_passed):
        diffs.append(f"length_mismatch:expected={len(expect_passed)}:actual={len(predicted)}")
    return (len(diffs) == 0), diffs


def comment_inconsistencies(predicted: List[Dict[str, Any]]) -> List[str]:
    """СИГНАЛ: комментарий противоречит метке passed (не gate)."""
    out: List[str] = []
    for i, item in enumerate(predicted):
        if not isinstance(item, dict):
            continue
        passed = item.get("passed")
        comment = str(item.get("comment") or "").lower()
        if not comment:
            continue
        has_confirm = any(m in comment for m in _CONFIRM_MARKERS)
        has_deny = any(m in comment for m in _DENY_MARKERS)
        if passed == 1 and has_deny and not has_confirm:
            out.append(f"item[{i}]:passed=1_but_comment_denies")
        elif passed == 0 and has_confirm and not has_deny:
            out.append(f"item[{i}]:passed=0_but_comment_confirms")
    return out

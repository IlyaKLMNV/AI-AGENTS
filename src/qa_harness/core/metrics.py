"""Метрики классификаторов: accuracy, confusion matrix, per-class accuracy.

Доменно-нейтрально: работает со списком пар (target, predicted) и списком меток.
Перенесено из старых `_accuracy`/`_confusion_matrix`/`_per_class_accuracy`
(message/verdict классификаторы) с сохранением округления до 2 знаков * 100.

confusion_matrix возвращается в РАЗРЕЖЕННОМ виде (только ненулевые ячейки) —
как требует docs/REPORT_SCHEMA.md.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

Pair = Tuple[str, str]  # (target, predicted)


def accuracy(pairs: Sequence[Pair]) -> Optional[float]:
    """Доля совпавших пар в процентах (round, 2 знака). None для пустого входа."""
    if not pairs:
        return None
    ok = sum(1 for t, p in pairs if t == p)
    return round(ok / len(pairs) * 100.0, 2)


def per_class_accuracy(pairs: Sequence[Pair], labels: Sequence[str]) -> Dict[str, Optional[float]]:
    """Accuracy по каждому target-классу (None, если класса нет во входе)."""
    out: Dict[str, Optional[float]] = {}
    for label in labels:
        items = [(t, p) for t, p in pairs if t == label]
        out[label] = accuracy(items)
    return out


def confusion_matrix(pairs: Sequence[Pair], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Разреженная матрица ошибок: {target: {predicted: count}} — только ненулевые ячейки."""
    label_set = set(labels)
    m: Dict[str, Dict[str, int]] = {}
    for t, p in pairs:
        if t in label_set and p in label_set:
            m.setdefault(t, {})
            m[t][p] = m[t].get(p, 0) + 1
    return m


def counts(values: Sequence[str], labels: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Счётчик значений (опц. ограниченный множеством labels)."""
    allowed = set(labels) if labels is not None else None
    return dict(Counter(v for v in values if allowed is None or v in allowed))


def split_summary(pairs: Sequence[Pair]) -> Dict[str, object]:
    """Краткая сводка по срезу датасета: {accuracy, total}."""
    return {"accuracy": accuracy(pairs), "total": len(pairs)}


def classification_metrics(pairs: Sequence[Pair], labels: Sequence[str]) -> Dict[str, object]:
    """Готовый блок metrics.classification для отчёта (см. REPORT_SCHEMA §2)."""
    targets = [t for t, _ in pairs]
    preds = [p for _, p in pairs]
    return {
        "labels": list(labels),
        "accuracy": accuracy(pairs),
        "per_class_accuracy": per_class_accuracy(pairs, labels),
        "confusion_matrix": confusion_matrix(pairs, labels),
        "counts_target": counts(targets, labels),
        "counts_predicted": counts(preds, labels),
    }

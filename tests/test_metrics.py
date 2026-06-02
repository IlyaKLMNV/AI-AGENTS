"""Юнит-тесты qa_harness.core.metrics."""

from __future__ import annotations

from qa_harness.core.metrics import (
    accuracy,
    classification_metrics,
    confusion_matrix,
    counts,
    per_class_accuracy,
)


def test_accuracy():
    assert accuracy([("a", "a"), ("a", "b")]) == 50.0
    assert accuracy([("a", "a"), ("b", "b")]) == 100.0
    assert accuracy([]) is None


def test_confusion_matrix_is_sparse():
    cm = confusion_matrix([("a", "a"), ("a", "b"), ("b", "b")], ["a", "b"])
    assert cm == {"a": {"a": 1, "b": 1}, "b": {"b": 1}}  # только ненулевые ячейки


def test_per_class_accuracy():
    pc = per_class_accuracy([("a", "a"), ("a", "b"), ("b", "b")], ["a", "b", "c"])
    assert pc == {"a": 50.0, "b": 100.0, "c": None}


def test_counts():
    assert counts(["a", "a", "b", "x"], ["a", "b"]) == {"a": 2, "b": 1}


def test_classification_metrics_block():
    m = classification_metrics([("a", "a"), ("a", "b")], ["a", "b"])
    assert m["labels"] == ["a", "b"]
    assert m["accuracy"] == 50.0
    assert m["counts_target"] == {"a": 2}
    assert m["counts_predicted"] == {"a": 1, "b": 1}
    assert m["confusion_matrix"] == {"a": {"a": 1, "b": 1}}

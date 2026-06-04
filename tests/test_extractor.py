"""Офлайн-тесты семантической проверки извлечения (golden expect/forbid)."""

from __future__ import annotations

from qa_harness.domain.extractor import check_semantics


def test_semantic_ok():
    ej = {
        "positions": [{"raw_text": "Backend разработчик", "operator": "AND"}],
        "locations": [{"raw_text": "Москва", "operator": "AND"}],
        "experience": {"from": 5, "to": None},
    }
    ok, diffs = check_semantics(ej, expect={"locations": ["Москва"], "experience": {"from": 5}},
                                forbid={"positions": ["Москва", "офис"]})
    assert ok and diffs == []


def test_semantic_missing():
    ej = {"positions": [{"raw_text": "Backend", "operator": "AND"}]}
    ok, diffs = check_semantics(ej, expect={"skills": ["Python"]}, forbid={})
    assert not ok and "missing:skills:Python" in diffs


def test_semantic_misplaced_city_in_positions():
    ej = {"positions": [{"raw_text": "Backend Москва", "operator": "AND"}]}
    ok, diffs = check_semantics(ej, expect={}, forbid={"positions": ["Москва"]})
    assert not ok and "misplaced:positions:Москва" in diffs


def test_semantic_format_left_in_locations():
    ej = {"locations": [{"raw_text": "офис", "operator": "AND"}]}
    ok, diffs = check_semantics(ej, expect={}, forbid={})
    assert not ok and any(d.startswith("format_in_locations:") for d in diffs)


def test_semantic_language_flag_mismatch():
    ej = {"languages": {"russian": False}}
    ok, diffs = check_semantics(ej, expect={"languages": {"russian": True}}, forbid={})
    assert not ok and "lang:russian!=True" in diffs

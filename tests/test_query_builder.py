"""Юнит-тесты domain/query_builder: формат, утечки, golden-семантика, загрузчик."""

from __future__ import annotations

from pathlib import Path

from qa_harness.domain.query_builder import (
    build_query_checks,
    check_query_semantics,
    detect_leakage,
    load_golden,
)

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "one_line_search_query_builder" / "golden.yaml"


# ----- формат -----

def test_format_ok():
    r = build_query_checks("Python AND (Django OR Flask)")
    assert r["ok"] and r["errors"] == [] and r["word_count"] == 5


def test_format_empty():
    r = build_query_checks("   ")
    assert not r["ok"] and "empty_query" in r["errors"]


def test_format_multiline_and_json_like():
    r = build_query_checks('{"q": "x"}\nsecond line')
    assert not r["ok"]
    assert "query_not_single_line" in r["errors"]
    assert "json_like_output" in r["errors"]


# ----- утечки -----

def test_leakage_detects_workformat_salary_process():
    assert "work_format_mentioned" in detect_leakage("Python AND удалённо")
    assert "salary_or_compensation_mentioned" in detect_leakage("Python зарплата 200000 руб")
    assert "application_process_mentioned" in detect_leakage("Python AND портфолио")


def test_leakage_clean_query():
    assert detect_leakage("Python AND (Django OR Flask) AND PostgreSQL") == []


# ----- семантика (golden) -----

def test_semantic_or_group_satisfied_by_any_form():
    ok, diffs = check_query_semantics("React AND TS", expect=[["react"], ["typescript", "ts"]], forbid=[])
    assert ok and diffs == []


def test_semantic_missing_group():
    ok, diffs = check_query_semantics("React only", expect=[["react"], ["redux"]], forbid=[])
    assert not ok and diffs == ["missing:redux"]


def test_semantic_forbidden_present():
    ok, diffs = check_query_semantics("React AND Python", expect=[["react"]], forbid=["python"])
    assert not ok and "forbidden:python" in diffs


def test_semantic_normalizes_case_and_yo_and_bare_string_group():
    # одиночные строки трактуются как группы из одного; нормализация lower + ё→е
    ok, _ = check_query_semantics("ПИТОН и Ёлка", expect=["питон", "елка"], forbid=[])
    assert ok


# ----- загрузчик golden -----

def test_load_golden_fixture():
    cases = load_golden(GOLDEN)
    assert len(cases) >= 5
    assert all(c.name and c.vacancy for c in cases)
    assert any(c.expect for c in cases) and any(c.forbid for c in cases)
    assert all(c.offline_query for c in cases)  # все кейсы реплеятся в --offline

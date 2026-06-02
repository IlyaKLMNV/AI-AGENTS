"""Офлайн-тесты pipeline: contract-валидация, step2-маппинг, кейсы (без сети)."""

from __future__ import annotations

from pathlib import Path

from qa_harness.pipeline import build_step3_payload, make_base_payload, validate_step1_contract
from qa_harness.pipeline.backend_client import classify_step3_error
from qa_harness.pipeline.cases import build_synthetic_cases, load_cases_from_dir, parse_steps
from qa_harness.pipeline.cases import Case
from qa_harness.pipeline.payload import map_level_to_seniority, range_obj_to_str_pair

CASES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "extractor_agent"


# ----- contract -----

def test_contract_valid():
    ej = {
        "positions": [{"raw_text": "Python разработчик", "operator": "AND"}],
        "experience": {"from": 5, "to": None},
        "level": ["senior"],
        "languages": {"russian": True},
    }
    ok, errors, _ = validate_step1_contract(ej, "Python разработчик опыт от 5 лет")
    assert ok and errors == []


def test_contract_unknown_field():
    ok, errors, _ = validate_step1_contract({"foo": 1}, "x")
    assert not ok and any("unknown_field" in e for e in errors)


def test_contract_bad_operator():
    ok, errors, _ = validate_step1_contract({"skills": [{"raw_text": "x", "operator": "XOR"}]}, "x")
    assert not ok and any("operator_invalid" in e for e in errors)


def test_contract_soft_language_warning():
    ok, _e, warnings = validate_step1_contract(
        {"skills": [{"raw_text": "React", "operator": "AND"}], "languages": {"english": True}},
        "нужен react, английский желателен",
    )
    assert ok and "languages_english_soft_requirement_marked_required" in warnings


# ----- step2 mapping -----

def test_build_payload_groups_and_sanitize():
    ej = {
        "positions": [{"raw_text": "Backend", "operator": "AND"}],
        "locations": [{"raw_text": "Москва", "operator": "AND"}, {"raw_text": "офис", "operator": "AND"}],
        "level": ["senior", "lead"],
        "experience": {"from": 5, "to": None},
    }
    p = build_step3_payload(ej, "Backend Москва офис", make_base_payload(limit=20), sanitize_office_geo=True)
    assert p["positions"] == [["all", ["Backend"]]]
    assert p["geos"] == [["all", ["Москва"]]]  # "офис" вычищен
    assert p["seniorityLevels"] == ["Senior", "Lead"]
    assert p["experience"] == ["5", ""]
    assert p["user_phrase"] == "Backend Москва офис"


def test_range_obj_to_str_pair():
    assert range_obj_to_str_pair({"from": 5, "to": None}) == ["5", ""]
    assert range_obj_to_str_pair({"from": 3, "to": 6}) == ["3", "6"]
    assert range_obj_to_str_pair({}) is None
    assert range_obj_to_str_pair({"from": None, "to": None}) is None


def test_map_level_to_seniority():
    assert map_level_to_seniority(["junior", "c-level", "bogus"]) == ["Junior", "C-Level"]
    assert map_level_to_seniority([]) is None


# ----- backend error classification -----

def test_classify_step3_error():
    assert classify_step3_error(400, "Positions or skills or keys must be set") == "insufficient_search_terms"
    assert classify_step3_error(401, "") == "auth_error"
    assert classify_step3_error(403, "forbidden") == "auth_error"
    assert classify_step3_error(500, "oops") == "http_error"


# ----- cases -----

def test_parse_steps():
    assert parse_steps("1") == [1]
    assert parse_steps("1,2,3") == [1, 2, 3]
    assert parse_steps("") == [1, 2, 3]
    assert parse_steps("9,bad") == [1, 2, 3]  # нет валидных -> дефолт


def test_build_synthetic_cases_deterministic():
    base = [Case(name="amp_1", input="Python разработчик из Москвы с опытом от пяти лет", source="real")]
    a = build_synthetic_cases(base, 3, seed=42)
    b = build_synthetic_cases(base, 3, seed=42)
    assert [c.input for c in a] == [c.input for c in b]  # детерминировано по сиду
    assert all(c.source == "syn" for c in a) and len(a) == 3


def test_load_cases_from_dir_reads_real_fixture():
    cases = load_cases_from_dir(CASES_DIR)
    assert len(cases) > 0
    assert all(c.input.strip() for c in cases)
    assert all(c.source == "real" for c in cases)  # amp_* -> real

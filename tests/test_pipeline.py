"""Офлайн-тесты pipeline: contract, step2-маппинг + coverage, парс step1, якоря (без сети)."""

from __future__ import annotations

from pathlib import Path

from qa_harness.pipeline import (
    build_step3_payload,
    make_base_payload,
    mapping_report,
    parse_extractor_json,
    validate_step1_contract,
)
from qa_harness.pipeline.backend_client import classify_step3_error
from qa_harness.pipeline.cases import load_anchors, parse_steps
from qa_harness.pipeline.payload import map_level_to_seniority, range_obj_to_str_pair

ANCHORS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "extractor_agent" / "anchors.yaml"


# ----- contract -----

def test_contract_valid():
    ej = {"positions": [{"raw_text": "Python", "operator": "AND"}], "experience": {"from": 5, "to": None}}
    ok, errors, _ = validate_step1_contract(ej, "Python опыт от 5 лет")
    assert ok and errors == []


def test_contract_unknown_field():
    ok, errors, _ = validate_step1_contract({"foo": 1}, "x")
    assert not ok and any("unknown_field" in e for e in errors)


def test_contract_bad_operator():
    ok, errors, _ = validate_step1_contract({"skills": [{"raw_text": "x", "operator": "XOR"}]}, "x")
    assert not ok and any("operator_invalid" in e for e in errors)


# ----- step2 mapping + coverage -----

def test_build_payload_groups_and_sanitize():
    ej = {
        "positions": [{"raw_text": "Backend", "operator": "AND"}],
        "locations": [{"raw_text": "Москва", "operator": "AND"}, {"raw_text": "офис", "operator": "AND"}],
        "level": ["senior"],
        "experience": {"from": 5, "to": None},
    }
    p = build_step3_payload(ej, "Backend Москва офис", make_base_payload(limit=20), sanitize_office_geo=True)
    assert p["positions"] == [["all", ["Backend"]]]
    assert p["geos"] == [["all", ["Москва"]]]  # "офис" вычищен
    assert p["seniorityLevels"] == ["Senior"]
    assert p["experience"] == ["5", ""]


def test_mapping_report_flags_sanitized_and_unmapped():
    ej = {
        "positions": [{"raw_text": "Backend", "operator": "AND"}],
        "locations": [{"raw_text": "Москва", "operator": "AND"}, {"raw_text": "офис", "operator": "AND"}],
        "business_spheres": [{"raw_text": "финтех", "operator": "AND"}],
    }
    p = build_step3_payload(ej, "x", make_base_payload(), sanitize_office_geo=True)
    rep = mapping_report(ej, p)
    assert rep["dropped"] == []                       # ничего не потеряно тихо
    assert "офис" in rep["sanitized"]                  # формат вычищен из гео — это ОК
    assert rep["unmapped_fields"] == ["business_spheres"]  # сфера извлечена, но не используется


def test_range_and_level_helpers():
    assert range_obj_to_str_pair({"from": 5, "to": None}) == ["5", ""]
    assert range_obj_to_str_pair({}) is None
    assert map_level_to_seniority(["junior", "c-level", "bogus"]) == ["Junior", "C-Level"]


# ----- step1 parse (ok / dirty / invalid) -----

def test_parse_extractor_json_ok():
    obj, status = parse_extractor_json('{"positions": []}')
    assert status == "ok" and obj == {"positions": []}


def test_parse_extractor_json_dirty():
    obj, status = parse_extractor_json('Вот результат: {"positions": []} спасибо')
    assert status == "dirty" and obj == {"positions": []}  # выдернули, но промпт нарушил «только JSON»


def test_parse_extractor_json_invalid():
    obj, status = parse_extractor_json("совсем не json")
    assert status == "invalid" and obj is None


# ----- backend error classification -----

def test_classify_step3_error():
    assert classify_step3_error(400, "Positions or skills or keys must be set") == "insufficient_search_terms"
    assert classify_step3_error(401, "") == "auth_error"
    assert classify_step3_error(500, "oops") == "http_error"


# ----- steps + anchors -----

def test_parse_steps():
    assert parse_steps("1") == [1]
    assert parse_steps("") == [1, 2, 3]
    assert parse_steps("9,bad") == [1, 2, 3]


def test_load_anchors_reads_fixture():
    anchors = load_anchors(ANCHORS)
    assert len(anchors) >= 10
    assert all(a.input.strip() and a.name for a in anchors)
    # у части якорей заданы golden-ожидания
    assert any(a.expect for a in anchors)
    assert any(a.forbid for a in anchors)

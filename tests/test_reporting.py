"""Тесты qa_harness.core.reporting: вычисление summary/failures, схемы, отсутствие BOM."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def _validate(doc: dict, schema_name: str) -> None:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(doc)


def _builder() -> ReportBuilder:
    rb = ReportBuilder(
        runner="message_classifier",
        prompt_under_test={"component": "message_classifier", "prompt_id": "pmpt_x", "prompt_version": "11"},
        run_id="20260102_000000",
        started_at="2026-01-02T00:00:00",
        models={"generator": None, "evaluator": None},
        seed=1234,
        args={"offline": True},
    )
    rb.add_case(
        CaseRecord(
            case_id="regression:a:v1",
            source="regression",
            passed=True,
            inputs={"criterion": "target == acceptance"},
            output={"raw": "acceptance", "parsed": "acceptance"},
            verdict={"evaluator": "label_match", "passed": True, "score": 1.0},
        )
    )
    rb.add_case(
        CaseRecord(
            case_id="regression:b:v1",
            source="regression",
            passed=False,
            inputs={"criterion": "target == no_reason"},
            output={"raw": "acceptance", "parsed": "acceptance"},
            verdict={
                "evaluator": "label_match",
                "passed": False,
                "score": 0.0,
                "reason_codes": ["misclassified->acceptance"],
            },
        )
    )
    rb.set_token_usage({"input": 10, "output": 4, "total": 14})
    return rb


def test_finalize_summary_and_failures():
    rb = _builder()
    metrics_doc, cases_doc = rb.finalize(
        {"classification": {"accuracy": 50.0}}, finished_at="2026-01-02T00:00:01", duration_s=1.0
    )
    assert metrics_doc["summary"] == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "errors": 0,
        "pass_rate": 50.0,
        "token_usage": {"input": 10, "output": 4, "total": 14},
    }
    fi = metrics_doc["failures_index"]
    assert len(fi) == 1 and fi[0]["case_id"] == "regression:b:v1"
    assert fi[0]["reason_codes"] == ["misclassified->acceptance"]
    assert len(cases_doc["cases"]) == 2


def test_docs_match_json_schemas():
    rb = _builder()
    metrics_doc, cases_doc = rb.finalize({"classification": {"accuracy": 50.0}})
    _validate(metrics_doc, "report.metrics.schema.json")
    _validate(cases_doc, "report.cases.schema.json")


def test_write_reports_utf8_no_bom(tmp_path):
    rb = _builder()
    metrics_doc, cases_doc = rb.finalize({})
    mp, cp = write_reports(tmp_path, "message_classifier", "rid123", metrics_doc, cases_doc)
    assert mp.name == "message_classifier_rid123.metrics.json"
    assert cp.name == "message_classifier_rid123.cases.json"
    for p in (mp, cp):
        head = p.read_bytes()[:3]
        assert head != b"\xef\xbb\xbf", f"{p.name} must not start with a UTF-8 BOM"

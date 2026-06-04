"""Offline end-to-end тест раннера message_classifier (без сети, без ключа)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from qa_harness.runners import message_classifier as mc

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def test_offline_run_produces_valid_reports(tmp_path):
    args = mc.build_parser().parse_args(["--offline", "--quiet", "--out-dir", str(tmp_path)])
    out = mc.run(args)

    assert out["metrics"].is_file()
    assert out["cases"].is_file()

    metrics = json.loads(out["metrics"].read_text(encoding="utf-8"))
    cases = json.loads(out["cases"].read_text(encoding="utf-8"))

    assert metrics["meta"]["runner"] == "message_classifier"
    assert metrics["meta"]["prompt_under_test"]["component"] == "message_classifier"
    assert metrics["summary"]["total"] == 8  # 8 размеченных кейсов в фикстуре
    assert metrics["summary"]["passed"] + metrics["summary"]["failed"] == 8
    assert "classification" in metrics["metrics"]
    assert len(cases["cases"]) == 8
    # каждый кейс несёт обязательный criterion
    assert all("criterion" in c["inputs"] for c in cases["cases"])

    # отчёты валидны против опубликованных JSON-схем
    jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / "report.metrics.schema.json").read_text(encoding="utf-8"))
    ).validate(metrics)
    jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / "report.cases.schema.json").read_text(encoding="utf-8"))
    ).validate(cases)

"""Offline end-to-end тест раннера verdict_classifier (без сети, без ключа)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from qa_harness.runners import verdict_classifier as vc

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def test_offline_run_produces_valid_reports(tmp_path):
    args = vc.build_parser().parse_args(["--offline", "--quiet", "--out-dir", str(tmp_path)])
    out = vc.run(args)

    assert out["metrics"].is_file() and out["cases"].is_file()

    metrics = json.loads(out["metrics"].read_text(encoding="utf-8"))
    cases = json.loads(out["cases"].read_text(encoding="utf-8"))

    assert metrics["meta"]["runner"] == "verdict_classifier"
    assert metrics["meta"]["prompt_under_test"]["component"] == "verdict_classifier"
    assert metrics["summary"]["total"] == 4  # 4 размеченных диалога в фикстуре
    assert "classification" in metrics["metrics"]
    # транскрипт диалога структурирован (несколько ходов), criterion обязателен
    assert all("criterion" in c["inputs"] for c in cases["cases"])
    assert any(len(c.get("transcript", [])) >= 2 for c in cases["cases"])

    jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / "report.metrics.schema.json").read_text(encoding="utf-8"))
    ).validate(metrics)
    jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / "report.cases.schema.json").read_text(encoding="utf-8"))
    ).validate(cases)

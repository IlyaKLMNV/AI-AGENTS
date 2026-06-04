"""Offline end-to-end тест раннера one_line_search_query_builder (replay, без сети/ключа)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from qa_harness.runners import one_line_search_query_builder as ol

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def test_offline_run_produces_valid_reports(tmp_path):
    args = ol.build_parser().parse_args(["--offline", "--quiet", "--out-dir", str(tmp_path)])
    out = ol.run(args)

    assert out["metrics"].is_file() and out["cases"].is_file()

    metrics = json.loads(out["metrics"].read_text(encoding="utf-8"))
    cases = json.loads(out["cases"].read_text(encoding="utf-8"))

    assert metrics["meta"]["runner"] == "one_line_search_query_builder"
    assert metrics["meta"]["prompt_under_test"]["component"] == "one_line_search_query_builder"
    assert metrics["summary"]["total"] >= 5
    # offline_query'и golden сконструированы проходить format+leakage+semantic -> всё passed, без инфра
    assert metrics["summary"]["failed"] == 0
    assert metrics["summary"]["errors"] == 0
    assert metrics["summary"]["passed"] == metrics["summary"]["total"]
    assert "builder" in metrics["metrics"]

    # каждый кейс несёт criterion + стадию step1_builder с артефактом-запросом
    assert all("criterion" in c["inputs"] for c in cases["cases"])
    assert all(any(s["name"] == "step1_builder" for s in c.get("stages", [])) for c in cases["cases"])

    # отчёты валидны против опубликованных JSON-схем (первый раннер со stages[], проверяющий схему)
    jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / "report.metrics.schema.json").read_text(encoding="utf-8"))
    ).validate(metrics)
    jsonschema.Draft202012Validator(
        json.loads((SCHEMA_DIR / "report.cases.schema.json").read_text(encoding="utf-8"))
    ).validate(cases)

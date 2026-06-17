"""Единый сборщик отчётов: два файла на прогон (metrics + cases).

Спроектирован против двух непохожих семейств сразу (classifier и search-pipeline,
P0-5): `metrics_extra` свободной формы вмещает classification/judge/backend/
deterministic, а CaseRecord имеет опциональные subjects[]/stages[] для sourcing/
extractor. Агрегаты (summary, failures_index) ВЫЧИСЛЯЮТСЯ здесь из вердиктов, а не
пишутся раннером вручную. Запись — строго UTF-8 без BOM.

См. docs/REPORT_SCHEMA.md и docs/schemas/report.{metrics,cases}.schema.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"


@dataclass
class CaseRecord:
    """Один кейс для cases.json. inputs ОБЯЗАН содержать 'criterion'."""

    case_id: str
    source: str
    passed: bool
    inputs: Dict[str, Any]
    verdict: Dict[str, Any]
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)
    checks: Optional[List[Dict[str, Any]]] = None
    subjects: Optional[List[Dict[str, Any]]] = None
    stages: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "case_id": self.case_id,
            "source": self.source,
            "passed": bool(self.passed),
            "inputs": self.inputs,
            "verdict": self.verdict,
        }
        if self.transcript:
            out["transcript"] = self.transcript
        if self.output:
            out["output"] = self.output
        if self.checks is not None:
            out["checks"] = self.checks
        if self.subjects is not None:
            out["subjects"] = self.subjects
        if self.stages is not None:
            out["stages"] = self.stages
        return out


class ReportBuilder:
    """Копит кейсы/ошибки и собирает (metrics_doc, cases_doc)."""

    def __init__(
        self,
        runner: str,
        prompt_under_test: Dict[str, Any],
        *,
        run_id: str,
        started_at: str,
        models: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        args: Optional[Dict[str, Any]] = None,
        git_commit: Optional[str] = None,
    ) -> None:
        self.runner = runner
        self.prompt_under_test = prompt_under_test
        self.run_id = run_id
        self.started_at = started_at
        self.models = models or {}
        self.seed = seed
        self.args = args or {}
        self.git_commit = git_commit
        self._cases: List[CaseRecord] = []
        self._errors: List[Dict[str, Any]] = []
        self._usage = {"input": 0, "output": 0, "total": 0}

    def add_case(self, rec: CaseRecord) -> None:
        self._cases.append(rec)

    def add_error(self, case_id: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {"case_id": case_id, "message": message}
        if data:
            entry["data"] = data
        self._errors.append(entry)

    def set_token_usage(self, usage: Dict[str, int]) -> None:
        self._usage = {
            "input": int(usage.get("input", 0)),
            "output": int(usage.get("output", 0)),
            "total": int(usage.get("total", 0)),
        }

    def _failures_index(self) -> List[Dict[str, Any]]:
        idx: List[Dict[str, Any]] = []
        for c in self._cases:
            if c.passed:
                continue
            v = c.verdict or {}
            reason_codes = list(v.get("reason_codes") or [])
            severity = (v.get("meta") or {}).get("severity") or "med"
            one_line = v.get("comment") or (reason_codes[0] if reason_codes else "failed")
            entry: Dict[str, Any] = {
                "case_id": c.case_id,
                "reason_codes": reason_codes,
                "severity": severity,
            }
            if c.source:
                entry["source"] = c.source
            if one_line:
                entry["one_line"] = one_line
            if v.get("evaluator"):
                entry["evaluator"] = v["evaluator"]
            idx.append(entry)
        # сортировка: high -> med -> low
        order = {"high": 0, "med": 1, "low": 2}
        idx.sort(key=lambda e: order.get(e["severity"], 1))
        return idx

    def finalize(
        self,
        metrics_extra: Optional[Dict[str, Any]] = None,
        *,
        finished_at: Optional[str] = None,
        duration_s: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        total = len(self._cases)
        passed = sum(1 for c in self._cases if c.passed)
        failed = total - passed
        pass_rate = round(passed / total * 100.0, 2) if total else 0.0

        meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "runner": self.runner,
            "prompt_under_test": self.prompt_under_test,
        }
        if finished_at is not None:
            meta["finished_at"] = finished_at
        if duration_s is not None:
            meta["duration_s"] = duration_s
        if self.models:
            meta["models"] = self.models
        meta["seed"] = self.seed
        if self.git_commit is not None:
            meta["git_commit"] = self.git_commit
        if self.args:
            meta["args"] = self.args

        metrics_doc: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "metrics",
            "meta": meta,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": len(self._errors),
                "pass_rate": pass_rate,
                "token_usage": dict(self._usage),
            },
            "metrics": metrics_extra or {},
        }
        failures = self._failures_index()
        if failures:
            metrics_doc["failures_index"] = failures
        if self._errors:
            metrics_doc["errors_index"] = list(self._errors)

        cases_doc: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "cases",
            "run_id": self.run_id,
            "runner": self.runner,
            "cases": [c.to_dict() for c in self._cases],
        }
        return metrics_doc, cases_doc


def write_reports(
    reports_dir: Path,
    runner: str,
    run_id: str,
    metrics_doc: Dict[str, Any],
    cases_doc: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Записать оба файла (UTF-8 без BOM). Вернуть (metrics_path, cases_path)."""
    out_dir = Path(reports_dir) / runner
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{runner}_{run_id}.metrics.json"
    cases_path = out_dir / f"{runner}_{run_id}.cases.json"
    metrics_path.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    cases_path.write_text(json.dumps(cases_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_path, cases_path

"""Единый сборщик отчётов: три файла на прогон (metrics + cases JSON + человекочитаемый review.md).

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
    # Ключ детерминированной сортировки кейсов в отчёте (напр. (scenario_index, variant)).
    # None → кейс сохраняет порядок вставки. Внутреннее поле, в to_dict не выводится.
    order: Optional[Tuple[Any, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        # Порядок ключей = порядок чтения проверяющего: вход (критерий) → диалог → ВЕРДИКТ.
        # Вердикт идёт ПОСЛЕДНИМ намеренно — его читают после того, как увидели ожидаемое
        # поведение и сам диалог (а не до, как было раньше).
        out: Dict[str, Any] = {
            "case_id": self.case_id,
            "passed": bool(self.passed),
            "source": self.source,
            "inputs": self.inputs,
        }
        if self.transcript:
            out["transcript"] = self.transcript
        if self.output:
            out["output"] = self.output
        if self.subjects is not None:
            out["subjects"] = self.subjects
        if self.stages is not None:
            out["stages"] = self.stages
        if self.checks is not None:
            out["checks"] = self.checks
        out["verdict"] = self.verdict
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
        # Детерминированный порядок отчёта: если у ВСЕХ кейсов задан order — сортируем по нему
        # (независимо от порядка завершения при concurrency). Иначе — порядок вставки (др. раннеры).
        if self._cases and all(c.order is not None for c in self._cases):
            self._cases.sort(key=lambda c: c.order)
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


# ── человекочитаемый рендер (Markdown) ───────────────────────────────────────
# Тот же набор данных, что в metrics/cases, но в порядке чтения проверяющего и так,
# чтобы по одному кейсу ОЖИДАНИЕ → ДИАЛОГ → ВЕРДИКТ шли РЯДОМ, а не были раскиданы.
# Пишется автоматически рядом с JSON (см. write_reports) — отдельный скрипт не нужен.

_ROLE_LABELS = {
    "candidate": "🧑 Кандидат",
    "assistant": "🤖 Ассистент",
    "recruiter": "🧑 Рекрутёр",
    "user": "🧑 Пользователь",
    "system": "⚙️ Система",
}


def _md_quote(text: Any) -> str:
    """Многострочный текст как markdown-блокквота (каждая строка с '> ', \\n сохранены)."""
    s = str(text or "").strip()
    if not s:
        return "> _(пусто)_"
    return "\n".join(("> " + ln) if ln.strip() else ">" for ln in s.splitlines())


def _render_turn(turn: Dict[str, Any]) -> str:
    role = str(turn.get("role") or "?").lower()
    label = _ROLE_LABELS.get(role, f"🔹 {role}")
    end = "  `⟦диалог завершён⟧`" if turn.get("ended") else ""
    return f"**{label}**{end}\n{_md_quote(turn.get('text'))}"


def _render_case(case: Dict[str, Any]) -> str:
    passed = bool(case.get("passed"))
    inputs = case.get("inputs") or {}
    scen = inputs.get("scenario") or {}
    title = f"### {'✅ PASS' if passed else '❌ FAIL'} — `{case.get('case_id', '?')}`"
    if scen.get("name"):
        title += f" · {scen['name']}"
    blocks: List[str] = [title]
    if inputs.get("criterion"):
        blocks.append(f"**Ожидалось:** {inputs['criterion']}")
    if case.get("transcript"):
        blocks.append("**Диалог:**\n\n" + "\n\n".join(_render_turn(t) for t in case["transcript"]))
    out = case.get("output") or {}
    if out.get("parsed") is not None:
        blocks.append(f"**Выход (parsed):** `{json.dumps(out['parsed'], ensure_ascii=False)}`")
    for key, heading in (("subjects", "Субъекты"), ("stages", "Этапы")):
        items = case.get(key)
        if items:
            lines = [f"**{heading}:**"]
            for it in items:
                mark = "✅" if it.get("passed") else "❌"
                lines.append(f"- {mark} {it.get('name') or it.get('id') or ''}")
            blocks.append("\n".join(lines))
    v = case.get("verdict") or {}
    head = f"**Вердикт — {'PASS' if v.get('passed') else 'FAIL'}**"
    judge = v.get("model") or v.get("evaluator")
    if judge:
        head += f" · судья `{judge}`"
    if v.get("turn_ref") is not None:
        head += f" · реплика {v['turn_ref']}"
    vlines = [head] + [f"- {rc}" for rc in (v.get("reason_codes") or [])]
    blocks.append("\n".join(vlines))
    if v.get("comment"):
        blocks.append(f"> 💬 {v['comment']}")
    for ch in (case.get("checks") or []):
        mark = "✅" if ch.get("passed") else "❌"
        line = f"- {mark} `{ch.get('rule')}`"
        if ch.get("detail"):
            line += f": {ch['detail']}"
        blocks.append(line)
    return "\n\n".join(blocks)


def render_review_md(metrics_doc: Dict[str, Any], cases_doc: Dict[str, Any]) -> str:
    """Человекочитаемый Markdown по двум документам отчёта: провалы — первыми, прошедшие — свёрнуты."""
    meta = (metrics_doc or {}).get("meta") or {}
    summary = (metrics_doc or {}).get("summary") or {}
    cases = (cases_doc or {}).get("cases") or []
    runner = meta.get("runner") or (cases_doc or {}).get("runner") or "?"
    run_id = meta.get("run_id") or (cases_doc or {}).get("run_id") or "?"
    put = meta.get("prompt_under_test") or {}

    header = [f"# QA-обзор — {runner} / {run_id}"]
    sub: List[str] = []
    if put:
        comp = put.get("component") or put.get("prompt_id") or "?"
        if put.get("source") == "local":
            # local: реально тестируется директория/версия из пакета prompts (не платформенный номер)
            line = f"промпт `{comp}` (local) `{put.get('local_component') or comp}` {put.get('local_version') or '?'}"
            if put.get("model"):
                line += f" · модель `{put['model']}`"
        else:
            line = f"промпт `{comp}` (stored) v{put.get('prompt_version') or '?'}"
        sub.append(line)
    if (meta.get("models") or {}).get("evaluator"):
        sub.append(f"судья `{meta['models']['evaluator']}`")
    if meta.get("duration_s") is not None:
        sub.append(f"{meta['duration_s']} c")
    if sub:
        header.append("*" + " · ".join(sub) + "*")
    header.append(
        "**Итог:** всего {} · ✅ {} · ❌ {} · ⚠️ инфра-ошибок {} · pass_rate {}%".format(
            summary.get("total", len(cases)), summary.get("passed", 0),
            summary.get("failed", 0), summary.get("errors", 0), summary.get("pass_rate", "—")))

    failed = [c for c in cases if not c.get("passed")]
    passed = [c for c in cases if c.get("passed")]
    errors = (metrics_doc or {}).get("errors_index") or []

    sections = ["\n".join(header)]
    if failed:
        sections.append("## ❌ Провалы ({})\n\n".format(len(failed))
                        + "\n\n---\n\n".join(_render_case(c) for c in failed))
    if errors:
        sections.append("## ⚠️ Инфра-ошибки ({})\n\n".format(len(errors))
                        + "\n".join(f"- `{e.get('case_id', '?')}` — {e.get('message', '')}" for e in errors))
    if passed:
        body = "\n\n---\n\n".join(_render_case(c) for c in passed)
        sections.append("## ✅ Прошли ({})\n\n".format(len(passed))
                        + f"<details><summary>Показать прошедшие кейсы</summary>\n\n{body}\n\n</details>")
    return "\n\n".join(sections) + "\n"


def write_reports(
    reports_dir: Path,
    runner: str,
    run_id: str,
    metrics_doc: Dict[str, Any],
    cases_doc: Dict[str, Any],
    *,
    write_review: bool = True,
) -> Tuple[Path, Path]:
    """Записать отчёты прогона (UTF-8 без BOM): metrics.json + cases.json (для машин) и
    review.md (человекочитаемый). Вернуть (metrics_path, cases_path).

    write_review=False — не писать review.md (используется screening_split: там человекочитаемость
    даёт cases.json, см. docs/screening_split/report_analysis.md §1)."""
    out_dir = Path(reports_dir) / runner
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{runner}_{run_id}.metrics.json"
    cases_path = out_dir / f"{runner}_{run_id}.cases.json"
    metrics_path.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    cases_path.write_text(json.dumps(cases_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if write_review:
        review_path = out_dir / f"{runner}_{run_id}.review.md"
        review_path.write_text(render_review_md(metrics_doc, cases_doc), encoding="utf-8")
    return metrics_path, cases_path

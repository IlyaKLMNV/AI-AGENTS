"""Тонкий раннер message_classifier (эталонный срез новой архитектуры).

Берёт фиксированные размеченные сообщения из regression_cases.json, классифицирует
каждое, судит LabelJudge'ом, считает метрики и пишет two-file отчёт.

Режимы классификации:
- по умолчанию (онлайн): stored-промпт message_classifier (нужен OPENAI_API_KEY);
- --offline: детерминированная эвристика без сети (для CI/демо).

Запуск:
  python -m qa_harness.runners.message_classifier --offline
  python -m qa_harness.runners.message_classifier            # онлайн, реальный промпт

Синтетическая LLM-генерация сообщений (как в старом раннере) будет добавлена
следующим инкрементом; здесь источник кейсов — фиксированная фикстура.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_harness.core import (
    PromptCfg,
    accumulate_usage,
    blank_usage,
    load_cfg,
    resolve_prompt,
    usage_total,
)
from qa_harness.core.metrics import classification_metrics
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.classifiers import HeuristicMessageClassifier, StoredPromptMessageClassifier
from qa_harness.domain.judge import CLASSES, LabelJudge

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGRESSION = REPO_ROOT / "tests" / "fixtures" / "message_classifier" / "regression_cases.json"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "message_classifier"


def load_regression_cases(path: Path) -> List[Dict[str, Any]]:
    """Загрузить фиксированные кейсы {id, target_class, message, scenario, description}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("regression cases JSON must be a list")
    cases: List[Dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        target = str(item.get("target_class") or "").strip().lower()
        message = str(item.get("message") or "").strip()
        if target not in CLASSES:
            raise ValueError(f"case #{i}: invalid target_class {target!r}")
        if not message:
            raise ValueError(f"case #{i}: empty message")
        cases.append(
            {
                "id": str(item.get("id") or f"case_{i}"),
                "target_class": target,
                "message": message,
                "scenario": str(item.get("scenario") or "").strip(),
                "description": str(item.get("description") or "").strip(),
            }
        )
    return cases


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="message_classifier QA runner (new architecture).")
    p.add_argument("--offline", action="store_true", help="Классифицировать офлайн-эвристикой, без сети.")
    p.add_argument("--regression-cases", type=Path, default=DEFAULT_REGRESSION, help="Путь к фикстуре с размеченными сообщениями.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Каталог отчётов (two-file).")
    p.add_argument("--cfg", type=Path, default=None, help="Путь к model.yaml (по умолчанию tests/tools/model.yaml).")
    p.add_argument("--prompt-id", default=None, help="Переопределить prompt_id (онлайн).")
    p.add_argument("--prompt-version", default=None, help="Переопределить prompt_version (онлайн).")
    p.add_argument("--seed", type=int, default=None, help="Сид (попадает в meta отчёта).")
    p.add_argument("--quiet", action="store_true", help="Без прогресс-логов.")
    return p


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    seed = args.seed if args.seed is not None else prompt.seed

    if args.offline:
        classifier: Any = HeuristicMessageClassifier()
    else:
        from qa_harness.core.llm_client import StoredPromptClient

        classifier = StoredPromptMessageClassifier(
            StoredPromptClient(prompt.prompt_id, prompt.prompt_version)
        )

    judge = LabelJudge(CLASSES)
    cases = load_regression_cases(args.regression_cases)

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={
            "component": RUNNER,
            "prompt_id": prompt.prompt_id,
            "prompt_version": prompt.prompt_version,
        },
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": None, "evaluator": None},
        seed=seed,
        args={"offline": bool(args.offline), "source": "regression", "regression_cases": str(args.regression_cases)},
    )

    usage_bucket = blank_usage()
    pairs: List = []

    for c in cases:
        target = c["target_class"]
        message = c["message"]
        predicted, raw, usage = classifier.classify(message)
        accumulate_usage(usage_bucket, usage)
        verdict = judge.evaluate(predicted, target)
        pairs.append((target, predicted))

        rb.add_case(
            CaseRecord(
                case_id=f"regression:{c['id']}:v1",
                source="regression",
                passed=verdict.passed,
                inputs={
                    "criterion": f"target_class == {target}",
                    "scenario": {"name": c["scenario"], "description": c["description"], "target_label": target},
                },
                transcript=[{"turn": 1, "role": "candidate", "text": message}],
                output={"raw": raw, "parsed": predicted},
                verdict=verdict.to_dict(),
            )
        )
        if not args.quiet:
            mark = "ok" if verdict.passed else "MISS"
            print(f"  [{mark}] {c['id']}: target={target} predicted={predicted}")

    rb.set_token_usage(usage_total(usage_bucket))
    finished = datetime.datetime.now()
    metrics_extra = {"classification": classification_metrics(pairs, CLASSES)}
    metrics_doc, cases_doc = rb.finalize(
        metrics_extra,
        finished_at=finished.isoformat(timespec="seconds"),
        duration_s=round((finished - started).total_seconds(), 3),
    )

    metrics_path, cases_path = write_reports(args.out_dir, RUNNER, run_id, metrics_doc, cases_doc)
    summary = metrics_doc["summary"]
    if not args.quiet:
        acc = metrics_doc["metrics"]["classification"]["accuracy"]
        print(
            f"[summary] total={summary['total']} passed={summary['passed']} "
            f"failed={summary['failed']} pass_rate={summary['pass_rate']}% accuracy={acc}%"
        )
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

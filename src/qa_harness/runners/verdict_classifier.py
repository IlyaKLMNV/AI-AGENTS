"""Тонкий раннер verdict_classifier (близнец message_classifier, но на диалогах).

Источники кейсов (--mode):
- regression — фиксированные диалоги из verdict_classifier/regression_cases.json (по умолчанию);
- synthetic  — LLM генерит диалоги под целевой вердикт (passed/failed/deadlock);
- all        — и то, и другое.

Классификация диалога -> вердикт: онлайн stored-промптом verdict_classifier; --offline —
детерминированная эвристика (только --mode regression). Метрики/отчёт — через core.

  python -m qa_harness.runners.verdict_classifier --offline
  python -m qa_harness.runners.verdict_classifier --mode all --dialogs-per-verdict 3 --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qa_harness.core import accumulate_usage, blank_usage, load_cfg, resolve_prompt, usage_total
from qa_harness.core.cdm import load_cdm_files, load_json
from qa_harness.core.metrics import classification_metrics, split_summary
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.classifiers import (
    VERDICTS,
    HeuristicVerdictClassifier,
    StoredPromptVerdictClassifier,
)
from qa_harness.domain.generators import DialogueGenerator, DialogueSpec, pick_verdict_hint
from qa_harness.domain.judge import LabelJudge
from qa_harness.domain.text import CANDIDATE_PREFIX, RECRUITER_PREFIX, speaker_for_line, split_dialogue_lines

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGRESSION = REPO_ROOT / "tests" / "fixtures" / "verdict_classifier" / "regression_cases.json"
DEFAULT_CDM_DIR = REPO_ROOT / "tests" / "fixtures" / "cdm" / "std"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"
RUNNER = "verdict_classifier"

Result = Tuple[str, str, str]  # (source, target, predicted)


def load_regression_cases(path: Path) -> List[Dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("regression cases JSON must be a list")
    cases: List[Dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        target = str(item.get("target_verdict") or "").strip().lower()
        dialogue = str(item.get("dialogue") or "").strip()
        if target not in VERDICTS:
            raise ValueError(f"case #{i}: invalid target_verdict {target!r}")
        if not dialogue:
            raise ValueError(f"case #{i}: empty dialogue")
        cases.append(
            {
                "id": str(item.get("id") or f"case_{i}"),
                "target_verdict": target,
                "dialogue": dialogue,
                "scenario": str(item.get("scenario") or "").strip(),
                "description": str(item.get("description") or "").strip(),
            }
        )
    return cases


def dialogue_to_transcript(dialogue: str) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for line in split_dialogue_lines(dialogue):
        role = speaker_for_line(line)
        if role is None:
            continue
        prefix = RECRUITER_PREFIX if role == "recruiter" else CANDIDATE_PREFIX
        turns.append({"turn": len(turns) + 1, "role": role, "text": line[len(prefix):].strip()})
    return turns


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="verdict_classifier QA runner (new architecture).")
    p.add_argument("--mode", choices=["regression", "synthetic", "all"], default="regression")
    p.add_argument("--offline", action="store_true", help="Офлайн-классификация эвристикой (только --mode regression).")
    p.add_argument("--dialogs-per-verdict", type=int, default=0, help="Сколько диалогов генерить на вердикт (synthetic/all).")
    p.add_argument("--regression-cases", type=Path, default=DEFAULT_REGRESSION)
    p.add_argument("--cdm-dir", type=Path, default=DEFAULT_CDM_DIR)
    p.add_argument("--cdm-count", type=int, default=None)
    p.add_argument("--noise-level", type=int, default=2)
    p.add_argument("--scenario-mode", choices=["random", "cycle"], default="random")
    p.add_argument("--scenario-count-per-verdict", type=int, default=None)
    p.add_argument("--max-attempts-multiplier", type=int, default=30)
    p.add_argument("--dialogue-gen-model", default=None, help=f"Модель генерации (по умолчанию {DEFAULT_GEN_MODEL}).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _run_regression(rb, classifier, judge, args, clf_bucket, results, quiet) -> None:
    for c in load_regression_cases(args.regression_cases):
        target, dialogue = c["target_verdict"], c["dialogue"]
        predicted, raw, usage = classifier.classify(dialogue)
        accumulate_usage(clf_bucket, usage)
        verdict = judge.evaluate(predicted, target)
        results.append(("regression", target, predicted))
        rb.add_case(
            CaseRecord(
                case_id=f"regression:{c['id']}:v1",
                source="regression",
                passed=verdict.passed,
                inputs={
                    "criterion": f"verdict == {target}",
                    "scenario": {"name": c["scenario"], "description": c["description"], "target_label": target},
                },
                transcript=dialogue_to_transcript(dialogue),
                output={"raw": raw, "parsed": predicted},
                verdict=verdict.to_dict(),
            )
        )
        if not quiet:
            print(f"  [{'ok' if verdict.passed else 'MISS'}] regression {c['id']}: target={target} predicted={predicted}")


def _run_synthetic(rb, generator, classifier, judge, args, clf_bucket, results, quiet, seed) -> None:
    rng = random.Random(seed)
    cycle_state: Dict[str, int] = {v: 0 for v in VERDICTS}
    cdm_paths = load_cdm_files(args.cdm_dir, args.cdm_count)

    for target in VERDICTS:
        need = args.dialogs_per_verdict
        attempts_limit = max(need * args.max_attempts_multiplier, 1)
        got = attempts = idx = 0
        while got < need and attempts < attempts_limit:
            attempts += 1
            cdm_path = rng.choice(cdm_paths)
            hint = pick_verdict_hint(target, rng, args.scenario_mode, args.scenario_count_per_verdict, cycle_state)
            try:
                dialogue = generator.generate(DialogueSpec(load_json(cdm_path), target, hint, args.noise_level))
                predicted, raw, usage = classifier.classify(dialogue)
                accumulate_usage(clf_bucket, usage)
            except Exception as e:  # noqa: BLE001 — попытка не удалась (валидация диалога/генерация), пробуем ещё
                rb.add_error(f"synthetic:{target}:attempt{attempts}", repr(e))
                continue
            idx += 1
            got += 1
            verdict = judge.evaluate(predicted, target)
            results.append(("synthetic", target, predicted))
            rb.add_case(
                CaseRecord(
                    case_id=f"synthetic:{target}:v{idx}",
                    source="synthetic",
                    passed=verdict.passed,
                    inputs={
                        "criterion": f"verdict == {target}",
                        "scenario": {"hint": hint, "target_label": target, "cdm_file": cdm_path.name},
                    },
                    transcript=dialogue_to_transcript(dialogue),
                    output={"raw": raw, "parsed": predicted},
                    verdict=verdict.to_dict(),
                )
            )
            if not quiet:
                print(f"  [{'ok' if verdict.passed else 'MISS'}] synthetic {target} {got}/{need}: predicted={predicted}")
        if got < need and not quiet:
            print(f"  [warn] {target}: получено {got}/{need} (лимит попыток исчерпан)")


def run(args: argparse.Namespace) -> Dict[str, Path]:
    if args.offline and args.mode != "regression":
        raise ValueError("--offline поддерживается только с --mode regression (синтетика требует сети).")
    if args.mode in ("synthetic", "all") and args.dialogs_per_verdict <= 0:
        raise ValueError("--dialogs-per-verdict must be > 0 when --mode includes synthetic")

    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    seed = args.seed if args.seed is not None else prompt.seed
    gen_model = args.dialogue_gen_model or DEFAULT_GEN_MODEL

    if args.offline:
        classifier: Any = HeuristicVerdictClassifier()
    else:
        from qa_harness.core.llm_client import StoredPromptClient

        classifier = StoredPromptVerdictClassifier(StoredPromptClient(prompt.prompt_id, prompt.prompt_version))

    judge = LabelJudge(VERDICTS)
    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test={"component": RUNNER, "prompt_id": prompt.prompt_id, "prompt_version": prompt.prompt_version},
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models={"generator": gen_model if args.mode in ("synthetic", "all") else None, "evaluator": None},
        seed=seed,
        args={
            "mode": args.mode,
            "offline": bool(args.offline),
            "dialogs_per_verdict": args.dialogs_per_verdict,
            "noise_level": args.noise_level,
            "scenario_mode": args.scenario_mode,
        },
    )

    clf_bucket = blank_usage()
    gen_bucket = blank_usage()
    results: List[Result] = []

    if args.mode in ("regression", "all"):
        _run_regression(rb, classifier, judge, args, clf_bucket, results, args.quiet)
    if args.mode in ("synthetic", "all"):
        from qa_harness.core.llm_client import ModelClient

        generator = DialogueGenerator(ModelClient(gen_model))
        _run_synthetic(rb, generator, classifier, judge, args, clf_bucket, results, args.quiet, seed)
        accumulate_usage(gen_bucket, generator.usage)

    total_bucket = blank_usage()
    accumulate_usage(total_bucket, clf_bucket)
    accumulate_usage(total_bucket, gen_bucket)
    rb.set_token_usage(usage_total(total_bucket))

    all_pairs = [(t, p) for _, t, p in results]
    classification = classification_metrics(all_pairs, VERDICTS)
    sources = sorted({s for s, _, _ in results})
    if len(sources) > 1:
        classification["by_split"] = {
            s: split_summary([(t, p) for s2, t, p in results if s2 == s]) for s in sources
        }
    metrics_extra = {
        "classification": classification,
        "token_usage_by_role": {"generator": usage_total(gen_bucket), "classifier": usage_total(clf_bucket)},
    }

    finished = datetime.datetime.now()
    metrics_doc, cases_doc = rb.finalize(
        metrics_extra,
        finished_at=finished.isoformat(timespec="seconds"),
        duration_s=round((finished - started).total_seconds(), 3),
    )
    metrics_path, cases_path = write_reports(args.out_dir, RUNNER, run_id, metrics_doc, cases_doc)

    if not args.quiet:
        s = metrics_doc["summary"]
        print(
            f"[summary] mode={args.mode} total={s['total']} passed={s['passed']} failed={s['failed']} "
            f"errors={s['errors']} pass_rate={s['pass_rate']}% accuracy={classification['accuracy']}%"
        )
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

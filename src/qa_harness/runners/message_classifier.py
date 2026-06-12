"""Тонкий раннер message_classifier (эталонный срез новой архитектуры).

Источники кейсов (--mode):
- regression — фиксированные размеченные сообщения из regression_cases.json (по умолчанию);
- synthetic  — LLM генерит свежие сообщения по классам (как старый раннер);
- all        — и то, и другое.

Классификация:
- онлайн (по умолчанию): stored-промпт message_classifier (нужен OPENAI_API_KEY);
- --offline: детерминированная эвристика без сети (только --mode regression).

Каждый кейс судится LabelJudge'ом; метрики (accuracy/confusion/by_split) и two-file
отчёт пишутся через core. Запуск:
  python -m qa_harness.runners.message_classifier --offline
  python -m qa_harness.runners.message_classifier --mode all --messages-per-class 3 --seed 42
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
from qa_harness.domain.classifiers import HeuristicMessageClassifier, StoredPromptMessageClassifier
from qa_harness.domain.generators import (
    CandidateMessageGenerator,
    GenerationPolicy,
    MessageSpec,
    VariantSampler,
    generate_valid,
    pick_scenario_hint,
    validate_candidate_message,
)
from qa_harness.domain.judge import CLASSES, LabelJudge

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGRESSION = REPO_ROOT / "tests" / "fixtures" / "message_classifier" / "regression_cases.json"
DEFAULT_CDM_DIR = REPO_ROOT / "tests" / "fixtures" / "cdm" / "std"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"
RUNNER = "message_classifier"

# (source, target, predicted)
Result = Tuple[str, str, str]


def load_regression_cases(path: Path) -> List[Dict[str, Any]]:
    raw = load_json(path)
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
    p.add_argument("--mode", choices=["regression", "synthetic", "all"], default="regression",
                   help="Источник кейсов (по умолчанию regression).")
    p.add_argument("--offline", action="store_true", help="Офлайн-классификация эвристикой (только --mode regression).")
    p.add_argument("--messages-per-class", type=int, default=0, help="Сколько сообщений генерить на класс (synthetic/all).")
    p.add_argument("--regression-cases", type=Path, default=DEFAULT_REGRESSION)
    p.add_argument("--cdm-dir", type=Path, default=DEFAULT_CDM_DIR, help="CDM-фикстуры для синтетической генерации.")
    p.add_argument("--cdm-count", type=int, default=None, help="Взять первые N CDM.")
    p.add_argument("--noise-level", type=int, default=2, help="0..2, уровень шума в генерации.")
    p.add_argument("--scenario-mode", choices=["random", "cycle"], default="random")
    p.add_argument("--scenario-count-per-class", type=int, default=None)
    p.add_argument("--max-attempts-multiplier", type=int, default=30, help="Лимит попыток на класс = N * множитель.")
    p.add_argument("--message-gen-model", default=None, help=f"Модель генерации (по умолчанию {DEFAULT_GEN_MODEL}).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _run_regression(rb, classifier, judge, args, clf_bucket, results, quiet) -> None:
    for c in load_regression_cases(args.regression_cases):
        target, message = c["target_class"], c["message"]
        predicted, raw, usage = classifier.classify(message)
        accumulate_usage(clf_bucket, usage)
        verdict = judge.evaluate(predicted, target)
        results.append(("regression", target, predicted))
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
        if not quiet:
            print(f"  [{'ok' if verdict.passed else 'MISS'}] regression {c['id']}: target={target} predicted={predicted}")


def _run_synthetic(rb, generator, classifier, judge, args, clf_bucket, results, quiet, seed) -> None:
    """Генерация сообщений с известной меткой через общий движок (generate_valid).

    Каждое сообщение: produce (свежие cdm+hint+стиль на попытку) → validate (класс) → retry. БЕЗ fallback:
    для размеченных данных лучше недодать (error+warn, как старый раннер), чем влить сомнительную метку
    в accuracy. VariantSampler даёт поверхностное разнообразие сверх scenario-hint.
    """
    rng = random.Random(seed)
    cycle_state: Dict[str, int] = {cls: 0 for cls in CLASSES}
    cdm_paths = load_cdm_files(args.cdm_dir, args.cdm_count)
    sampler = VariantSampler(seed)
    variant_counter = 0

    for target in CLASSES:
        need = args.messages_per_class
        for idx in range(1, need + 1):
            meta: Dict[str, Any] = {}
            style = sampler.at(variant_counter)
            variant_counter += 1

            def produce(_attempt, target=target, meta=meta, style=style):
                cdm_path = rng.choice(cdm_paths)
                hint = pick_scenario_hint(target, rng, args.scenario_mode, args.scenario_count_per_class, cycle_state)
                spec = MessageSpec(load_json(cdm_path), target, f"{hint} | {style.hint()}", args.noise_level)
                msg = generator.generate(spec)
                meta.update(cdm_file=cdm_path.name, hint=hint)
                return msg, None

            gr = generate_valid(
                produce,
                lambda m, target=target: validate_candidate_message(target, m),
                policy=GenerationPolicy(max_retries=max(args.max_attempts_multiplier, 1)),
            )
            if not gr.ok:
                rb.add_error(f"synthetic:{target}:v{idx}", "; ".join(gr.errors[-2:]) or "generation failed")
                if not quiet:
                    print(f"  [warn] synthetic {target} v{idx}: генерация не удалась")
                continue

            message = gr.item
            predicted, raw, usage = classifier.classify(message)
            accumulate_usage(clf_bucket, usage)
            verdict = judge.evaluate(predicted, target)
            results.append(("synthetic", target, predicted))
            rb.add_case(
                CaseRecord(
                    case_id=f"synthetic:{target}:v{idx}",
                    source="synthetic",
                    passed=verdict.passed,
                    inputs={
                        "criterion": f"target_class == {target}",
                        "scenario": {"hint": meta.get("hint"), "target_label": target,
                                     "cdm_file": meta.get("cdm_file"), "gen_source": gr.source},
                    },
                    transcript=[{"turn": 1, "role": "candidate", "text": message}],
                    output={"raw": raw, "parsed": predicted},
                    verdict=verdict.to_dict(),
                )
            )
            if not quiet:
                tag = " (fallback)" if gr.source == "fallback" else ""
                print(f"  [{'ok' if verdict.passed else 'MISS'}] synthetic {target} {idx}/{need}: predicted={predicted}{tag}")


def run(args: argparse.Namespace) -> Dict[str, Path]:
    if args.offline and args.mode != "regression":
        raise ValueError("--offline поддерживается только с --mode regression (синтетика требует сети).")
    if args.mode in ("synthetic", "all") and args.messages_per_class <= 0:
        raise ValueError("--messages-per-class must be > 0 when --mode includes synthetic")

    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    seed = args.seed if args.seed is not None else prompt.seed
    gen_model = args.message_gen_model or DEFAULT_GEN_MODEL

    if args.offline:
        classifier: Any = HeuristicMessageClassifier()
    else:
        from qa_harness.core.llm_client import StoredPromptClient

        classifier = StoredPromptMessageClassifier(StoredPromptClient(prompt.prompt_id, prompt.prompt_version))

    judge = LabelJudge(CLASSES)
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
            "messages_per_class": args.messages_per_class,
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

        generator = CandidateMessageGenerator(ModelClient(gen_model))
        _run_synthetic(rb, generator, classifier, judge, args, clf_bucket, results, args.quiet, seed)
        accumulate_usage(gen_bucket, generator.usage)

    total_bucket = blank_usage()
    accumulate_usage(total_bucket, clf_bucket)
    accumulate_usage(total_bucket, gen_bucket)
    rb.set_token_usage(usage_total(total_bucket))

    all_pairs = [(t, p) for _, t, p in results]
    classification = classification_metrics(all_pairs, CLASSES)
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

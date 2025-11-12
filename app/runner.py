from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import time
import sys
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List

import yaml

from adapters.adapters import dialog_to_text, names_from_cdm, to_vacancy_info
from messageLabelGenerator.classifierLLM import ClassifierAssistant
from screeningAssistant.screeningAss import Assistants as ScreeningAssistants
from screening_autofill.screeningAutofill import ScreeningAutofill
from verdict_classifier.chatClassifierLLM import ChatClassifierAssistant

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable
FIXTURES_DIR = ROOT / "tests" / "fixtures"
RAW_DIR = FIXTURES_DIR / "dialogs_raw"
PARSED_DIR = FIXTURES_DIR / "dialogs_parsed"
MSGS_DIR = FIXTURES_DIR / "messages_single" / "unlabeled"
CDM_DIR = FIXTURES_DIR / "cdm"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
DEFAULT_DIALOG_LIMIT = 10


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PARSED_DIR, MSGS_DIR, CDM_DIR, REPORTS_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def run_subprocess(args: List[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


# ---------- Commands ----------

def cmd_gen_fixtures(_: argparse.Namespace) -> None:
    """Convert raw dialogs and ensure CDM fixtures exist."""
    ensure_dirs()
    run_subprocess(
        [
            PYTHON_BIN,
            "-m",
            "tests.tools.convert_dialogs",
            "--in_dir",
            str(RAW_DIR),
            "--out_parsed_dir",
            str(PARSED_DIR),
            "--out_msgs_dir",
            str(MSGS_DIR),
        ]
    )
    for existing in CDM_DIR.glob("*.json"):
        existing.unlink()
    run_subprocess(
        [
            PYTHON_BIN,
            "-m",
            "tests.tools.make_vacancies",
            "--out_dir",
            str(CDM_DIR),
            "--n",
            "3",
        ]
    )
    print("Fixtures generated.")


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _prompt_report(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for component in (
        "message_classifier",
        "screening_assistant",
        "screening_autofill",
        "verdict_classifier",
    ):
        comp_cfg = _component_cfg(cfg, component)
        summary[component] = {
            "id": comp_cfg.get("prompt_id"),
            "version": comp_cfg.get("prompt_version"),
        }
    return summary


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input_token_count") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output_token_count") or 0
        total_tokens = usage.get("total_tokens") or usage.get("token_count")
    else:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or getattr(usage, "input_token_count", None) or 0
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or getattr(usage, "output_token_count", None) or 0
        total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    input_tokens, output_tokens, total_tokens = _extract_usage_numbers(usage)
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens


def classify_message(message_text: str, cfg: Dict[str, Any]) -> tuple[str, Any]:
    mc_cfg = _component_cfg(cfg, "message_classifier")
    assistant = ClassifierAssistant(
        prompt_id=mc_cfg.get("prompt_id"),
        prompt_version=mc_cfg.get("prompt_version"),
    )
    label = assistant.run(message_text).strip()
    return label, getattr(assistant, "last_usage", None)


def run_screening_assistant(
    cdm_path: pathlib.Path,
    transcript_messages: List[str],
    cfg: Dict[str, Any],
) -> tuple[Dict[str, object], Dict[str, int]]:
    cdm = json.loads(cdm_path.read_text(encoding="utf-8"))
    vacancy_info = to_vacancy_info(cdm)
    names = names_from_cdm(cdm)
    sa_cfg = _component_cfg(cfg, "screening_assistant")
    assistant = ScreeningAssistants(
        api_key=os.environ.get("OPENAI_API_KEY"),
        vacancy_info=vacancy_info,
        recruiter_name=names["recruiter_name"],
        candidate_name=names["candidate_name"],
        prompt_id=sa_cfg.get("prompt_id"),
        prompt_version=sa_cfg.get("prompt_version"),
    )

    conversation_id = assistant.create_thread()
    turns: List[tuple[str, str]] = []
    ended = False
    usage_totals = _blank_usage()
    for user_msg in transcript_messages:
        result = assistant.add_message_and_run(conversation_id, user_msg)
        _accumulate_usage(usage_totals, getattr(assistant, "last_usage", None))
        turns.append(("candidate", user_msg))
        response_text = result.response if result and result.response else ""
        if response_text:
            turns.append(("assistant", response_text))
        if result and result.conversation_end:
            ended = True
            break
        if len(turns) > 20:
            break
    return {"conversation_id": conversation_id, "ended": ended, "turns": turns}, usage_totals


def run_autofill(dialog_text: str, cfg: Dict[str, Any]) -> tuple[Dict[str, object], Dict[str, int]]:
    af_cfg = _component_cfg(cfg, "screening_autofill")
    autofiller = ScreeningAutofill(
        prompt_id=af_cfg.get("prompt_id"),
        prompt_version=af_cfg.get("prompt_version"),
    )
    payload = autofiller.run(dialog_text)
    usage = getattr(autofiller, "last_usage", None)
    usage_dict = _blank_usage()
    _accumulate_usage(usage_dict, usage)
    return payload, usage_dict


def run_verdict(dialog_text: str, cfg: Dict[str, Any]) -> tuple[str, Dict[str, int]]:
    verdict_cfg = _component_cfg(cfg, "verdict_classifier")
    classifier = ChatClassifierAssistant(
        prompt_id=verdict_cfg.get("prompt_id"),
        prompt_version=verdict_cfg.get("prompt_version"),
    )
    verdict = classifier.run(dialog_text).strip()
    usage = getattr(classifier, "last_usage", None)
    usage_dict = _blank_usage()
    _accumulate_usage(usage_dict, usage)
    return verdict, usage_dict


def _load_candidate_messages(dialog_path: pathlib.Path, limit: int | None = None) -> List[str]:
    lines = [
        json.loads(line)
        for line in dialog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    messages = [entry["text"] for entry in lines if entry.get("role") == "candidate"]
    if limit is None:
        return messages
    return messages[:limit]


def _write_dialog_report(dialog_report: Dict[str, Any], target_dir: pathlib.Path) -> pathlib.Path:
    filename = dialog_report["dialog_file"].replace(".dialog.jsonl", "") + ".json"
    path = target_dir / filename
    path.write_text(json.dumps(dialog_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_dialog_case(
    dialog_path: pathlib.Path,
    cdm_path: pathlib.Path,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    dialog_text = dialog_to_text(str(dialog_path))
    candidate_msgs = _load_candidate_messages(dialog_path, limit=None)
    classifier_results: List[Dict[str, str]] = []
    modules_status = {
        "message_classifier": False,
        "screening_assistant": False,
        "screening_autofill": False,
        "verdict_classifier": False,
    }
    errors: Dict[str, str] = {}
    module_usage = {
        "message_classifier": _blank_usage(),
        "screening_assistant": _blank_usage(),
        "screening_autofill": _blank_usage(),
        "verdict_classifier": _blank_usage(),
    }

    for msg in candidate_msgs:
        try:
            label, usage = classify_message(msg, cfg)
            classifier_results.append({"text": msg, "label": label})
            _accumulate_usage(module_usage["message_classifier"], usage)
        except Exception as exc:
            errors["message_classifier"] = str(exc)
            break
    else:
        modules_status["message_classifier"] = True

    assistant_result: Dict[str, Any] | None = None
    try:
        assistant_result, assistant_usage = run_screening_assistant(cdm_path, candidate_msgs, cfg)
        modules_status["screening_assistant"] = True
        _accumulate_usage(module_usage["screening_assistant"], assistant_usage)
    except Exception as exc:
        errors["screening_assistant"] = str(exc)

    assistant_turns = []
    assistant_ended = False
    first_message_compliance: bool | None = None
    if assistant_result:
        assistant_turns = [
            {"role": role, "text": text} for role, text in assistant_result["turns"]
        ]
        assistant_ended = bool(assistant_result.get("ended"))
        first_reply = next((turn for turn in assistant_turns if turn["role"] == "assistant"), None)
        if first_reply:
            text = first_reply["text"].lower()
            salary_mentioned = any(keyword in text for keyword in ("зарплат", "вилка", "доход"))
            location_mentioned = any(keyword in text for keyword in ("город", "локац", "формат работы"))
            first_message_compliance = salary_mentioned and location_mentioned

    autofill_payload: Dict[str, Any] | None = None
    try:
        autofill_payload, autofill_usage = run_autofill(dialog_text, cfg)
        modules_status["screening_autofill"] = True
        _accumulate_usage(module_usage["screening_autofill"], autofill_usage)
    except Exception as exc:
        errors["screening_autofill"] = str(exc)

    verdict: str | None = None
    try:
        verdict, verdict_usage = run_verdict(dialog_text, cfg)
        modules_status["verdict_classifier"] = True
        _accumulate_usage(module_usage["verdict_classifier"], verdict_usage)
    except Exception as exc:
        errors["verdict_classifier"] = str(exc)

    success = all(modules_status.values())
    duration = time.perf_counter() - start_time

    total_usage = _blank_usage()
    for usage in module_usage.values():
        _accumulate_usage(total_usage, usage)
    module_usage["total"] = total_usage

    return {
        "dialog_file": dialog_path.name,
        "cdm_file": cdm_path.name,
        "candidate_messages": candidate_msgs,
        "dialog_text": dialog_text,
        "classifier_outputs": classifier_results,
        "assistant_turns": assistant_turns,
        "assistant_ended": assistant_ended,
        "first_message_compliance": first_message_compliance,
        "autofill": autofill_payload,
        "verdict": verdict,
        "modules": modules_status,
        "errors": errors,
        "success": success,
        "duration_sec": duration,
        "token_usage": module_usage,
    }


def _compute_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    success_count = sum(1 for case in cases if case["success"])
    assistant_end = sum(1 for case in cases if case.get("assistant_ended"))
    first_message_checks = [
        case.get("first_message_compliance")
        for case in cases
        if case.get("first_message_compliance") is not None
    ]
    first_message_rate = (
        sum(1 for flag in first_message_checks if flag) / len(first_message_checks)
        if first_message_checks
        else 0.0
    )
    assistant_turn_counts = [
        sum(1 for turn in case["assistant_turns"] if turn["role"] == "assistant")
        for case in cases
    ]
    avg_assistant_turns = (
        sum(assistant_turn_counts) / len(assistant_turn_counts) if assistant_turn_counts else 0.0
    )
    classifier_entries = [entry for case in cases for entry in case["classifier_outputs"]]
    label_counts: Dict[str, int] = {}
    for entry in classifier_entries:
        label = entry.get("label") or "unknown"
        label_counts[label] = label_counts.get(label, 0) + 1

    module_stats: Dict[str, Dict[str, int]] = {}
    for case in cases:
        for module, status in case["modules"].items():
            stat = module_stats.setdefault(module, {"passed": 0, "total": 0})
            stat["total"] += 1
            if status:
                stat["passed"] += 1

    module_success_rate = {
        module: (data["passed"] / data["total"]) if data["total"] else 0.0
        for module, data in module_stats.items()
    }
    token_usage_totals = {
        "message_classifier": _blank_usage(),
        "screening_assistant": _blank_usage(),
        "screening_autofill": _blank_usage(),
        "verdict_classifier": _blank_usage(),
    }
    total_usage = _blank_usage()
    for case in cases:
        case_usage = case.get("token_usage") or {}
        for module in ("message_classifier", "screening_assistant", "screening_autofill", "verdict_classifier"):
            _accumulate_usage(token_usage_totals[module], case_usage.get(module))
        _accumulate_usage(total_usage, case_usage.get("total"))

    classifier_metrics = {
        "samples": 0,
        "accuracy": None,
        "per_class": {},
        "confusion_matrix": {},
        "note": "Ground truth labels not provided; only label distribution is reported.",
    }
    avg_duration = (
        sum(case.get("duration_sec") or 0.0 for case in cases) / total if total else 0.0
    )

    return {
        "total_dialogs": total,
        "pipeline_success_rate": (success_count / total) if total else 0.0,
        "assistant_end_rate": (assistant_end / total) if total else 0.0,
        "first_message_compliance_rate": first_message_rate,
        "average_assistant_turns": avg_assistant_turns,
        "average_duration_sec": avg_duration,
        "classifier_label_distribution": label_counts,
        "classifier_metrics": classifier_metrics,
        "module_success_rate": module_success_rate,
        "token_usage_by_module": token_usage_totals,
        "token_usage_total": total_usage,
    }


def _assign_cdm_files() -> List[pathlib.Path]:
    cdm_files = sorted(CDM_DIR.glob("*.json"))
    if not cdm_files:
        raise FileNotFoundError("No CDM fixtures found. Run gen-fixtures first.")
    return cdm_files


def cmd_unit(args: argparse.Namespace) -> None:
    """Run the entire pipeline for each parsed dialog and store rich reports."""
    ensure_dirs()
    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    dialog_files = sorted(PARSED_DIR.glob("*.dialog.jsonl"))
    if not dialog_files:
        print("No parsed dialogs. Run: python -m app.runner gen-fixtures")
        return
    cdm_files = _assign_cdm_files()

    limit = getattr(args, "limit", DEFAULT_DIALOG_LIMIT) or DEFAULT_DIALOG_LIMIT
    dialog_files = dialog_files[:limit]
    if not dialog_files:
        print("No dialogs to process with the provided limit.")
        return
    total = len(dialog_files)
    started_at = datetime.datetime.now()
    run_id = f"{started_at.strftime('%Y%m%d_%H%M%S')}_n{total}"
    run_dir = RUNS_DIR / run_id
    dialog_dir = run_dir / "dialogs"
    dialog_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    case_refs = []
    failures = []
    for idx, dialog_path in enumerate(dialog_files):
        cdm_path = cdm_files[idx % len(cdm_files)]
        print(f"[{idx + 1}/{total}] Processing {dialog_path.name} (CDM: {cdm_path.name})")
        case = run_dialog_case(dialog_path, cdm_path, cfg)
        cases.append(case)
        report_path = _write_dialog_report(case, dialog_dir)
        status_icon = "✓" if case["success"] else "✗"
        print(f"    {status_icon} modules={case['modules']} assistant_end={case['assistant_ended']}")
        report_rel = str(report_path.relative_to(ROOT))
        case_refs.append(
            {
                "dialog_file": case["dialog_file"],
                "report": report_rel,
                "success": case["success"],
            }
        )
        if not case["success"]:
            failures.append(
                {
                    "dialog_file": case["dialog_file"],
                    "modules_failed": [module for module, status in case["modules"].items() if not status],
                    "errors": case["errors"],
                    "report": report_rel,
                }
            )

    summary = _compute_summary(cases)
    summary["prompts"] = _prompt_report(cfg)
    summary["case_reports"] = case_refs
    summary["failures"] = failures
    summary["run_id"] = run_id
    summary["started_at"] = started_at.isoformat()

    summary_path = run_dir / f"report-{run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Full suite report ->", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for fixtures and assistant runs.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("gen-fixtures")
    unit_parser = subparsers.add_parser("unit")
    unit_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DIALOG_LIMIT,
        help=f"Maximum number of dialogs to process (default: {DEFAULT_DIALOG_LIMIT}).",
    )
    args = parser.parse_args()

    if args.cmd == "gen-fixtures":
        cmd_gen_fixtures(args)
    elif args.cmd == "unit":
        cmd_unit(args)


if __name__ == "__main__":
    main()

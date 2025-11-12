from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time
import sys
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
DIALOG_REPORT_DIR = REPORTS_DIR / "dialogs"


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PARSED_DIR, MSGS_DIR, CDM_DIR, REPORTS_DIR, DIALOG_REPORT_DIR):
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


def classify_message(message_text: str, cfg: Dict[str, Any]) -> str:
    mc_cfg = _component_cfg(cfg, "message_classifier")
    assistant = ClassifierAssistant(
        prompt_id=mc_cfg.get("prompt_id"),
        prompt_version=mc_cfg.get("prompt_version"),
    )
    return assistant.run(message_text).strip()


def run_screening_assistant(
    cdm_path: pathlib.Path,
    transcript_messages: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, object]:
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
    for user_msg in transcript_messages:
        result = assistant.add_message_and_run(conversation_id, user_msg)
        turns.append(("candidate", user_msg))
        response_text = result.response if result and result.response else ""
        if response_text:
            turns.append(("assistant", response_text))
        if result and result.conversation_end:
            ended = True
            break
        if len(turns) > 20:
            break
    return {"conversation_id": conversation_id, "ended": ended, "turns": turns}


def run_autofill(dialog_text: str, cfg: Dict[str, Any]) -> Dict[str, object]:
    af_cfg = _component_cfg(cfg, "screening_autofill")
    autofiller = ScreeningAutofill(
        prompt_id=af_cfg.get("prompt_id"),
        prompt_version=af_cfg.get("prompt_version"),
    )
    return autofiller.run(dialog_text)


def run_verdict(dialog_text: str, cfg: Dict[str, Any]) -> str:
    verdict_cfg = _component_cfg(cfg, "verdict_classifier")
    classifier = ChatClassifierAssistant(
        prompt_id=verdict_cfg.get("prompt_id"),
        prompt_version=verdict_cfg.get("prompt_version"),
    )
    return classifier.run(dialog_text).strip()


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


def _write_dialog_report(dialog_report: Dict[str, Any]) -> pathlib.Path:
    filename = dialog_report["dialog_file"].replace(".dialog.jsonl", "") + ".json"
    path = DIALOG_REPORT_DIR / filename
    path.write_text(json.dumps(dialog_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_dialog_case(
    dialog_path: pathlib.Path,
    cdm_path: pathlib.Path,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
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

    for msg in candidate_msgs:
        try:
            label = classify_message(msg, cfg)
            classifier_results.append({"text": msg, "label": label})
        except Exception as exc:
            errors["message_classifier"] = str(exc)
            break
    else:
        modules_status["message_classifier"] = True

    assistant_result: Dict[str, Any] | None = None
    try:
        assistant_result = run_screening_assistant(cdm_path, candidate_msgs, cfg)
        modules_status["screening_assistant"] = True
    except Exception as exc:
        errors["screening_assistant"] = str(exc)

    assistant_turns = []
    assistant_ended = False
    if assistant_result:
        assistant_turns = [
            {"role": role, "text": text} for role, text in assistant_result["turns"]
        ]
        assistant_ended = bool(assistant_result.get("ended"))

    autofill_payload: Dict[str, Any] | None = None
    try:
        autofill_payload = run_autofill(dialog_text, cfg)
        modules_status["screening_autofill"] = True
    except Exception as exc:
        errors["screening_autofill"] = str(exc)

    verdict: str | None = None
    try:
        verdict = run_verdict(dialog_text, cfg)
        modules_status["verdict_classifier"] = True
    except Exception as exc:
        errors["verdict_classifier"] = str(exc)

    success = all(modules_status.values())

    return {
        "dialog_file": dialog_path.name,
        "cdm_file": cdm_path.name,
        "candidate_messages": candidate_msgs,
        "dialog_text": dialog_text,
        "classifier_outputs": classifier_results,
        "assistant_turns": assistant_turns,
        "assistant_ended": assistant_ended,
        "autofill": autofill_payload,
        "verdict": verdict,
        "modules": modules_status,
        "errors": errors,
        "success": success,
    }


def _compute_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    success_count = sum(1 for case in cases if case["success"])
    assistant_end = sum(1 for case in cases if case.get("assistant_ended"))
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

    return {
        "total_dialogs": total,
        "pipeline_success_rate": (success_count / total) if total else 0.0,
        "assistant_end_rate": (assistant_end / total) if total else 0.0,
        "module_success_rate": module_success_rate,
    }


def _assign_cdm_files() -> List[pathlib.Path]:
    cdm_files = sorted(CDM_DIR.glob("*.json"))
    if not cdm_files:
        raise FileNotFoundError("No CDM fixtures found. Run gen-fixtures first.")
    return cdm_files


def cmd_unit(_: argparse.Namespace) -> None:
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

    cases = []
    case_refs = []
    for idx, dialog_path in enumerate(dialog_files):
        cdm_path = cdm_files[idx % len(cdm_files)]
        case = run_dialog_case(dialog_path, cdm_path, cfg)
        cases.append(case)
        report_path = _write_dialog_report(case)
        case_refs.append(
            {
                "dialog_file": case["dialog_file"],
                "report": str(report_path.relative_to(ROOT)),
                "success": case["success"],
            }
        )

    summary = _compute_summary(cases)
    summary["prompts"] = _prompt_report(cfg)
    summary["case_reports"] = case_refs

    summary_path = REPORTS_DIR / f"run_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Full suite report ->", summary_path)


def cmd_e2e(args: argparse.Namespace) -> None:
    """Alias for cmd_unit to keep backward compatibility."""
    print("[INFO] 'e2e' now runs the full suite (identical to 'unit').")
    cmd_unit(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for fixtures and assistant runs.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("gen-fixtures")
    subparsers.add_parser("unit")
    subparsers.add_parser("e2e")
    args = parser.parse_args()

    if args.cmd == "gen-fixtures":
        cmd_gen_fixtures(args)
    elif args.cmd == "unit":
        cmd_unit(args)
    elif args.cmd == "e2e":
        cmd_e2e(args)


if __name__ == "__main__":
    main()

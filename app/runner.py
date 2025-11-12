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


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PARSED_DIR, MSGS_DIR, CDM_DIR, REPORTS_DIR):
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


def _load_candidate_messages(dialog_path: pathlib.Path, limit: int) -> List[str]:
    lines = [
        json.loads(line)
        for line in dialog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [entry["text"] for entry in lines if entry.get("role") == "candidate"][:limit]


def cmd_unit(_: argparse.Namespace) -> None:
    """Smoke tests for classifier and screening assistant flows."""
    ensure_dirs()
    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    ok, fail = 0, 0
    samples = [
        ("И сколько БФТ Холдинг платит за эту роль?", "acceptance"),
        ("Нет, спасибо", "no_reason"),
        ("Спасибо, уже нашёл работу", "reason_farewell"),
        ("Это к Никите Чугунову?", "human_needed"),
    ]
    for sample_text, expected in samples:
        got = classify_message(sample_text, cfg)
        if got == expected:
            ok += 1
        else:
            fail += 1
            print(f"[MC FAIL] '{sample_text}' -> got={got}, expected={expected}")

    any_dialog = next(PARSED_DIR.glob("*.dialog.jsonl"), None)
    cdm_path = next(CDM_DIR.glob("*.json"), None)
    if any_dialog and cdm_path:
        candidate_msgs = _load_candidate_messages(any_dialog, limit=3)
        if candidate_msgs:
            sa_res = run_screening_assistant(cdm_path, candidate_msgs, cfg)
            if sa_res["ended"]:
                ok += 1
            else:
                fail += 1
                print("[SA FAIL] conversation did not end (no END flag)")
        else:
            print(f"[SA SKIP] No candidate messages in {any_dialog.name}")
    else:
        print("[SA SKIP] Missing parsed dialogs or CDM fixtures. Run gen-fixtures first.")

    print(f"[UNIT] OK={ok} FAIL={fail}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"unit_{int(time.time())}.txt"
    prompt_snapshot = json.dumps(_prompt_report(cfg), ensure_ascii=False)
    report_path.write_text(f"OK={ok} FAIL={fail}\nPROMPTS={prompt_snapshot}", encoding="utf-8")


def cmd_e2e(_: argparse.Namespace) -> None:
    """End-to-end: assistant -> autofill -> verdict -> report."""
    ensure_dirs()
    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)
    any_dialog = next(PARSED_DIR.glob("*.dialog.jsonl"), None)
    cdm_path = next(CDM_DIR.glob("*.json"), None)
    if not any_dialog or not cdm_path:
        print(f"No parsed dialogs or CDM fixtures. Run: {PYTHON_BIN} -m app.runner gen-fixtures")
        return

    candidate_msgs = _load_candidate_messages(any_dialog, limit=6)
    sa_res = run_screening_assistant(cdm_path, candidate_msgs, cfg)
    dialog_text = dialog_to_text(str(any_dialog))
    autofill_payload = run_autofill(dialog_text, cfg)
    verdict = run_verdict(dialog_text, cfg)

    report = {
        "dialog_file": any_dialog.name,
        "assistant_ended": sa_res["ended"],
        "autofill": autofill_payload,
        "verdict": verdict,
        "prompts": _prompt_report(cfg),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"e2e_{int(time.time())}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("E2E report ->", report_path)


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

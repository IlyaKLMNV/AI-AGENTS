"""Unified CLI entry point to exercise project assistants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

from common.settings import OPENAI_API_KEY
from screeningAssistant.screeningAss import (
    Assistants,
    AssistantError as ScreeningAssistantError,
)
from screening_autofill.screeningAutofill import (
    ScreeningAutofill,
    AssistantError as AutofillAssistantError,
)
from verdict_classifier.chatClassifierLLM import (
    ChatClassifierAssistant,
    AssistantError as VerdictAssistantError,
)


DEMO_DIALOG = (
    "Candidate: Hi! Is remote work possible?\n"
    "Recruiter: Hello! Yes, we plan a remote-friendly format.\n"
    "Candidate: I am based in Prague and expect at least 4000 EUR.\n"
)

VACANCY_INFO: Dict[str, object] = {
    "title": "Python Developer",
    "company_name": "Acme Corp",
    "responsibilities": "Build and maintain internal automation services.",
    "work_format": "remote",
    "location": "Remote",
    "min_salary": "3500 EUR",
    "max_salary": "4500 EUR",
    "company_info": {
        "firm_description": "Acme Corp builds data platforms for fintech clients.",
        "vacancy_url": "https://example.com/vacancy/python-developer",
    },
    "questions": (
        "1. What is your current location?\n"
        "2. What salary range are you expecting?\n"
        "3. Do you have production experience with Django or FastAPI?"
    ),
}

RECRUITER_NAME = "Recruiter Smith"
DEFAULT_CANDIDATE_NAME = "Candidate Doe"


def _load_dialog(path: str | None) -> str:
    if not path:
        return DEMO_DIALOG

    dialog_path = Path(path)
    if not dialog_path.is_file():
        raise FileNotFoundError(f"Dialog file not found: {dialog_path}")

    return dialog_path.read_text(encoding="utf-8")


def _ensure_api_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Populate .env before running demos.")


def run_verdict(dialog: str) -> None:
    classifier = ChatClassifierAssistant()
    try:
        result = classifier.run(dialog)
    except VerdictAssistantError as exc:
        raise RuntimeError(f"verdict classifier call failed: {exc}") from exc

    print("\n[VERDICT CLASSIFIER]")
    print(result)


def run_autofill(dialog: str) -> None:
    autofiller = ScreeningAutofill()
    try:
        payload = autofiller.run(dialog)
    except AutofillAssistantError as exc:
        raise RuntimeError(f"screening autofill call failed: {exc}") from exc

    print("\n[SCREENING AUTOFILL]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_screening_assistant(dialog: str, candidate_name: str) -> None:
    assistant = Assistants(
        api_key=OPENAI_API_KEY,
        vacancy_info=VACANCY_INFO,
        recruiter_name=RECRUITER_NAME,
        candidate_name=candidate_name,
    )

    try:
        conversation_id = assistant.create_thread()
        result = assistant.add_message_and_run(conversation_id, dialog)
    except ScreeningAssistantError as exc:
        raise RuntimeError(f"screening assistant call failed: {exc}") from exc

    print("\n[SCREENING ASSISTANT]")
    if result is None:
        print("No response returned.")
        return

    print(f"conversation_end={result.conversation_end}")
    if result.response:
        print(result.response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run demo scenarios against project assistants.")
    parser.add_argument(
        "--demo",
        choices=("verdict", "autofill", "assistant"),
        required=True,
        help="Select which assistant to exercise.",
    )
    parser.add_argument(
        "--file",
        help="Path to a UTF-8 encoded dialog file. Falls back to an internal demo dialog.",
    )
    parser.add_argument(
        "--candidate",
        default=DEFAULT_CANDIDATE_NAME,
        help="Name passed to screening assistant demo.",
    )
    return parser


def main() -> None:
    _ensure_api_key()
    parser = build_parser()
    args = parser.parse_args()

    dialog = _load_dialog(args.file)

    runners: Dict[str, Callable[[str], None]] = {
        "verdict": run_verdict,
        "autofill": run_autofill,
    }

    if args.demo in runners:
        runners[args.demo](dialog)
        return

    run_screening_assistant(dialog, args.candidate)


if __name__ == "__main__":
    main()

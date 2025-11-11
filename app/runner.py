"""Unified CLI entry point to exercise project assistants."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

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

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


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


@dataclass
class CaseData:
    dialog: str
    vacancy_info: Dict[str, object]
    recruiter_name: str
    candidate_name: str


def _resolve_case_path(case: str) -> Path:
    candidate = Path(case)
    if candidate.is_file():
        return candidate

    if not case.endswith(".json"):
        candidate = CASES_DIR / f"{case}.json"
    else:
        candidate = CASES_DIR / case

    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not find JSON case '{case}'. "
        f"Checked absolute path and {CASES_DIR}"
    )


def _load_text_dialog(path: Optional[str]) -> str:
    if not path:
        return DEMO_DIALOG

    dialog_path = Path(path)
    if not dialog_path.is_file():
        raise FileNotFoundError(f"Dialog file not found: {dialog_path}")

    return dialog_path.read_text(encoding="utf-8")


def _infer_names_from_messages(
    messages: Iterable[dict[str, object]],
    recruiter_fallback: str,
    candidate_fallback: str,
) -> tuple[str, str]:
    recruiter = recruiter_fallback or RECRUITER_NAME
    candidate = candidate_fallback or DEFAULT_CANDIDATE_NAME

    for message in messages:
        owner = message.get("is_owner")
        username = str(message.get("username") or "").strip()

        if owner and recruiter == RECRUITER_NAME and username:
            recruiter = username
        if not owner and candidate == DEFAULT_CANDIDATE_NAME and username:
            candidate = username

    return recruiter, candidate


def _format_dialog(messages: Iterable[dict[str, object]], recruiter_name: str, candidate_name: str) -> str:
    lines = []
    for message in messages:
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        owner = message.get("is_owner")
        speaker = recruiter_name if owner else candidate_name
        role = "Recruiter" if owner else "Candidate"
        lines.append(f"{role} ({speaker}): {text}")
    return "\n".join(lines)


def _load_case(case: str) -> CaseData:
    case_path = _resolve_case_path(case)
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    messages: Iterable[dict[str, object]]
    vacancy = VACANCY_INFO.copy()
    recruiter = RECRUITER_NAME
    candidate = DEFAULT_CANDIDATE_NAME

    if isinstance(payload, dict):
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError(f"'messages' must be a list in {case_path}")

        vacancy = payload.get("vacancy_info") or vacancy
        recruiter = payload.get("recruiter_name") or recruiter
        candidate = payload.get("candidate_name") or candidate
    elif isinstance(payload, list):
        messages = payload
    else:
        raise ValueError(f"Unsupported JSON structure in {case_path}")

    recruiter, candidate = _infer_names_from_messages(messages, recruiter, candidate)
    dialog = _format_dialog(messages, recruiter, candidate)

    if not dialog:
        raise ValueError(f"No textual messages found in {case_path}")

    return CaseData(
        dialog=dialog,
        vacancy_info=vacancy,
        recruiter_name=recruiter,
        candidate_name=candidate,
    )


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


def run_screening_assistant(
    dialog: str,
    candidate_name: str,
    recruiter_name: str,
    vacancy_info: Dict[str, object],
) -> None:
    assistant = Assistants(
        api_key=OPENAI_API_KEY,
        vacancy_info=vacancy_info,
        recruiter_name=recruiter_name,
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
        "--case",
        help="Name or path of JSON conversation placed in the cases/ directory.",
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

    if args.case:
        case = _load_case(args.case)
    else:
        dialog = _load_text_dialog(args.file)
        case = CaseData(
            dialog=dialog,
            vacancy_info=VACANCY_INFO,
            recruiter_name=RECRUITER_NAME,
            candidate_name=args.candidate,
        )

    runners: Dict[str, Callable[[str], None]] = {
        "verdict": run_verdict,
        "autofill": run_autofill,
    }

    if args.demo in runners:
        runners[args.demo](case.dialog)
        return

    run_screening_assistant(
        case.dialog,
        case.candidate_name,
        case.recruiter_name,
        case.vacancy_info,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import time
import sys
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List

import yaml
from openai import OpenAI

from adapters.adapters import names_from_cdm, to_vacancy_info
from messageLabelGenerator.classifierLLM import ClassifierAssistant
from screeningAssistant.screeningAss import Assistants as ScreeningAssistants
from screening_autofill.screeningAutofill import ScreeningAutofill
from verdict_classifier.chatClassifierLLM import ChatClassifierAssistant

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable
FIXTURES_DIR = ROOT / "tests" / "fixtures"
RAW_DIR = FIXTURES_DIR / "dialogs_raw"
PARSED_DIR = FIXTURES_DIR / "dialogs_parsed"
CDM_DIR = FIXTURES_DIR / "cdm"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
DEFAULT_DIALOG_LIMIT = 5
MAX_SIMULATION_TURNS = 10
QUESTION_TOKEN_MIN_LENGTH = 4


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PARSED_DIR, CDM_DIR, REPORTS_DIR, RUNS_DIR):
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
            "5",
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


def _question_keywords(question: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", question.lower())
    tokens = {token for token in cleaned.split() if len(token) >= QUESTION_TOKEN_MIN_LENGTH}
    return tokens
class CandidateSimulator:
    def __init__(self, prompt_id: str, prompt_version: str | int | None, display_name: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.prompt_id = prompt_id
        self.prompt_version = str(prompt_version) if prompt_version is not None else None
        self.last_usage: Any = None
        self.display_name = display_name

    def generate(
        self,
        history: List[Dict[str, str]],
        vacancy: Dict[str, Any],
    ) -> str:
        payload_lines = [
            "Dialog history (JSON list of turns, role is assistant/candidate):",
            json.dumps(history, ensure_ascii=False),
            "Vacancy payload:",
            json.dumps(vacancy, ensure_ascii=False),
            "Task: respond на русском, оставаясь в рамках заданного промпта.",
        ]
        payload = "\n".join(payload_lines)
        prompt = {"id": self.prompt_id}
        if self.prompt_version is not None:
            prompt["version"] = self.prompt_version
        response = self.client.responses.create(prompt=prompt, input=payload)
        self.last_usage = getattr(response, "usage", None)
        text = (getattr(response, "output_text", "") or "").strip()
        if not text:
            raise AssistantError("Candidate simulator returned empty response.")
        return text


def classify_message(message_text: str, cfg: Dict[str, Any]) -> tuple[str, Any]:
    mc_cfg = _component_cfg(cfg, "message_classifier")
    assistant = ClassifierAssistant(
        prompt_id=mc_cfg.get("prompt_id"),
        prompt_version=mc_cfg.get("prompt_version"),
    )
    label = assistant.run(message_text).strip()
    return label, getattr(assistant, "last_usage", None)


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


def _write_dialog_report(dialog_report: Dict[str, Any], target_dir: pathlib.Path) -> pathlib.Path:
    filename = dialog_report["dialog_file"].replace(".dialog.jsonl", "") + ".json"
    path = target_dir / filename
    payload = {
        "candidate_profile": dialog_report.get("candidate_profile"),
        "cdm_file": dialog_report.get("cdm_file"),
        "conversation": dialog_report.get("conversation", []),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_dialog_case(
    cdm_path: pathlib.Path,
    cfg: Dict[str, Any],
    candidate_simulator: CandidateSimulator,
    candidate_profile_key: str,
    scenario_name: str,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    cdm = json.loads(cdm_path.read_text(encoding="utf-8"))
    vacancy_info = to_vacancy_info(cdm)
    names = names_from_cdm(cdm)
    classifier_results: List[Dict[str, str]] = []
    modules_status = {
        "message_classifier": False,
        "screening_assistant": False,
        "screening_autofill": False,
        "verdict_classifier": False,
        "candidate_simulator": False,
    }
    errors: Dict[str, str] = {}
    module_usage = {
        "message_classifier": _blank_usage(),
        "screening_assistant": _blank_usage(),
        "screening_autofill": _blank_usage(),
        "verdict_classifier": _blank_usage(),
        "candidate_simulator": _blank_usage(),
    }

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
    template_args = {
        "recruiter_name": names["recruiter_name"],
        "candidate_name": candidate_simulator.display_name or candidate_profile_key.replace("_", " ").title(),
        "company": vacancy_info["company_name"],
        "title": vacancy_info["title"],
        "location": vacancy_info.get("location") or vacancy_info.get("work_format") or "",
    }
    template = cdm.get("first_message_template", "")
    try:
        first_message = template.format(**template_args) if template else ""
    except Exception:
        first_message = template or ""
    if not first_message:
        first_message = (
            f"Здравствуйте, {template_args['candidate_name']}! Это {template_args['recruiter_name']} из "
            f"{template_args['company']}. Подскажите, пожалуйста, где вы сейчас находитесь и "
            "какая net-компенсация будет комфортна?"
        )
    conversation: List[Dict[str, str]] = [{"role": "assistant", "text": first_message}]
    assistant_ended = False

    try:
        for turn in range(MAX_SIMULATION_TURNS):
            try:
                candidate_message = candidate_simulator.generate(conversation, cdm["vacancy"])
                _accumulate_usage(module_usage["candidate_simulator"], getattr(candidate_simulator, "last_usage", None))
                conversation.append({"role": "candidate", "text": candidate_message})
            except Exception as exc:
                errors["candidate_simulator"] = str(exc)
                break

            try:
                label, usage = classify_message(candidate_message, cfg)
                classifier_results.append({"text": candidate_message, "label": label})
                _accumulate_usage(module_usage["message_classifier"], usage)
                modules_status["message_classifier"] = True
            except Exception as exc:
                errors["message_classifier"] = str(exc)
                break

            result = assistant.add_message_and_run(conversation_id, candidate_message)
            _accumulate_usage(module_usage["screening_assistant"], getattr(assistant, "last_usage", None))
            response_text = result.response if result and result.response else ""
            if response_text:
                conversation.append({"role": "assistant", "text": response_text})
            if result and result.conversation_end:
                assistant_ended = True
                break
    except Exception as exc:
        errors.setdefault("screening_assistant", str(exc))
    else:
        modules_status["candidate_simulator"] = True
        modules_status["screening_assistant"] = True

    if "message_classifier" not in errors and classifier_results:
        modules_status["message_classifier"] = True

    first_message_compliance: bool | None = None
    first_reply = conversation[0] if conversation else None
    if first_reply and first_reply.get("text"):
        first_text = first_reply["text"].lower()
        questions_text = cdm["vacancy"].get("questions") or ""
        question_lines = [line.strip() for line in questions_text.splitlines() if line.strip()]
        required_questions = question_lines[:2]
        if required_questions:
            compliance = True
            for q in required_questions:
                keywords = _question_keywords(q)
                if not keywords:
                    continue
                if not any(keyword.lower() in first_text for keyword in keywords):
                    compliance = False
                    break
            first_message_compliance = compliance

    dialog_text = conversation_to_text(conversation)

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

    total_usage = _blank_usage()
    token_usage = {}
    for module_name, usage in module_usage.items():
        token_usage[module_name] = usage.copy()
        _accumulate_usage(total_usage, usage)
    token_usage["total"] = total_usage

    duration = time.perf_counter() - start_time
    success = all(modules_status.values())
    return {
        "dialog_file": scenario_name,
        "cdm_file": cdm_path.name,
        "conversation": conversation,
        "candidate_profile": candidate_profile_key,
        "assistant_ended": assistant_ended,
        "first_message_compliance": first_message_compliance,
        "classifier_outputs": classifier_results,
        "autofill": autofill_payload,
        "verdict": verdict,
        "modules": modules_status,
        "errors": errors,
        "token_usage": token_usage,
        "success": success,
        "duration_sec": duration,
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
    classifier_entries = [
        (case.get("dialog_file"), entry.get("label"))
        for case in cases
        for entry in case["classifier_outputs"]
    ]
    label_counts: Dict[str, int] = {}
    label_dialogs: Dict[str, List[str]] = {}
    for dialog_name, label in classifier_entries:
        label = label or "unknown"
        label_counts[label] = label_counts.get(label, 0) + 1
        label_dialogs.setdefault(label, []).append(dialog_name or "unknown")
    verdict_counts: Dict[str, int] = {}
    verdict_dialogs: Dict[str, List[str]] = {}
    for case in cases:
        verdict_value = case.get("verdict") or "unknown"
        verdict_counts[verdict_value] = verdict_counts.get(verdict_value, 0) + 1
        verdict_dialogs.setdefault(verdict_value, []).append(case.get("dialog_file") or "unknown")

    total_usage = _blank_usage()
    for case in cases:
        case_usage = case.get("token_usage") or {}
        _accumulate_usage(total_usage, case_usage.get("total"))

    avg_duration = (
        sum(case.get("duration_sec") or 0.0 for case in cases) / total if total else 0.0
    )

    return {
        "total_dialogs": total,
        "pipeline_success_rate": (success_count / total) if total else 0.0,
        "assistant_end_rate": (assistant_end / total) if total else 0.0,
        "first_message_compliance_rate": first_message_rate,
        "average_duration_sec": avg_duration,
        "classifier_label_distribution": label_counts,
        "verdict_distribution": verdict_counts,
        "classifier_dialogs": label_dialogs,
        "verdict_dialogs": verdict_dialogs,
        "token_usage_total": total_usage,
    }


def _assign_cdm_files() -> List[pathlib.Path]:
    cdm_files = sorted(CDM_DIR.glob("*.json"))
    if not cdm_files:
        raise FileNotFoundError("No CDM fixtures found. Run gen-fixtures first.")
    return cdm_files

def conversation_to_text(conversation: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for turn in conversation:
        role = "Recruiter" if turn.get("role") == "assistant" else "Candidate"
        text = turn.get("text") or ""
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def cmd_unit(args: argparse.Namespace) -> None:
    """Run the entire pipeline for each vacancy against selected candidate personas."""
    ensure_dirs()
    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    cdm_files = _assign_cdm_files()
    limit = max(1, getattr(args, "limit", DEFAULT_DIALOG_LIMIT))
    vacancies = cdm_files[:limit]
    if not vacancies:
        print("No CDM fixtures. Run: python -m app.runner gen-fixtures")
        return

    sim_cfg = cfg.get("candidate_simulator") or {}
    available_profiles = list(sim_cfg.keys())
    if not available_profiles:
        raise ValueError("candidate_simulator section is empty in config.")
    if getattr(args, "candidate_profiles", None):
        selected_profiles = [p for p in args.candidate_profiles if p in sim_cfg]
    else:
        selected_profiles = available_profiles
    if not selected_profiles:
        raise ValueError("No valid candidate profiles selected.")

    simulators: Dict[str, CandidateSimulator] = {}
    for key in selected_profiles:
        profile_cfg = sim_cfg[key]
        simulators[key] = CandidateSimulator(
            prompt_id=profile_cfg.get("prompt_id"),
            prompt_version=profile_cfg.get("prompt_version"),
            display_name=profile_cfg.get("display_name"),
        )

    started_at = datetime.datetime.now()
    total_cases = len(vacancies) * len(selected_profiles)
    run_id = f"{started_at.strftime('%Y%m%d_%H%M%S')}_n{total_cases}"
    run_dir = RUNS_DIR / run_id
    dialog_dir = run_dir / "dialogs"
    dialog_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    case_refs = []
    failures = []
    case_counter = 0
    for cdm_path in vacancies:
        for profile_key in selected_profiles:
            case_counter += 1
            scenario_name = f"{cdm_path.stem}__{profile_key}"
            print(f"[{case_counter}/{total_cases}] Processing {scenario_name} (CDM: {cdm_path.name})")
            simulator = simulators[profile_key]
            case = run_dialog_case(
                cdm_path,
                cfg,
                simulator,
                profile_key,
                scenario_name,
            )
            cases.append(case)
            report_path = _write_dialog_report(case, dialog_dir)
            status_icon = "✓" if case["success"] else "✗"
            print(f"    {status_icon} modules={case['modules']} assistant_end={case['assistant_ended']}")
            report_rel = str(report_path.relative_to(ROOT))
            case_refs.append(
                {
                    "dialog_file": case["dialog_file"],
                    "candidate_profile": profile_key,
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
        help=f"Maximum number of vacancies to process (default: {DEFAULT_DIALOG_LIMIT}).",
    )
    unit_parser.add_argument(
        "--candidate-profiles",
        nargs="+",
        help="Candidate personas to simulate (keys from candidate_simulator section). Default: all profiles.",
    )
    args = parser.parse_args()

    if args.cmd == "gen-fixtures":
        cmd_gen_fixtures(args)
    elif args.cmd == "unit":
        cmd_unit(args)


if __name__ == "__main__":
    main()



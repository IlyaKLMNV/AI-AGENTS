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
from typing import Any, Dict, List

import yaml
from openai import OpenAI

import traceback

from adapters.adapters import names_from_cdm, to_vacancy_info, to_input_form
from messageLabelGenerator.classifierLLM import ClassifierAssistant, AssistantError
from screeningAssistant.screeningAss import Assistants as ScreeningAssistants
from screening_autofill.screeningAutofill import ScreeningAutofill
from verdict_classifier.chatClassifierLLM import ChatClassifierAssistant

# --- Paths & constants ---

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable
FIXTURES_DIR = ROOT / "tests" / "fixtures"
CDM_DIR = FIXTURES_DIR / "cdm"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
DEFAULT_DIALOG_LIMIT = 5
MAX_SIMULATION_TURNS = 10

# Подключаем генератор Telegram сообщений (адаптер)
TELEGRAM_GEN_DIR = ROOT / "telegramMessageGenerator-main"
if TELEGRAM_GEN_DIR.is_dir():
    sys.path.append(str(TELEGRAM_GEN_DIR))

TELEGRAM_GENERATOR_AVAILABLE: bool = False
TELEGRAM_IMPORT_ERROR: str | None = None

try:
    from telegramGenerator import InputForm as TGInputForm, TelegramMessageGenerator  # type: ignore

    TELEGRAM_GENERATOR_AVAILABLE = True
    TELEGRAM_IMPORT_ERROR = None
except Exception as exc:
    TGInputForm = None  # type: ignore
    TelegramMessageGenerator = None  # type: ignore
    TELEGRAM_GENERATOR_AVAILABLE = False
    TELEGRAM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[telegram_generator] import failed: {TELEGRAM_IMPORT_ERROR}", file=sys.stderr)
    traceback.print_exc()


# ---------- Utils ----------


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    """Создаем только те директории, которые реально нужны пайплайну."""
    for directory in (CDM_DIR, REPORTS_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def run_subprocess(args: List[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


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
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_token_count")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_token_count")
            or 0
        )
        total_tokens = usage.get("total_tokens") or usage.get("token_count")
    else:
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", None) or getattr(
            usage, "token_count", None
        )
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    input_tokens, output_tokens, total_tokens = _extract_usage_numbers(usage)
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens


# --- эвристики для первого касания ---


def _asked_about_location(text: str, vacancy_info: Dict[str, Any]) -> bool:
    """Спрашивает ли сообщение про город / локацию / формат работы."""
    t = text.lower()
    patterns = [
        "город",
        "находитесь",
        "локац",
        "офис",
        "удал",
        "remote",
        "гибрид",
        "формат работы",
        "timezone",
        "часовой пояс",
    ]
    if any(p in t for p in patterns):
        return True
    # прямое упоминание города из вакансии
    location = (vacancy_info.get("location") or "").lower()
    if location and location in t:
        return True
    return False


def _asked_about_salary(text: str) -> bool:
    """Спрашивает ли сообщение про деньги / вилку / компенсацию."""
    t = text.lower()
    patterns = [
        "зп",
        "зарплат",
        "компенсац",
        "доход",
        "оплат",
        "вилка",
        "net",
        "gross",
        "уровень дохода",
        "уровень компенсации",
        "уровень оплаты",
        "ожидания по зп",
        "ожидания по зарплате",
    ]
    return any(p in t for p in patterns)


def _asked_about_experience(text: str, vacancy: Dict[str, Any]) -> bool:
    """
    Спрашивает ли сообщение про опыт - общие слова и ключевые скиллы из вакансии.
    """
    t = text.lower()
    base_patterns = [
        "опыт",
        "experience",
        "сколько лет",
        "стаж",
        "background",
        "работали ли вы",
        "работали раньше",
        "делали ли вы",
        "проекты с",
    ]
    if any(p in t for p in base_patterns):
        return True

    # ключевые навыки из вакансии
    skills = vacancy.get("vacancy_skills") or []
    for skill in skills:
        skill = str(skill).lower()
        if skill and skill in t:
            return True

    # фразы из responsibilities
    responsibilities = (vacancy.get("responsibilities") or "").lower()
    for token in re.split(r"[,\.\n;/]+", responsibilities):
        token = token.strip()
        if len(token) >= 5 and token in t:
            return True

    return False


# ---------- Candidate simulator ----------


class CandidateSimulator:
    def __init__(
        self,
        prompt_id: str,
        prompt_version: str | int | None,
        display_name: str | None = None,
    ):
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
        prompt: Dict[str, Any] = {"id": self.prompt_id}
        if self.prompt_version is not None:
            prompt["version"] = self.prompt_version
        response = self.client.responses.create(prompt=prompt, input=payload)
        self.last_usage = getattr(response, "usage", None)
        text = (getattr(response, "output_text", "") or "").strip()
        if not text:
            raise AssistantError("Candidate simulator returned empty response.")
        return text


# ---------- Module wrappers ----------


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


# ---------- First touch helpers ----------


def _build_classifier_cutoff_reply(
    label: str,
    names: Dict[str, str],
    vacancy_info: Dict[str, Any],
) -> str:
    """Шаблонный ответ рекрутера, если по первому сообщению решили завершить диалог."""
    candidate_name = names.get("candidate_name") or "коллега"
    title = vacancy_info.get("title") or "вакансия"
    company = vacancy_info.get("company_name") or ""

    if label == "reason_farewell":
        return (
            f"Спасибо за ваш ответ, {candidate_name}! "
            "Понимаю вашу позицию и не буду дальше отвлекать. "
            "Если в будущем ситуация изменится, буду рада или рада вернуться к диалогу."
        )
    if label == "no_reason":
        return (
            f"Спасибо, что ответили, {candidate_name}. "
            f"Зафиксирую, что {title} в {company} сейчас неактуальна. "
            "Тогда не буду отвлекать вас дальше, хорошего дня."
        )

    return (
        f"Спасибо за ваш ответ, {candidate_name}! "
        "Если интерес к вакансии изменится, можем вернуться к общению."
    )


def _build_first_message(
    cdm: Dict[str, Any],
    names: Dict[str, str],
    vacancy_info: Dict[str, Any],
    errors: Dict[str, str] | None = None,
) -> tuple[str, str]:
    """
    Возвращает (текст первого сообщения, источник).

    Источники:
      - "telegram_generator"          - успешно сгенерировано telegramMessageGenerator
      - "telegram_generator_error"    - генератор упал или вернул пусто, используется fallback
      - "cdm_template" / "cdm_template_raw" - сообщение из шаблона CDM
      - "fallback_default"            - жестко прошитый текст по умолчанию
    """
    template_args = {
        "recruiter_name": names["recruiter_name"],
        "candidate_name": names["candidate_name"],
        "company": vacancy_info["company_name"],
        "title": vacancy_info["title"],
        "location": vacancy_info.get("location") or vacancy_info.get("work_format") or "",
    }

    # если генератор недоступен - сохраним причину в errors один раз
    if not TELEGRAM_GENERATOR_AVAILABLE and TELEGRAM_IMPORT_ERROR and errors is not None:
        errors.setdefault("telegram_generator_import", TELEGRAM_IMPORT_ERROR)

    telegram_source: str | None = None

    print(
        "[telegram_generator] debug: available="
        f"{TELEGRAM_GENERATOR_AVAILABLE}, TGInputForm={bool(TGInputForm)}, "
        f"TelegramMessageGenerator={bool(TelegramMessageGenerator)}",
        file=sys.stderr,
    )

    # 1) Пробуем telegramMessageGenerator, если доступен
    if TELEGRAM_GENERATOR_AVAILABLE and TGInputForm is not None and TelegramMessageGenerator is not None:
        try:
            form_dict = to_input_form(cdm)
            input_form = TGInputForm(**form_dict)
            api_key = os.environ.get("OPENAI_API_KEY")
            generator = TelegramMessageGenerator(api_key=api_key)
            first_message = (generator.generate_message(input_form) or "").strip()
            if first_message:
                return first_message, "telegram_generator"
            else:
                # генератор вернул пустую строку
                if errors is not None:
                    errors["telegram_generator"] = "empty_response"
                print("[telegram_generator] Empty response from generator", file=sys.stderr)
                return "", "telegram_generator_error"
        except Exception as exc:
            # явный лог
            if errors is not None:
                errors["telegram_generator"] = f"{type(exc).__name__}: {exc}"
            print("[telegram_generator] ERROR while generating first message:", file=sys.stderr)
            print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            # не ломаем пайплайн - продолжаем на CDM-шаблон

    # 2) Шаблон из CDM
    template = cdm.get("first_message_template", "") or ""
    try:
        first_message = template.format(**template_args) if template else ""
        source = "cdm_template"
    except Exception:
        first_message = template
        source = "cdm_template_raw"

    # 3) Жестко прошитый fallback
    if not first_message:
        first_message = (
            f"Здравствуйте, {template_args['candidate_name']}! Это {template_args['recruiter_name']} из "
            f"{template_args['company']}. Подскажите, пожалуйста, где вы сейчас находитесь и "
            "какая net компенсация будет комфортна."
        )
        source = "fallback_default"

    if telegram_source:
        source = telegram_source

    return first_message, source


# ---------- Core pipeline ----------


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
    vacancy = cdm.get("vacancy") or {}
    names = names_from_cdm(cdm)

    # красивое отображаемое имя кандидата
    candidate_display_name = (
        (candidate_simulator.display_name or "").strip()
        or names.get("candidate_name")
        or (candidate_profile_key.replace("_", " ").title() if candidate_profile_key else None)
        or "Кандидат"
    )
    cdm.setdefault("candidate", {})["candidate_name"] = candidate_display_name
    names["candidate_name"] = candidate_display_name

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

    # если генератор не завелся - запишем это в errors
    if not TELEGRAM_GENERATOR_AVAILABLE and TELEGRAM_IMPORT_ERROR:
        errors.setdefault("telegram_generator_import", TELEGRAM_IMPORT_ERROR)

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

    # первое касание
    first_message, first_message_source = _build_first_message(cdm, names, vacancy_info, errors)
    conversation: List[Dict[str, str]] = [{"role": "assistant", "text": first_message}]
    assistant_ended = False

    fm_text = first_message or ""
    first_message_asked_location = _asked_about_location(fm_text, vacancy_info)
    first_message_asked_salary = _asked_about_salary(fm_text)
    first_message_asked_experience = _asked_about_experience(fm_text, vacancy)

    classified_first = False
    cutoff_labels = {"reason_farewell", "no_reason"}
    cutoff_label_for_case: str | None = None

    try:
        for _turn in range(MAX_SIMULATION_TURNS):
            # кандидат
            try:
                candidate_message = candidate_simulator.generate(conversation, cdm["vacancy"])
                _accumulate_usage(
                    module_usage["candidate_simulator"],
                    getattr(candidate_simulator, "last_usage", None),
                )
                conversation.append({"role": "candidate", "text": candidate_message})
            except Exception as exc:
                errors["candidate_simulator"] = str(exc)
                break

            # классификатор - только первое сообщение кандидата
            if not classified_first:
                try:
                    label, usage = classify_message(candidate_message, cfg)
                    classifier_results.append({"text": candidate_message, "label": label})
                    _accumulate_usage(module_usage["message_classifier"], usage)
                    modules_status["message_classifier"] = True
                    classified_first = True

                    # ранний останов по reason_farewell / no_reason
                    if label in cutoff_labels:
                        cutoff_label_for_case = label
                        cutoff_reply = _build_classifier_cutoff_reply(label, names, vacancy_info)
                        conversation.append({"role": "assistant", "text": cutoff_reply})
                        assistant_ended = True
                        # считаем, что screening_assistant "отработал" ветку
                        modules_status["screening_assistant"] = True
                        break
                except Exception as exc:
                    errors["message_classifier"] = str(exc)
                    break

            # ассистент
            result = assistant.add_message_and_run(conversation_id, candidate_message)
            _accumulate_usage(
                module_usage["screening_assistant"],
                getattr(assistant, "last_usage", None),
            )
            modules_status["screening_assistant"] = True

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

    if "message_classifier" not in errors and classifier_results:
        modules_status["message_classifier"] = True

    dialog_text = conversation_to_text(conversation)

    # автофилл
    autofill_payload: Dict[str, Any] | None = None
    try:
        autofill_payload, autofill_usage = run_autofill(dialog_text, cfg)
        modules_status["screening_autofill"] = True
        _accumulate_usage(module_usage["screening_autofill"], autofill_usage)
    except Exception as exc:
        errors["screening_autofill"] = str(exc)

    # вердикт
    verdict: str | None = None
    try:
        # если классификатор первого сообщения дал reason_farewell / no_reason -
        # фиксируем вердикт здесь и не дергаем вердикт-классификатор
        if cutoff_label_for_case in cutoff_labels:
            verdict = "failed"
            verdict_usage = _blank_usage()
        else:
            verdict, verdict_usage = run_verdict(dialog_text, cfg)
        modules_status["verdict_classifier"] = True
        _accumulate_usage(module_usage["verdict_classifier"], verdict_usage)
    except Exception as exc:
        errors["verdict_classifier"] = str(exc)

    # агрегация использования токенов
    total_usage = _blank_usage()
    token_usage: Dict[str, Dict[str, int]] = {}
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
        "first_message_source": first_message_source,
        "first_message_asked_location": first_message_asked_location,
        "first_message_asked_salary": first_message_asked_salary,
        "first_message_asked_experience": first_message_asked_experience,
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

    loc_flags = [bool(case.get("first_message_asked_location")) for case in cases]
    sal_flags = [bool(case.get("first_message_asked_salary")) for case in cases]
    exp_flags = [bool(case.get("first_message_asked_experience")) for case in cases]

    first_message_location_rate = (
        sum(1 for f in loc_flags if f) / len(loc_flags) if loc_flags else 0.0
    )
    first_message_salary_rate = (
        sum(1 for f in sal_flags if f) / len(sal_flags) if sal_flags else 0.0
    )
    first_message_experience_rate = (
        sum(1 for f in exp_flags if f) / len(exp_flags) if exp_flags else 0.0
    )

    source_counts: Dict[str, int] = {}
    for case in cases:
        src = case.get("first_message_source") or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

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
        verdict_dialogs.setdefault(verdict_value, []).append(
            case.get("dialog_file") or "unknown"
        )

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
        "first_message_location_rate": first_message_location_rate,
        "first_message_salary_rate": first_message_salary_rate,
        "first_message_experience_rate": first_message_experience_rate,
        "first_message_source_distribution": source_counts,
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


# ---------- Commands ----------


def cmd_gen_fixtures(_: argparse.Namespace) -> None:
    """Генерация CDM фикстур для вакансий."""
    ensure_dirs()
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
            "10",
        ]
    )
    print("CDM fixtures generated.")


def cmd_unit(args: argparse.Namespace) -> None:
    """Запуск всего пайплайна для каждой вакансии и выбранных профилей кандидатов."""
    ensure_dirs()
    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    print(
        f"[telegram_generator] status at unit start: "
        f"available={TELEGRAM_GENERATOR_AVAILABLE}, import_error={TELEGRAM_IMPORT_ERROR}",
        file=sys.stderr,
    )

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

    cases: List[Dict[str, Any]] = []
    case_refs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    case_counter = 0
    for cdm_path in vacancies:
        for profile_key in selected_profiles:
            case_counter += 1
            scenario_name = f"{cdm_path.stem}__{profile_key}"
            print(
                f"[{case_counter}/{total_cases}] Processing {scenario_name} "
                f"(CDM: {cdm_path.name})"
            )
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
            print(
                f"    {status_icon} modules={case['modules']} "
                f"assistant_end={case['assistant_ended']} "
                f"first_source={case.get('first_message_source')} "
                f"errors={case.get('errors')}",
            )
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
                        "modules_failed": [
                            module for module, status in case["modules"].items() if not status
                        ],
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

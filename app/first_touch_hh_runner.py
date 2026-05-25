from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import yaml
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]
CDM_DIR = ROOT / "tests" / "fixtures" / "cdm" / "hh"
REPORTS_DIR = ROOT / "tests" / "reports" / "first_touch_hh"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"

DEFAULT_LIMIT = 10
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
TEXT_FILE_ENCODING = "utf-8-sig"
DEFAULT_MODE = "all"

SALARY_PATTERNS = (
    r"\bзарплат",
    r"\bвилк",
    r"\bоклад",
    r"\bкомпенсац",
    r"\bgross\b",
    r"\bnet\b",
    r"\bруб",
    r"₽",
    r"\bдмс\b",
)
SOURCE_PATTERNS = (
    r"\blinkedin\b",
    r"\bgithub\b",
    r"\bhh\b",
    r"\bпортфолио\b",
    r"\bрезюме\b",
)
CANDIDATE_FACT_PATTERNS = (
    r"\bу\s+вас\s+опыт\b",
    r"\bваш\s+опыт\b",
    r"\bваши\s+навыки\b",
    r"\bваш\s+стек\b",
    r"\bвы\s+работали\b",
    r"\bрелевантн\w*\s+опыт\b",
    r"\bподходите\b",
    r"\bинтерес\w*\s+вам\s+эта\s+вакан",
)
INFORMAL_PATTERNS = (
    r"\bты\b",
    r"\bтебе\b",
    r"\bтебя\b",
    r"\bтвой\b",
    r"\bтвоя\b",
    r"\bтвою\b",
    r"\bколлега\b",
    r"\bдружище\b",
)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]",
    flags=re.UNICODE,
)


@dataclass
class EvalResult:
    facts_present: Dict[str, bool]
    hallucinated_facts: List[str]
    question_present: bool
    company_rule_ok: bool
    comment: str


@dataclass
class CdmEntry:
    path: pathlib.Path
    data: Dict[str, Any]


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    description: str
    selector: Callable[[Dict[str, Any], Dict[str, str]], bool]
    override_builder: Callable[[Dict[str, Any], Dict[str, str]], Dict[str, str]]


class FirstTouchHHGenerator:
    def __init__(self, prompt_id: str, prompt_version: str | None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def generate_message(self, input_variables: Dict[str, str]) -> str:
        response = self.client.responses.create(
            prompt=self.prompt,
            input=json.dumps(input_variables, ensure_ascii=False),
            text={"format": {"type": "text"}},
        )
        self.last_usage = getattr(response, "usage", None)
        text = _normalize_text(getattr(response, "output_text", "") or "")
        if not text:
            raise RuntimeError("Assistant returned empty message")
        return text


class FirstTouchHHEvaluator:
    def __init__(self, model: str) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.last_usage: Any = None
        self.last_raw_output: str = ""

    def evaluate(self, case: Dict[str, Any], message: str) -> EvalResult:
        response = self.client.responses.create(
            model=self.model,
            input=_eval_payload(case, message),
        )
        self.last_usage = getattr(response, "usage", None)
        raw = _normalize_text(getattr(response, "output_text", "") or "")
        self.last_raw_output = raw
        data = _safe_json_loads(raw)

        expected_facts = dict(case.get("expected_facts") or {})
        facts_present: Dict[str, bool] = {}
        for key in expected_facts:
            if isinstance(data, dict) and isinstance(data.get("facts_present"), dict):
                facts_present[key] = bool(data["facts_present"].get(key))
            else:
                facts_present[key] = False

        hallucinated_facts: List[str] = []
        question_present = "?" in message
        company_rule_ok = True
        comment = ""

        if isinstance(data, dict):
            hallucinated = data.get("hallucinated_facts") or []
            if not isinstance(hallucinated, list):
                hallucinated = [str(hallucinated)]
            hallucinated_facts = [str(x).strip() for x in hallucinated if str(x).strip()]
            if "question_present" in data:
                question_present = bool(data.get("question_present"))
            if "company_rule_ok" in data:
                company_rule_ok = bool(data.get("company_rule_ok"))
            comment = str(data.get("comment") or "").strip()

        return EvalResult(
            facts_present=facts_present,
            hallucinated_facts=hallucinated_facts,
            question_present=question_present,
            company_rule_ok=company_rule_ok,
            comment=comment,
        )


def _log(message: str) -> None:
    print(message)


def ensure_dirs(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding=TEXT_FILE_ENCODING))


def load_cdm_paths(path: pathlib.Path) -> List[pathlib.Path]:
    return sorted(path.glob("*.json"))


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input_token_count") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output_token_count") or 0
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
        total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)

    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    input_tokens, output_tokens, total_tokens = _extract_usage_numbers(usage)
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _normalize_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalized_low(text: str) -> str:
    return _normalize_text(text).lower().replace("ё", "е")


def _extract_json_substring(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    return None


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty json text")
    try:
        return json.loads(text)
    except Exception:
        extracted = _extract_json_substring(text)
        if not extracted:
            raise
        return json.loads(extracted)


def _normalize_for_match(text: str) -> str:
    normalized = _normalized_low(text)
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def _contains_normalized(needle: str, haystack: str) -> bool:
    n = _normalize_for_match(needle)
    h = _normalize_for_match(haystack)
    return bool(n) and n in h


def _has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text or "", flags=re.IGNORECASE) for pattern in patterns)


def _find_forbidden_markers(message: str, markers: List[str]) -> List[str]:
    low = _normalized_low(message)
    return sorted({marker for marker in markers if marker and _normalized_low(marker) in low})


def _find_missing_required_markers(message: str, markers: List[str]) -> List[str]:
    low = _normalized_low(message)
    return [marker for marker in markers if marker and _normalized_low(marker) not in low]


def _count_paragraphs(message: str) -> int:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", _normalize_text(message)) if block.strip()]
    return len(blocks)


def _starts_with_expected_greeting(candidate_name: str, message: str) -> bool:
    stripped = _normalize_text(message)
    if candidate_name:
        return stripped.startswith(f"{candidate_name}, здравствуйте!")
    return stripped.startswith("Здравствуйте!")


def _last_nonempty_line(message: str) -> str:
    lines = [line.strip() for line in _normalize_text(message).split("\n") if line.strip()]
    return lines[-1] if lines else ""


def _text_has_remote_fact(text: str) -> bool:
    low = _normalized_low(text)
    return any(token in low for token in ("удален", "удаленн", "remote"))


def _text_has_accreditation_fact(text: str) -> bool:
    low = _normalized_low(text)
    return "аккредит" in low


def _text_has_hybrid_fact(text: str) -> bool:
    low = _normalized_low(text)
    return "гибрид" in low


def _text_has_office_fact(text: str) -> bool:
    low = _normalized_low(text)
    return "офис" in low


def _is_neutral_company(company_name: str) -> bool:
    normalized = _normalized_low(company_name)
    return not normalized or normalized == "рекрутинговое агентство"


def _company_mode(input_variables: Dict[str, str]) -> str:
    return "neutral" if _is_neutral_company(str(input_variables.get("hiring_company_name") or "")) else "named"


def _contains_invented_company_reference(message: str, input_variables: Dict[str, str]) -> bool:
    company = str(input_variables.get("hiring_company_name") or "").strip()
    if company and company.lower() != "рекрутинговое агентство":
        return False
    low = _normalize_text(message).lower()
    if "рекрутинговое агентство" in low:
        return True
    return bool(re.search(r"\bв компании\s+[A-ZА-ЯЁ][\w\-]+", message or ""))


def _standard_forbidden_markers() -> List[str]:
    return ["linkedin", "github", "hh", "зарплат", "вилка", "дмс"]


def _sanitize_vacancy_text(raw_text: str, *, drop_company_descriptors: bool = False) -> str:
    lines = [line.strip() for line in _normalize_text(raw_text).split("\n") if line.strip()]
    cleaned: List[str] = []
    for line in lines:
        low = _normalized_low(line)

        if re.match(r"^https?://", line, flags=re.IGNORECASE):
            continue
        if any(token in low for token in ("если кандидат", "нужно проверить", "следует проверить", "проверить готовность")):
            continue
        if drop_company_descriptors and "компан" in low:
            continue
        if "удаленная роль" in low and "привязана к москве" in low:
            cleaned.append("Удаленный формат работы, позиция привязана к Москве.")
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _build_input_variables_from_cdm(cdm: Dict[str, Any]) -> Dict[str, str]:
    vacancy = dict((cdm or {}).get("vacancy") or {})
    candidate = dict((cdm or {}).get("candidate") or {})
    return {
        "candidate_name": str(candidate.get("candidate_name") or "").strip(),
        "recruiter_name": str(candidate.get("recruiter_name") or "").strip(),
        "hiring_company_name": str(vacancy.get("company_name") or "").strip(),
        "vacancy_name": str(vacancy.get("title") or "").strip(),
        "vacancy_text": _sanitize_vacancy_text(str(vacancy.get("vacancy_description") or "").strip()),
    }


def _build_expected_facts(input_variables: Dict[str, str]) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    vacancy_name = str(input_variables.get("vacancy_name") or "").strip()
    company_name = str(input_variables.get("hiring_company_name") or "").strip()
    vacancy_text = str(input_variables.get("vacancy_text") or "").strip()

    if vacancy_name:
        expected["vacancy_name"] = vacancy_name
    if company_name and not _is_neutral_company(company_name):
        expected["company_name"] = company_name
    if _text_has_remote_fact(vacancy_text):
        expected["remote"] = "удаленный формат работы"
    if _text_has_accreditation_fact(vacancy_text):
        expected["accreditation"] = "IT-аккредитация"
    return expected


def _build_required_markers(input_variables: Dict[str, str]) -> List[str]:
    vacancy_text = str(input_variables.get("vacancy_text") or "")
    required: List[str] = []
    if _text_has_remote_fact(vacancy_text):
        required.append("удален")
    if _text_has_accreditation_fact(vacancy_text):
        required.append("аккредит")
    return required


def _build_forbidden_markers(input_variables: Dict[str, str]) -> List[str]:
    vacancy_text = str(input_variables.get("vacancy_text") or "")
    forbidden = _standard_forbidden_markers()
    if not _text_has_remote_fact(vacancy_text):
        forbidden.extend(["удален", "remote"])
    if not _text_has_accreditation_fact(vacancy_text):
        forbidden.append("аккредит")
    if _is_neutral_company(str(input_variables.get("hiring_company_name") or "")):
        forbidden.append("рекрутинговое агентство")
    return sorted(set(forbidden))


def _scenario_base_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    return True


def _scenario_identity_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {}


def _scenario_candidate_name_empty_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    return bool(base_vars.get("candidate_name"))


def _scenario_candidate_name_empty_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {"candidate_name": ""}


def _scenario_recruiter_name_empty_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    return bool(base_vars.get("recruiter_name"))


def _scenario_recruiter_name_empty_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {"recruiter_name": ""}


def _scenario_company_name_empty_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    return bool(base_vars.get("hiring_company_name"))


def _scenario_company_name_empty_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {
        "hiring_company_name": "",
        "vacancy_text": _sanitize_vacancy_text(str(base_vars.get("vacancy_text") or ""), drop_company_descriptors=True),
    }


def _scenario_recruiting_agency_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    return bool(base_vars.get("hiring_company_name")) and not _is_neutral_company(str(base_vars.get("hiring_company_name") or ""))


def _scenario_recruiting_agency_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {
        "hiring_company_name": "рекрутинговое агентство",
        "vacancy_text": _sanitize_vacancy_text(str(base_vars.get("vacancy_text") or ""), drop_company_descriptors=True),
    }


def _scenario_sparse_vacancy_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    return True


def _scenario_sparse_vacancy_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {"vacancy_text": "Развитие внутренних сервисов и участие в новых задачах команды."}


def _scenario_hybrid_no_remote_selector(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> bool:
    vacancy_text = str(base_vars.get("vacancy_text") or "")
    return _text_has_hybrid_fact(vacancy_text) and not _text_has_remote_fact(vacancy_text)


def _scenario_hybrid_no_remote_override(cdm: Dict[str, Any], base_vars: Dict[str, str]) -> Dict[str, str]:
    return {}


SCENARIO_SPECS: List[ScenarioSpec] = [
    ScenarioSpec(
        id="base",
        description="Base HH case from cdm without input overrides.",
        selector=_scenario_base_selector,
        override_builder=_scenario_identity_override,
    ),
    ScenarioSpec(
        id="candidate_name_empty",
        description="Override candidate_name to empty and require neutral greeting.",
        selector=_scenario_candidate_name_empty_selector,
        override_builder=_scenario_candidate_name_empty_override,
    ),
    ScenarioSpec(
        id="recruiter_name_empty",
        description="Override recruiter_name to empty and remove recruiter intro.",
        selector=_scenario_recruiter_name_empty_selector,
        override_builder=_scenario_recruiter_name_empty_override,
    ),
    ScenarioSpec(
        id="company_name_empty",
        description="Override company name to empty and require neutral company handling.",
        selector=_scenario_company_name_empty_selector,
        override_builder=_scenario_company_name_empty_override,
    ),
    ScenarioSpec(
        id="recruiting_agency_company",
        description="Override company name to 'рекрутинговое агентство' and require neutral company handling.",
        selector=_scenario_recruiting_agency_selector,
        override_builder=_scenario_recruiting_agency_override,
    ),
    ScenarioSpec(
        id="sparse_vacancy_text",
        description="Override vacancy_text with sparse wording and ensure the model does not hallucinate specifics.",
        selector=_scenario_sparse_vacancy_selector,
        override_builder=_scenario_sparse_vacancy_override,
    ),
    ScenarioSpec(
        id="hybrid_no_remote_focus",
        description="Hybrid vacancy should not turn into a remote-focused first message.",
        selector=_scenario_hybrid_no_remote_selector,
        override_builder=_scenario_hybrid_no_remote_override,
    ),
]


def _resolve_prompt_cfg() -> tuple[str, Optional[str]]:
    prompt_id = os.environ.get("FIRST_TOUCH_HH_PROMPT_ID")
    prompt_version = os.environ.get("FIRST_TOUCH_HH_PROMPT_VERSION")
    if prompt_id:
        return str(prompt_id), (str(prompt_version) if prompt_version else None)

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")

    cfg = load_yaml(CFG_PATH)
    comp = _component_cfg(cfg, "first_touch_hh")
    prompt_id = comp.get("prompt_id") if isinstance(comp, dict) else None
    prompt_version = comp.get("prompt_version") if isinstance(comp, dict) else None
    if not prompt_id:
        raise RuntimeError(
            "Missing FIRST_TOUCH_HH_PROMPT_ID env var and prompt_id in tests/tools/model.yaml (first_touch_hh section)."
        )

    os.environ["FIRST_TOUCH_HH_PROMPT_ID"] = str(prompt_id)
    if prompt_version is not None:
        os.environ["FIRST_TOUCH_HH_PROMPT_VERSION"] = str(prompt_version)
    return str(prompt_id), (str(prompt_version) if prompt_version is not None else None)


def _eval_instruction() -> str:
    return (
        "Ты строгий QA-ревьюер первого HH-сообщения рекрутера.\n"
        "Даны:\n"
        "- input_variables: единственный источник правды о кандидате и вакансии\n"
        "- expected_facts: факты, которые желательно увидеть в сообщении\n"
        "- generated_message: текст сообщения\n\n"
        "Задача:\n"
        "1) facts_present: для каждого ключа из expected_facts определить true/false.\n"
        "2) hallucinated_facts: перечисли фактические утверждения про компанию, вакансию, формат работы, бонусы, зарплату, кандидата,\n"
        "   которые не поддерживаются input_variables.\n"
        "3) question_present: есть ли в сообщении итоговый вопросительный CTA.\n"
        "4) company_rule_ok:\n"
        "   - если hiring_company_name пустой или равен 'рекрутинговое агентство', сообщение не должно представлять это как название работодателя;\n"
        "   - при пустом hiring_company_name или 'рекрутинговое агентство' допустимы нейтральные описания работодателя, если они прямо следуют из vacancy_text;\n"
        "   - если hiring_company_name задан и не равен 'рекрутинговое агентство', сообщение может упоминать это название.\n"
        "Не считай галлюцинацией общие рекрутерские фразы без новых фактов.\n"
        "Верни строго JSON:\n"
        "{"
        '"facts_present": {"vacancy_name": true/false, ...},'
        '"hallucinated_facts": ["..."],'
        '"question_present": true/false,'
        '"company_rule_ok": true/false,'
        '"comment": "кратко на русском"'
        "}"
    )


def _eval_payload(case: Dict[str, Any], message: str) -> str:
    payload = {
        "instruction": _eval_instruction(),
        "input_variables": case.get("input_variables") or {},
        "expected_facts": case.get("expected_facts") or {},
        "generated_message": message,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_base_case(entry: CdmEntry) -> Dict[str, Any]:
    input_variables = _build_input_variables_from_cdm(entry.data)
    return {
        "case_type": "cdm",
        "scenario_id": "base",
        "id": entry.path.stem,
        "description": f"Base HH case from {entry.path.name}",
        "base_cdm_file": entry.path.name,
        "scenario_overrides": {},
        "base_input_variables": dict(input_variables),
        "input_variables": input_variables,
        "expected_facts": _build_expected_facts(input_variables),
        "required_markers": _build_required_markers(input_variables),
        "forbidden_markers": _build_forbidden_markers(input_variables),
    }


def _build_override_case(entry: CdmEntry, spec: ScenarioSpec) -> Dict[str, Any]:
    base_input_variables = _build_input_variables_from_cdm(entry.data)
    overrides = spec.override_builder(entry.data, dict(base_input_variables))
    input_variables = dict(base_input_variables)
    input_variables.update(overrides)

    return {
        "case_type": "override",
        "scenario_id": spec.id,
        "id": f"{entry.path.stem}__{spec.id}",
        "description": spec.description,
        "base_cdm_file": entry.path.name,
        "scenario_overrides": overrides,
        "base_input_variables": base_input_variables,
        "input_variables": input_variables,
        "expected_facts": _build_expected_facts(input_variables),
        "required_markers": _build_required_markers(input_variables),
        "forbidden_markers": _build_forbidden_markers(input_variables),
    }


def _select_override_cases(entries: List[CdmEntry]) -> List[Dict[str, Any]]:
    override_cases: List[Dict[str, Any]] = []
    for spec in SCENARIO_SPECS:
        if spec.id == "base":
            continue
        for entry in entries:
            base_vars = _build_input_variables_from_cdm(entry.data)
            if spec.selector(entry.data, base_vars):
                override_cases.append(_build_override_case(entry, spec))
                break
    return override_cases


def _build_cases(entries: List[CdmEntry], mode: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if mode in ("cdm", "all"):
        cases.extend(_build_base_case(entry) for entry in entries)
    if mode in ("overrides", "all"):
        cases.extend(_select_override_cases(entries))
    return cases


def _compact_result_payload(
    *,
    passed: bool,
    passed_strict: bool,
    question_present: bool,
    company_rule_ok: bool,
    question_count: int,
    paragraph_count: int,
    last_line_has_question: bool,
    greeting_ok: bool,
    recruiter_intro_ok: bool,
    salary_mentioned: bool,
    source_mentioned: bool,
    candidate_facts_mentioned: bool,
    informal_tone_detected: bool,
    emoji_present: bool,
    missing_required_keys: List[str],
    missing_required_markers: List[str],
    forbidden_markers_found: List[str],
    hallucinated_facts: List[str],
    fail_reasons: List[str],
    comment: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "passed": passed,
        "passed_strict": passed_strict,
        "fail_reasons": fail_reasons,
    }
    if missing_required_keys:
        result["missing_required_keys"] = missing_required_keys
    if missing_required_markers:
        result["missing_required_markers"] = missing_required_markers
    if forbidden_markers_found:
        result["forbidden_markers_found"] = forbidden_markers_found
    if hallucinated_facts:
        result["hallucinated_facts"] = hallucinated_facts

    failed_checks: Dict[str, Any] = {}
    if question_count != 1:
        failed_checks["question_count"] = question_count
    if paragraph_count < 3 or paragraph_count > 5:
        failed_checks["paragraph_count"] = paragraph_count
    if not last_line_has_question:
        failed_checks["last_line_has_question"] = last_line_has_question
    if not question_present:
        failed_checks["question_present"] = question_present
    if not greeting_ok:
        failed_checks["greeting_ok"] = greeting_ok
    if not recruiter_intro_ok:
        failed_checks["recruiter_intro_ok"] = recruiter_intro_ok
    if not company_rule_ok:
        failed_checks["company_rule_ok"] = company_rule_ok
    if salary_mentioned:
        failed_checks["salary_mentioned"] = salary_mentioned
    if source_mentioned:
        failed_checks["source_mentioned"] = source_mentioned
    if candidate_facts_mentioned:
        failed_checks["candidate_facts_mentioned"] = candidate_facts_mentioned
    if informal_tone_detected:
        failed_checks["informal_tone_detected"] = informal_tone_detected
    if emoji_present:
        failed_checks["emoji_present"] = emoji_present
    if failed_checks:
        result["failed_checks"] = failed_checks
    if comment and (not passed or hallucinated_facts):
        result["comment"] = comment
    return result


def _counts_by_key(cases: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        value = str(case.get(key) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _compute_summary(cases: List[Dict[str, Any]], requested_limit: int, mode: str) -> Dict[str, Any]:
    result_cases = [c for c in cases if "result" in c]
    error_cases = [c for c in cases if "error" in c]

    total = len(result_cases)
    cdm_cases = sum(1 for c in result_cases if c.get("case_type") == "cdm")
    override_cases = sum(1 for c in result_cases if c.get("case_type") == "override")
    pass_count = sum(1 for c in result_cases if c.get("result", {}).get("passed"))
    failed_count = total - pass_count
    strict_pass_count = sum(1 for c in result_cases if c.get("result", {}).get("passed_strict"))
    question_count = sum(1 for c in result_cases if c.get("result", {}).get("question_present"))
    hallucination_free_count = sum(1 for c in result_cases if not (c.get("result", {}).get("hallucinated_facts") or []))
    company_rule_failures = sum(1 for c in result_cases if not c.get("result", {}).get("company_rule_ok", True))

    fail_reasons_dist: Dict[str, int] = {}
    hallucinated_dist: Dict[str, int] = {}
    missing_required_dist: Dict[str, int] = {}
    failures_by_scenario: Dict[str, int] = {}

    for case in result_cases:
        result = case.get("result") or {}
        scenario_id = str(case.get("scenario_id") or "")
        if not result.get("passed") and scenario_id:
            failures_by_scenario[scenario_id] = failures_by_scenario.get(scenario_id, 0) + 1
        for item in result.get("fail_reasons", []) or []:
            fail_reasons_dist[item] = fail_reasons_dist.get(item, 0) + 1
        for item in result.get("hallucinated_facts", []) or []:
            hallucinated_dist[item] = hallucinated_dist.get(item, 0) + 1
        for item in result.get("missing_required_keys", []) or []:
            missing_required_dist[item] = missing_required_dist.get(item, 0) + 1

    return {
        "mode": mode,
        "requested_limit": requested_limit,
        "total_cases": total,
        "cdm_cases": cdm_cases,
        "override_cases": override_cases,
        "passed_cases": pass_count,
        "failed_cases": failed_count,
        "errors_count": len(error_cases),
        "pass_rate": (pass_count / total) if total else 0.0,
        "strict_pass_rate": (strict_pass_count / total) if total else 0.0,
        "question_rate": (question_count / total) if total else 0.0,
        "hallucination_free_rate": (hallucination_free_count / total) if total else 0.0,
        "company_rule_failures": company_rule_failures,
        "counts_by_case_type": _counts_by_key(result_cases, "case_type"),
        "counts_by_scenario": _counts_by_key(result_cases, "scenario_id"),
        "failures_by_scenario": failures_by_scenario,
        "fail_reasons_distribution": fail_reasons_dist,
        "hallucinated_facts_distribution": hallucinated_dist,
        "missing_required_distribution": missing_required_dist,
    }


def _report_definitions() -> Dict[str, Any]:
    return {
        "pass_criteria": [
            "missing_required_keys is empty",
            "missing_required_markers is empty",
            "forbidden_markers_found is empty",
            "question_count == 1",
            "last_line_has_question = true",
            "paragraph_count between 3 and 5",
            "greeting_ok = true",
            "recruiter_intro_ok = true",
            "company_rule_ok = true",
            "salary_mentioned = false",
            "source_mentioned = false",
            "candidate_facts_mentioned = false",
            "informal_tone_detected = false",
            "emoji_present = false",
        ],
        "strict_criteria": [
            "passed = true",
            "hallucinated_facts is empty",
        ],
        "notes": {
            "case_sources": "HH runner строит base-кейсы из tests/fixtures/cdm/hh и special override-кейсы прямо в коде runner.",
            "overrides": "Override-кейсы мутируют только те input_variables, которые нужны для проверки конкретного правила prompt.",
            "company_rule_ok": "Для пустого company name и 'рекрутинговое агентство' сообщение должно использовать нейтральное company handling.",
            "hallucinated_facts": "LLM-judge проверяет, что сообщение не выдумывает факты вне input_variables.",
            "question_count": "У FIRST_TOUCH_HH должен быть ровно один итоговый CTA-вопрос.",
        },
    }


def _collect_failures(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cases if "result" in c and not c.get("result", {}).get("passed")]


def _collect_errors(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cases if "error" in c]


def _recruiter_intro_ok(recruiter_name: str, message: str) -> bool:
    low = _normalized_low(message)
    if not recruiter_name:
        return "меня зовут" not in low
    if "меня зовут" not in low:
        return True
    return _contains_normalized(f"Меня зовут {recruiter_name}", message)


def run_suite(
    limit: int,
    mode: str,
    eval_model: str,
    cdm_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    if mode not in ("cdm", "overrides", "all"):
        raise ValueError("--mode must be cdm|overrides|all")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    prompt_id, prompt_version = _resolve_prompt_cfg()

    all_cdm_paths = load_cdm_paths(cdm_dir)
    if not all_cdm_paths:
        raise FileNotFoundError(f"No HH CDM fixtures found in {cdm_dir}")
    selected_cdm_paths = all_cdm_paths[:limit] if limit > 0 else all_cdm_paths
    selected_entries = [CdmEntry(path=path, data=load_json(path)) for path in selected_cdm_paths]
    selected_cases = _build_cases(selected_entries, mode=mode)

    _log(
        "[init] "
        f"run_id={run_id} mode={mode} prompt_id={prompt_id} prompt_version={prompt_version} "
        f"eval_model={eval_model} cdm_dir={cdm_dir} requested_limit={limit} "
        f"selected_cdm={len(selected_entries)} total_cases={len(selected_cases)}"
    )

    generator = FirstTouchHHGenerator(prompt_id=prompt_id, prompt_version=prompt_version)
    evaluator = FirstTouchHHEvaluator(model=eval_model)

    token_usage_total = _blank_usage()
    token_usage = {
        "message_generator": _blank_usage(),
        "message_evaluator": _blank_usage(),
    }
    cases: List[Dict[str, Any]] = []

    total_cases = len(selected_cases)
    for idx, case in enumerate(selected_cases, start=1):
        case_id = str(case.get("id") or f"case_{idx}")
        scenario_id = str(case.get("scenario_id") or "")
        base_cdm_file = str(case.get("base_cdm_file") or "")
        _log(f"[run] case {idx}/{total_cases} ({case_id}) scenario={scenario_id} cdm={base_cdm_file}")

        try:
            input_variables = dict(case.get("input_variables") or {})
            message = _normalize_text(generator.generate_message(input_variables))
            _accumulate_usage(token_usage_total, generator.last_usage)
            _accumulate_usage(token_usage["message_generator"], generator.last_usage)

            eval_result = evaluator.evaluate(case=case, message=message)
            _accumulate_usage(token_usage_total, evaluator.last_usage)
            _accumulate_usage(token_usage["message_evaluator"], evaluator.last_usage)

            expected_facts = dict(case.get("expected_facts") or {})
            missing_required_keys = [k for k in expected_facts if not bool(eval_result.facts_present.get(k, False))]
            missing_required_markers = _find_missing_required_markers(message, list(case.get("required_markers") or []))
            forbidden_markers_found = _find_forbidden_markers(message, list(case.get("forbidden_markers") or []))

            question_count = message.count("?")
            paragraph_count = _count_paragraphs(message)
            last_line_has_question = "?" in _last_nonempty_line(message)
            greeting_ok = _starts_with_expected_greeting(str(input_variables.get("candidate_name") or "").strip(), message)
            salary_mentioned = _has_any_pattern(message, SALARY_PATTERNS)
            source_mentioned = _has_any_pattern(message, SOURCE_PATTERNS)
            candidate_facts_mentioned = _has_any_pattern(message, CANDIDATE_FACT_PATTERNS)
            informal_tone_detected = _has_any_pattern(message, INFORMAL_PATTERNS)
            emoji_present = bool(EMOJI_PATTERN.search(message))

            company_mode = _company_mode(input_variables)
            company_rule_ok = bool(eval_result.company_rule_ok)
            if company_mode == "named":
                company_name = str(input_variables.get("hiring_company_name") or "").strip()
                company_rule_ok = company_rule_ok and bool(company_name) and _contains_normalized(company_name, message)
            else:
                company_rule_ok = company_rule_ok and not _contains_invented_company_reference(message, input_variables)

            recruiter_name = str(input_variables.get("recruiter_name") or "").strip()
            recruiter_intro_ok = _recruiter_intro_ok(recruiter_name, message)

            fail_reasons: List[str] = []
            if missing_required_keys:
                fail_reasons.append("missing_required_keys")
            if missing_required_markers:
                fail_reasons.append("missing_required_markers")
            if forbidden_markers_found:
                fail_reasons.append("forbidden_markers")
            if question_count != 1:
                fail_reasons.append("bad_question_count")
            if not last_line_has_question:
                fail_reasons.append("cta_not_last")
            if paragraph_count < 3 or paragraph_count > 5:
                fail_reasons.append("bad_paragraph_count")
            if not greeting_ok:
                fail_reasons.append("bad_greeting")
            if not recruiter_intro_ok:
                fail_reasons.append("bad_recruiter_intro")
            if not company_rule_ok:
                fail_reasons.append("bad_company_handling")
            if salary_mentioned:
                fail_reasons.append("salary_mentioned")
            if source_mentioned:
                fail_reasons.append("source_mentioned")
            if candidate_facts_mentioned:
                fail_reasons.append("candidate_facts_mentioned")
            if informal_tone_detected:
                fail_reasons.append("informal_tone_detected")
            if emoji_present:
                fail_reasons.append("emoji_present")

            passed = not fail_reasons
            passed_strict = bool(passed and not eval_result.hallucinated_facts)

            case_record = {
                "case_type": str(case.get("case_type") or ""),
                "scenario_id": scenario_id,
                "case_id": case_id,
                "description": str(case.get("description") or ""),
                "base_cdm_file": base_cdm_file,
                "input_variables": input_variables,
                "expected_facts": expected_facts,
                "message": message,
                "result": _compact_result_payload(
                    passed=passed,
                    passed_strict=passed_strict,
                    question_present=eval_result.question_present,
                    company_rule_ok=company_rule_ok,
                    question_count=question_count,
                    paragraph_count=paragraph_count,
                    last_line_has_question=last_line_has_question,
                    greeting_ok=greeting_ok,
                    recruiter_intro_ok=recruiter_intro_ok,
                    salary_mentioned=salary_mentioned,
                    source_mentioned=source_mentioned,
                    candidate_facts_mentioned=candidate_facts_mentioned,
                    informal_tone_detected=informal_tone_detected,
                    emoji_present=emoji_present,
                    missing_required_keys=missing_required_keys,
                    missing_required_markers=missing_required_markers,
                    forbidden_markers_found=forbidden_markers_found,
                    hallucinated_facts=eval_result.hallucinated_facts,
                    fail_reasons=fail_reasons,
                    comment=eval_result.comment,
                ),
            }
            scenario_overrides = dict(case.get("scenario_overrides") or {})
            if scenario_overrides:
                case_record["scenario_overrides"] = scenario_overrides
            cases.append(case_record)

            if passed:
                _log(f"  [ok] case={case_id} strict={passed_strict} question_count={question_count}")
            else:
                _log(f"  [fail] case={case_id} reasons={fail_reasons}")

        except Exception as exc:
            error_case = {
                "case_type": str(case.get("case_type") or ""),
                "scenario_id": scenario_id,
                "case_id": case_id,
                "description": str(case.get("description") or ""),
                "base_cdm_file": base_cdm_file,
                "error": f"{type(exc).__name__}: {exc}",
            }
            scenario_overrides = dict(case.get("scenario_overrides") or {})
            if scenario_overrides:
                error_case["scenario_overrides"] = scenario_overrides
            cases.append(error_case)
            _log(f"  [error] case={case_id} {type(exc).__name__}: {exc}")

    finished_at = datetime.datetime.now()
    failures = _collect_failures(cases)
    errors = _collect_errors(cases)
    summary = _compute_summary(cases=cases, requested_limit=limit, mode=mode)

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "mode": mode,
        "cdm_count": len(selected_entries),
        "prompt": {"prompt_id": prompt_id, "prompt_version": prompt_version},
        "eval_model": eval_model,
        "token_usage_total": token_usage_total,
        "token_usage": token_usage,
        "summary": summary,
        "cases": cases,
        "failures": failures,
        "errors": errors,
    }

    ensure_dirs(out_dir)
    out_path = out_dir / f"first_touch_hh_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding=TEXT_FILE_ENCODING)

    _log(
        "[summary] "
        f"total_cases={summary['total_cases']} "
        f"passed={summary['passed_cases']} "
        f"failed={summary['failed_cases']} "
        f"errors={summary['errors_count']} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}"
    )
    _log(f"[done] report saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FIRST_TOUCH_HH prompt test")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="How many HH cdm fixtures to use as a base.")
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["cdm", "overrides", "all"],
        help="Run only base cdm cases, only built-in override cases, or both.",
    )
    parser.add_argument("--cdm-dir", type=pathlib.Path, default=CDM_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPORTS_DIR)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    args = parser.parse_args()

    report_path = run_suite(
        limit=int(args.limit),
        mode=str(args.mode),
        eval_model=str(args.eval_model),
        cdm_dir=args.cdm_dir,
        out_dir=args.out_dir,
    )
    print("Report ->", report_path)


if __name__ == "__main__":
    main()

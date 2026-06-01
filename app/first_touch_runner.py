from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from openai import OpenAI

from adapters.adapters import to_input_form

ROOT = pathlib.Path(__file__).resolve().parents[1]
CDM_DIR = ROOT / "tests" / "fixtures" / "cdm" / "std"
REPORTS_DIR = ROOT / "tests" / "reports" / "first_touch"
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"

DEFAULT_LIMIT = 10
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"

SOURCE_OBJECT_CASES: List[Dict[str, Any]] = [
    {
        "id": "candidate_source_linkedin_profile",
        "candidate_source": "профиль LinkedIn",
        "allowed_forms": ["ваш профиль linkedin", "ваш профиль на linkedin"],
        "bare_forms": ["профиль linkedin", "профиль на linkedin"],
    },
    {
        "id": "candidate_source_hh_resume",
        "candidate_source": "резюме HH",
        "allowed_forms": ["ваше резюме на hh"],
        "bare_forms": ["резюме hh", "резюме на hh"],
    },
    {
        "id": "candidate_source_github",
        "candidate_source": "GitHub",
        "allowed_forms": ["ваш github", "ваш профиль на github"],
        "bare_forms": ["github"],
    },
    {
        "id": "candidate_source_portfolio",
        "candidate_source": "портфолио",
        "allowed_forms": ["ваше портфолио"],
        "bare_forms": ["портфолио"],
    },
    {
        "id": "candidate_source_linkedin_page",
        "candidate_source": "страница на LinkedIn",
        "allowed_forms": ["вашу страницу на linkedin"],
        "bare_forms": ["страница на linkedin", "ваша страница на linkedin", "профиль linkedin", "профиль на linkedin"],
    },
    {
        "id": "candidate_source_empty",
        "candidate_source": "",
        "allowed_forms": [],
        "bare_forms": ["linkedin", "hh", "github", "портфолио"],
    },
    {
        "id": "candidate_source_github_account",
        "candidate_source": "аккаунт GitHub",
        "allowed_forms": ["ваш аккаунт github", "ваш профиль на github"],
        "bare_forms": ["аккаунт github", "ваш github"],
    },
]

PROMPT_RULE_CASES: List[Dict[str, Any]] = [
    {
        "id": "case_empty_recruiter_name",
        "description": "Empty recruiter_name should not produce a self-introduction by name.",
        "candidate_name": "Илья",
        "recruiter_name": "",
        "company_name": "ExampleSoft",
        "company_description": "Продуктовая IT-компания, развивающая B2B SaaS платформу.",
        "vacancy_name": "Backend Engineer",
        "vacancy_responsibilities": "Разработка backend-сервисов и интеграций.",
        "vacancy_stack": "Python, FastAPI",
        "checks": {
            "recruiter_name_absent": True,
            "vacancy_reason_present": True,
            "plain_message_only": True,
            "single_soft_cta": True,
        },
    },
    {
        "id": "case_remote_priority",
        "description": "Remote should be surfaced as an attractive factor when it is explicitly present in context.",
        "candidate_name": "Илья",
        "recruiter_name": "Анна",
        "company_name": "ExampleSoft",
        "company_description": "Аккредитованная IT-компания. Полностью удаленный формат работы по России.",
        "vacancy_name": "Backend Engineer",
        "vacancy_responsibilities": "Разработка backend-сервисов и интеграций.",
        "vacancy_stack": "Python, FastAPI",
        "checks": {
            "recruiter_name_present": True,
            "vacancy_reason_present": True,
            "remote_priority": True,
            "plain_message_only": True,
            "single_soft_cta": True,
        },
    },
    {
        "id": "case_no_work_format_specified",
        "description": "If work format is not present in context, it must not be invented in the message.",
        "candidate_name": "Илья",
        "recruiter_name": "Анна",
        "company_name": "ExampleSoft",
        "company_description": "Продуктовая команда, которая развивает B2B SaaS платформу для автоматизации финансовых процессов.",
        "vacancy_name": "Backend Engineer",
        "vacancy_responsibilities": "Разработка backend-сервисов, интеграций и внутренних API.",
        "vacancy_stack": "Python, FastAPI",
        "checks": {
            "recruiter_name_present": True,
            "vacancy_reason_present": True,
            "no_work_format_specified": True,
            "no_fake_remote": True,
            "plain_message_only": True,
            "single_soft_cta": True,
        },
    },
    {
        "id": "case_hybrid_not_overemphasized",
        "description": "Hybrid may be present in context but should not replace the core value proposition or turn into fake remote.",
        "candidate_name": "Илья",
        "recruiter_name": "Анна",
        "company_name": "ExampleSoft",
        "company_description": "Продуктовая аналитическая команда в e-commerce. Гибридный формат работы, важно быть в Москве.",
        "vacancy_name": "Product Analyst",
        "vacancy_responsibilities": "Анализ продуктовых метрик, построение BI-отчетности и работа с экспериментами.",
        "vacancy_stack": "SQL, Python",
        "checks": {
            "recruiter_name_present": True,
            "vacancy_reason_present": True,
            "hybrid_not_overemphasized": True,
            "no_fake_remote": True,
            "plain_message_only": True,
            "single_soft_cta": True,
        },
    },
    {
        "id": "case_it_accreditation",
        "description": "IT accreditation should be usable as a positive fact when it is explicitly present in context.",
        "candidate_name": "Илья",
        "recruiter_name": "Анна",
        "company_name": "ExampleSoft",
        "company_description": "Аккредитованная IT-компания, которая развивает платформу цифровых сервисов для бизнеса.",
        "vacancy_name": "Backend Engineer",
        "vacancy_responsibilities": "Разработка backend-сервисов и API для внутренних продуктовых команд.",
        "vacancy_stack": "Python, FastAPI",
        "checks": {
            "recruiter_name_present": True,
            "vacancy_reason_present": True,
            "it_accreditation_priority": True,
            "no_fake_remote": True,
            "plain_message_only": True,
            "single_soft_cta": True,
        },
    },
    {
        "id": "case_single_soft_cta",
        "description": "CTA should stay soft, singular, and not branch into alternatives.",
        "candidate_name": "Илья",
        "recruiter_name": "Анна",
        "company_name": "ExampleSoft",
        "company_description": "Продуктовая IT-компания, развивающая внутреннюю платформу цифровых сервисов.",
        "vacancy_name": "Backend Engineer",
        "vacancy_responsibilities": "Разработка backend-сервисов. Интеграции с внутренними системами. Поддержка API для продуктовых команд.",
        "vacancy_stack": "Python, FastAPI",
        "checks": {
            "recruiter_name_present": True,
            "vacancy_reason_present": True,
            "single_soft_cta": True,
            "no_double_question_cta": True,
            "no_alternative_cta": True,
            "plain_message_only": True,
        },
    },
    {
        "id": "case_plain_message_only",
        "description": "The model must return only the ready-to-send message without meta commentary.",
        "candidate_name": "Илья",
        "recruiter_name": "Анна",
        "company_name": "ExampleSoft",
        "company_description": "Продуктовая IT-компания для B2B клиентов.",
        "vacancy_name": "Backend Engineer",
        "vacancy_responsibilities": (
            "Разработка backend-сервисов и интеграций.\n"
            "Поддержка API для продуктовых команд.\n"
            "Участие в развитии внутренних платформ."
        ),
        "vacancy_stack": "Python, FastAPI, PostgreSQL",
        "checks": {
            "recruiter_name_present": True,
            "vacancy_reason_present": True,
            "plain_message_only": True,
            "single_soft_cta": True,
        },
    },
]

class AssistantError(Exception):
    """Ошибка генерации первого сообщения."""


class InputForm:
    """Структура входных данных для генерации первого касания.

    Перенесена из бывшего внешнего пакета telegramMessageGenerator-main, чтобы раннер
    не зависел от стороннего кода. Поля и логика вычислений сохранены без изменений.
    """

    def __init__(
        self,
        recruiter_name: str,
        company_name: str,
        vacancy_name: str,
        company_industry: Optional[str] = None,
        vacancy_skills: Optional[List[str]] = None,
        vacancy_responsibilities: Optional[str] = None,
        candidate_name: Optional[str] = None,
        candidate_contacts=None,
        candidate_job_list=None,
        candidate_skills=None,
        salary_range_from: Optional[int] = None,
        salary_range_to: Optional[int] = None,
        use_salary: bool = False,
        formality: Optional[bool] = False,
        company_description: Optional[str] = None,
        vacancy_stack: Optional[str] = None,
    ):
        if candidate_skills is None:
            candidate_skills = []
        if candidate_contacts is None:
            candidate_contacts = []
        if candidate_job_list is None:
            candidate_job_list = []
        if vacancy_skills is None:
            vacancy_skills = []
        self.recruiter_name = recruiter_name

        self.formality = "formal" if formality else "informal"

        self.company_name = company_name
        self.vacancy_name = vacancy_name
        self.vacancy_responsibilities = vacancy_responsibilities

        self.company_industry = company_industry
        self.vacancy_skills = ", ".join(vacancy_skills)

        self.salary_range_from = salary_range_from
        self.salary_range_to = salary_range_to
        self.salary = self._get_salary(use_salary)

        self.company_description = company_description

        self.candidate_name = candidate_name

        self.candidate_contacts = self._social_accounts_found(candidate_contacts)
        self.candidate_skills = ", ".join(self._skills_search(candidate_skills, vacancy_skills))

        self.candidate_job_list = candidate_job_list
        self.candidate_LinkedIn = self._company_industry_search(candidate_job_list, self.company_industry)

        self.reasons = self._get_reasons()

        self.vacancy_stack = vacancy_stack

    def _get_reasons(self):
        reasons = ""
        if self.candidate_skills:
            reasons += f"Кандидат имеет нужные скиллы: {self.candidate_skills}. "
        if self.candidate_LinkedIn:
            reasons += f"Кандидат ранее работал в {self.company_industry} сфере. "
        if not self.candidate_skills and not self.candidate_LinkedIn:
            reasons += "Кандидат имеет релевантный опыт."

        return reasons

    def _get_salary(self, use_salary: bool):
        if not use_salary:
            return ""

        if self.salary_range_from and self.salary_range_to:
            return f"от {self.salary_range_from} до {self.salary_range_to} рублей"
        elif self.salary_range_from and not self.salary_range_to:
            return f"от {self.salary_range_from} рублей"
        elif not self.salary_range_from and self.salary_range_to:
            return f"до {self.salary_range_to} рублей"
        else:
            return ""

    def _social_accounts_found(self, accounts: List) -> List:
        result = []

        if self._social_search(accounts, "ln"):
            result.append("профиль LinkedIn")

        if self._social_search(accounts, "hh"):
            result.append("резюме HH")

        if self._social_search(accounts, "github"):
            result.append("профиль GitHub")

        if self._social_search(accounts, "mk"):
            result.append("резюме Хабр Карьеры")

        return result

    @staticmethod
    def _company_industry_search(jobs_list: List, search_term: str) -> bool:
        return any(
            category.get("title", "").lower() == search_term.lower()
            for job in jobs_list
            for category in job.get("company_norm", {}).get("categories", [])
        )

    @staticmethod
    def _skills_search(skills_list: List, vacancy_skills: List, quantiles=(4.1, 3.1)) -> List:
        skill_set = set(vacancy_skills)
        quantile_set = set(quantiles)

        matching_skills = list(
            filter(lambda d: d.get("skill") in skill_set and d.get("quantile") in quantile_set, skills_list)
        )

        return [d.get("skill") for d in matching_skills]

    @staticmethod
    def _social_search(accounts: List, social: str) -> bool:
        return any(account["type"] == social for account in accounts)


class FirstTouchGenerator:
    """Генерация первого касания через stored-prompt `first_touch` (tests/tools/model.yaml).

    Работает так же, как FirstTouchHHGenerator в first_touch_hh_runner: передаёт набор
    input-переменных в сохранённый промпт через Responses API. Заменяет внешний
    TelegramMessageGenerator (контракт payload и постобработка подписи сохранены).
    """

    def __init__(self, prompt_id: str, prompt_version: str | None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def generate_message(self, input_form: "InputForm") -> str:
        candidate_source = ""
        if input_form.candidate_contacts:
            candidate_source = input_form.candidate_contacts[0]

        payload = {
            "candidate_name": input_form.candidate_name,
            "recruiter_name": input_form.recruiter_name,
            "candidate_source": candidate_source,
            "reason_of_communication": input_form.reasons,
            "hiring_company_name": input_form.company_name,
            "vacancy_name": input_form.vacancy_name,
            "vacancy_responsibilities": input_form.vacancy_responsibilities,
            "message_formality": input_form.formality,
            "company_description": input_form.company_description,
            "vacancy_stack": input_form.vacancy_stack,
            "salary_range": input_form.salary,
        }

        response = self.client.responses.create(
            prompt=self.prompt,
            input=json.dumps(payload, ensure_ascii=False),
            text={"format": {"type": "text"}},
        )
        self.last_usage = getattr(response, "usage", None)
        output = getattr(response, "output_text", "") or ""
        if not output:
            raise AssistantError("Ответ от ассистента пустой")
        return self._ending_check(output, "С уважением")

    @staticmethod
    def _ending_check(text: str, phrase: str) -> str:
        lines = text.split("\n")
        if len(lines) >= 2 and any(phrase.lower() in line.lower() for line in lines[-2:]):
            del lines[-2:]
        return "\n".join(lines)


@dataclass
class EvalResult:
    facts_present: Dict[str, bool]
    hallucinated_facts: List[str]
    question_present: bool
    comment: str


def ensure_dirs(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _normalize_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_source_text(s: str) -> str:
    text = _normalize_text(s).lower().replace("ё", "е")
    text = re.sub(r"[^\w#+.]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


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


def _extract_numbers(text: str) -> List[int]:
    nums: List[int] = []
    for raw in re.findall(r"\d+(?:[\s\u00A0]\d+)*", text or ""):
        cleaned = re.sub(r"[\s\u00A0]+", "", raw)
        try:
            nums.append(int(cleaned))
        except ValueError:
            continue
    return nums


def _extra_numbers(facts: Dict[str, str], message: str) -> List[str]:
    allowed: set[int] = set()
    for value in facts.values():
        allowed.update(_extract_numbers(str(value)))
    msg_nums = set(_extract_numbers(message))
    extra = sorted(n for n in msg_nums if n not in allowed)
    return [f"extra_numbers:{extra}"] if extra else []


def _normalize_for_match(text: str) -> str:
    t = _normalize_text(text).lower()
    t = re.sub(r"[\W_]+", "", t, flags=re.UNICODE)
    return t


def _contains_normalized(needle: str, haystack: str) -> bool:
    n = _normalize_for_match(needle)
    h = _normalize_for_match(haystack)
    return bool(n) and n in h


def _has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text or "", flags=re.IGNORECASE) for pattern in patterns)


def _last_nonempty_line(message: str) -> str:
    lines = [line.strip() for line in _normalize_text(message).split("\n") if line.strip()]
    return lines[-1] if lines else ""


def _find_unpossessive_mentions(message_norm: str, bare_phrase: str) -> List[str]:
    violations: List[str] = []
    search_from = 0
    while True:
        idx = message_norm.find(bare_phrase, search_from)
        if idx < 0:
            break
        prefix = message_norm[:idx].rstrip()
        prev_word = prefix.split(" ")[-1] if prefix else ""
        if prev_word not in {"ваш", "ваше", "вашу"}:
            violations.append(bare_phrase)
        search_from = idx + len(bare_phrase)
    return violations


def _find_phrase_spans(text: str, phrase: str) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    search_from = 0
    while True:
        idx = text.find(phrase, search_from)
        if idx < 0:
            break
        spans.append((idx, idx + len(phrase)))
        search_from = idx + len(phrase)
    return spans


def _check_candidate_source_possessive(
    candidate_source: str,
    message: str,
    source_spec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = {
        "source_possessive_ok": True,
        "missing_possessive_source_forms": [],
        "forbidden_source_forms_found": [],
    }

    if source_spec is None:
        return result

    message_norm = _normalize_source_text(message)
    allowed_forms = [_normalize_source_text(x) for x in source_spec.get("allowed_forms", []) if str(x).strip()]
    bare_forms = [_normalize_source_text(x) for x in source_spec.get("bare_forms", []) if str(x).strip()]
    allowed_spans: List[tuple[int, int]] = []
    for allowed in allowed_forms:
        allowed_spans.extend(_find_phrase_spans(message_norm, allowed))

    if candidate_source:
        has_allowed = any(allowed and allowed in message_norm for allowed in allowed_forms)
        if not has_allowed:
            result["missing_possessive_source_forms"] = allowed_forms

        forbidden_found: List[str] = []
        for bare in bare_forms:
            if not bare:
                continue
            search_from = 0
            while True:
                idx = message_norm.find(bare, search_from)
                if idx < 0:
                    break
                end = idx + len(bare)
                if any(span_start <= idx and end <= span_end for span_start, span_end in allowed_spans):
                    search_from = end
                    continue
                prefix = message_norm[:idx].rstrip()
                prev_word = prefix.split(" ")[-1] if prefix else ""
                if prev_word not in {"ваш", "ваше", "вашу"}:
                    forbidden_found.append(bare)
                search_from = end

        result["forbidden_source_forms_found"] = sorted(set(forbidden_found))
        result["source_possessive_ok"] = has_allowed and not result["forbidden_source_forms_found"]
        return result

    forbidden_found = [bare for bare in bare_forms if bare and bare in message_norm]
    result["forbidden_source_forms_found"] = sorted(set(forbidden_found))
    result["source_possessive_ok"] = not result["forbidden_source_forms_found"]
    return result


def _expected_work_mode_from_cdm(cdm: Dict[str, Any]) -> str | None:
    vacancy = (cdm or {}).get("vacancy") or {}
    for key in ("work_format", "workFormat", "work_mode", "workMode"):
        val = vacancy.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def _company_name_present_when_hidden(company_hidden: bool, original_company_name: str | None, message: str) -> bool:
    if not company_hidden:
        return False
    if not original_company_name or not original_company_name.strip():
        return False
    name = original_company_name.strip()
    if name.lower() == "скрыто":
        return False
    return _contains_normalized(name, message)


def _is_placeholder_company_name(company_name: str) -> bool:
    normalized = _normalize_source_text(company_name)
    return normalized in {"", "скрыто", "не указано", "неуказано", "n a", "na", "unknown"}


def _text_has_remote_fact(text: str) -> bool:
    low = _normalize_text(text).lower().replace("ё", "е")
    return any(token in low for token in ("удален", "удаленн", "remote"))


def _text_has_accreditation_fact(text: str) -> bool:
    low = _normalize_text(text).lower().replace("ё", "е")
    return "аккредит" in low


def _text_has_hybrid_fact(text: str) -> bool:
    low = _normalize_text(text).lower().replace("ё", "е")
    return "гибрид" in low


def _text_has_office_fact(text: str) -> bool:
    low = _normalize_text(text).lower().replace("ё", "е")
    return "офис" in low


def _tech_tokens(text: str) -> set[str]:
    t = _normalize_text(text).lower()
    tokens = re.findall(r"[a-z0-9\+\#\.]{2,}", t, flags=re.IGNORECASE)
    return {tok for tok in tokens if tok}


def _stack_partially_mentioned(expected_stack: str, message: str) -> bool:
    exp = _tech_tokens(expected_stack)
    if not exp:
        return True
    msg = _tech_tokens(message)
    return bool(exp & msg)


def _build_expected_facts_report(input_form: InputForm, include_salary: bool) -> Dict[str, str]:
    facts: Dict[str, str] = {
        "company_name": str(getattr(input_form, "company_name", "") or "").strip(),
        "vacancy_name": str(getattr(input_form, "vacancy_name", "") or "").strip(),
        "company_description": str(getattr(input_form, "company_description", "") or "").strip(),
        "vacancy_responsibilities": str(getattr(input_form, "vacancy_responsibilities", "") or "").strip(),
        "vacancy_stack": str(getattr(input_form, "vacancy_stack", "") or "").strip(),
        "vacancy_skills": str(getattr(input_form, "vacancy_skills", "") or "").strip(),
    }
    if include_salary:
        facts["salary_range"] = str(getattr(input_form, "salary", "") or "").strip()
    return {k: v for k, v in facts.items() if v}


def _build_expected_facts_for_eval(expected_facts_report: Dict[str, str], company_hidden: bool) -> Dict[str, str]:
    facts = dict(expected_facts_report)
    # IMPORTANT:
    # - vacancy_name is always allowed and should always be evaluated
    # - company_name is not evaluated when hidden (we enforce hiding via separate forbidden-check)
    if company_hidden or _is_placeholder_company_name(str(facts.get("company_name") or "")):
        facts.pop("company_name", None)
    return facts


def _build_allowed_context_facts(input_form: InputForm) -> Dict[str, str]:
    candidate_source = str(getattr(input_form, "candidate_source", "") or "").strip()
    if not candidate_source:
        contacts = getattr(input_form, "candidate_contacts", None) or []
        if contacts:
            candidate_source = str(contacts[0] or "").strip()

    reason = str(getattr(input_form, "reason_of_communication", "") or "").strip()
    if not reason:
        reason = str(getattr(input_form, "reasons", "") or "").strip()

    allowed: Dict[str, str] = {}
    if candidate_source:
        allowed["candidate_source"] = candidate_source
    if reason:
        allowed["reason_of_communication"] = reason
    return allowed


def _required_keys(expected_facts_report: Dict[str, str], include_salary: bool) -> List[str]:
    # vacancy_name is ALWAYS required
    keys: List[str] = ["vacancy_name", "vacancy_responsibilities"]

    company_name = expected_facts_report.get("company_name", "")
    if company_name and not _is_placeholder_company_name(company_name):
        keys.insert(0, "company_name")

    if include_salary and "salary_range" in expected_facts_report:
        keys.append("salary_range")

    return keys


def _load_cdm_paths(cdm_dir: pathlib.Path) -> List[pathlib.Path]:
    all_files = sorted(cdm_dir.glob("*.json"))
    if not all_files and (cdm_dir / "std").is_dir():
        all_files = sorted((cdm_dir / "std").glob("*.json"))
    return all_files


def _build_source_possessive_input_form(candidate_source: str) -> InputForm:
    input_form = InputForm(
        recruiter_name="Анна",
        company_name="ExampleSoft",
        company_description="Продуктовая IT-компания",
        vacancy_name="Backend Engineer",
        vacancy_stack="Python, FastAPI",
        company_industry="IT",
        vacancy_responsibilities="Разработка backend-сервисов и интеграций.",
        vacancy_skills=["Python", "FastAPI"],
        salary_range_from=None,
        salary_range_to=None,
        use_salary=False,
        formality=True,
        candidate_name="Илья",
        candidate_contacts=[],
        candidate_job_list=[],
        candidate_skills=[],
    )
    input_form.candidate_contacts = [candidate_source] if candidate_source else []
    input_form.candidate_source = candidate_source
    input_form.reasons = "Кандидат имеет релевантный опыт."
    input_form.reason_of_communication = input_form.reasons
    return input_form


META_PREFIX_PATTERNS: tuple[str, ...] = (
    r"^\s*вот\s+(?:вариант|текст|сообщение)\b",
    r"^\s*готов(?:ый|ое)\s+(?:текст|сообщение)\b",
    r"^\s*конечно[,!\s]+вот\b",
    r"^\s*ниже\b",
    r"^\s*сообщение:\s*$",
)
META_SUFFIX_PATTERNS: tuple[str, ...] = (
    r"\n\s*(?:если нужно|при необходимости|могу также|если хотите)\b",
    r"\n\s*(?:примечание|комментарий):",
)
FORMAT_MENTION_PATTERNS: tuple[str, ...] = (
    r"\bудален\w*\b",
    r"\bremote\b",
    r"\bгибрид\w*\b",
    r"\bофис\w*\b",
)
ALTERNATIVE_CTA_PATTERNS: tuple[str, ...] = (
    r"\sили\s",
    r"\sлибо\s",
)


def _build_prompt_rule_input_form(case: Dict[str, Any]) -> InputForm:
    vacancy_skills = [skill.strip() for skill in str(case.get("vacancy_stack") or "").split(",") if skill.strip()]
    input_form = InputForm(
        recruiter_name=str(case.get("recruiter_name") or "").strip(),
        company_name=str(case.get("company_name") or "").strip(),
        company_description=str(case.get("company_description") or "").strip(),
        vacancy_name=str(case.get("vacancy_name") or "").strip(),
        vacancy_stack=str(case.get("vacancy_stack") or "").strip(),
        company_industry="IT",
        vacancy_responsibilities=str(case.get("vacancy_responsibilities") or "").strip(),
        vacancy_skills=vacancy_skills,
        salary_range_from=None,
        salary_range_to=None,
        use_salary=False,
        formality=True,
        candidate_name=str(case.get("candidate_name") or "").strip(),
        candidate_contacts=[],
        candidate_job_list=[],
        candidate_skills=[],
    )
    input_form.reasons = str(case.get("reason_of_communication") or "Кандидат имеет релевантный опыт.").strip()
    input_form.reason_of_communication = input_form.reasons
    candidate_source = str(case.get("candidate_source") or "").strip()
    input_form.candidate_source = candidate_source
    input_form.candidate_contacts = [candidate_source] if candidate_source else []
    return input_form


def _prompt_rule_case_context(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_name": str(case.get("candidate_name") or "").strip(),
        "recruiter_name": str(case.get("recruiter_name") or "").strip(),
        "company_name": str(case.get("company_name") or "").strip(),
        "company_description": str(case.get("company_description") or "").strip(),
        "vacancy_name": str(case.get("vacancy_name") or "").strip(),
        "vacancy_responsibilities": str(case.get("vacancy_responsibilities") or "").strip(),
        "vacancy_stack": str(case.get("vacancy_stack") or "").strip(),
    }


def _context_text_for_format_checks(case: Dict[str, Any]) -> str:
    return "\n".join(
        [
            str(case.get("company_description") or "").strip(),
            str(case.get("vacancy_responsibilities") or "").strip(),
        ]
    ).strip()


def _plain_message_only_ok(message: str) -> bool:
    text = _normalize_text(message)
    if not text:
        return False
    if "```" in text:
        return False
    for pattern in META_PREFIX_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False
    for pattern in META_SUFFIX_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False
    return True


def _cta_question_count(message: str) -> int:
    return (message or "").count("?")


def _last_line_has_question(message: str) -> bool:
    return "?" in _last_nonempty_line(message)


def _no_alternative_cta_ok(message: str) -> bool:
    last_line = _last_nonempty_line(message).lower()
    return not any(re.search(pattern, last_line, flags=re.IGNORECASE) for pattern in ALTERNATIVE_CTA_PATTERNS)


def _recruiter_name_absent_ok(recruiter_name: str, message: str) -> bool:
    if recruiter_name:
        return True
    low = _normalize_text(message).lower()
    return "меня зовут" not in low and "мое имя" not in low and "моё имя" not in low


def _recruiter_name_present_ok(recruiter_name: str, message: str) -> bool:
    if not recruiter_name:
        return True
    return _contains_normalized(recruiter_name, message)


def _vacancy_reason_present_ok(vacancy_name: str, message: str) -> bool:
    if not vacancy_name:
        return True
    low = _normalize_text(message).lower()
    return _contains_normalized(vacancy_name, message) and any(token in low for token in ("ваканси", "позици", "роль"))


def _message_mentions_any_work_format(message: str) -> bool:
    return _has_any_pattern(message, FORMAT_MENTION_PATTERNS)


def _remote_priority_ok(case: Dict[str, Any], message: str) -> bool:
    context = _context_text_for_format_checks(case)
    if not _text_has_remote_fact(context):
        return True
    return _text_has_remote_fact(message)


def _no_fake_remote_ok(case: Dict[str, Any], message: str) -> bool:
    context = _context_text_for_format_checks(case)
    if _text_has_remote_fact(context):
        return True
    return not _text_has_remote_fact(message)


def _no_work_format_specified_ok(case: Dict[str, Any], message: str) -> bool:
    context = _context_text_for_format_checks(case)
    if _text_has_remote_fact(context) or _text_has_hybrid_fact(context) or _text_has_office_fact(context):
        return True
    return not _message_mentions_any_work_format(message)


def _hybrid_not_overemphasized_ok(case: Dict[str, Any], message: str, eval_result: EvalResult) -> bool:
    context = _context_text_for_format_checks(case)
    if not _text_has_hybrid_fact(context) or _text_has_remote_fact(context):
        return True
    if _text_has_remote_fact(message):
        return False
    if not _text_has_hybrid_fact(message) and not _text_has_office_fact(message):
        return True
    return bool(
        eval_result.facts_present.get("company_description")
        or eval_result.facts_present.get("vacancy_responsibilities")
    )


def _it_accreditation_priority_ok(case: Dict[str, Any], message: str) -> bool:
    context = _context_text_for_format_checks(case)
    if not _text_has_accreditation_fact(context):
        return True
    return _text_has_accreditation_fact(message)


def _evaluate_prompt_rule_checks(
    case: Dict[str, Any],
    message: str,
    eval_result: EvalResult,
) -> Dict[str, Any]:
    checks = dict(case.get("checks") or {})
    recruiter_name = str(case.get("recruiter_name") or "").strip()
    vacancy_name = str(case.get("vacancy_name") or "").strip()
    question_count = _cta_question_count(message)
    results: Dict[str, Any] = {
        "recruiter_name_absent_ok": _recruiter_name_absent_ok(recruiter_name, message),
        "recruiter_name_present_ok": _recruiter_name_present_ok(recruiter_name, message),
        "vacancy_reason_present_ok": _vacancy_reason_present_ok(vacancy_name, message),
        "remote_priority_ok": _remote_priority_ok(case, message),
        "no_fake_remote_ok": _no_fake_remote_ok(case, message),
        "no_work_format_specified_ok": _no_work_format_specified_ok(case, message),
        "hybrid_not_overemphasized_ok": _hybrid_not_overemphasized_ok(case, message, eval_result),
        "it_accreditation_priority_ok": _it_accreditation_priority_ok(case, message),
        "single_cta_ok": question_count == 1 and _last_line_has_question(message),
        "no_double_question_cta": question_count == 1,
        "no_alternative_cta": _no_alternative_cta_ok(message),
        "plain_message_only": _plain_message_only_ok(message),
        "question_count": question_count,
        "last_line_has_question": _last_line_has_question(message),
    }

    fail_reasons: List[str] = []
    if checks.get("recruiter_name_absent") and not results["recruiter_name_absent_ok"]:
        fail_reasons.append("bad_recruiter_name_fallback")
    if checks.get("recruiter_name_present") and not results["recruiter_name_present_ok"]:
        fail_reasons.append("missing_recruiter_name")
    if checks.get("vacancy_reason_present") and not results["vacancy_reason_present_ok"]:
        fail_reasons.append("missing_vacancy_reason")
    if checks.get("remote_priority") and not results["remote_priority_ok"]:
        fail_reasons.append("missing_remote_priority")
    if checks.get("no_fake_remote") and not results["no_fake_remote_ok"]:
        fail_reasons.append("fake_remote_mentioned")
    if checks.get("no_work_format_specified") and not results["no_work_format_specified_ok"]:
        fail_reasons.append("invented_work_format")
    if checks.get("hybrid_not_overemphasized") and not results["hybrid_not_overemphasized_ok"]:
        fail_reasons.append("hybrid_overemphasized")
    if checks.get("it_accreditation_priority") and not results["it_accreditation_priority_ok"]:
        fail_reasons.append("missing_it_accreditation_priority")
    if checks.get("single_soft_cta") and not results["single_cta_ok"]:
        fail_reasons.append("bad_single_cta")
    if checks.get("no_double_question_cta") and not results["no_double_question_cta"]:
        fail_reasons.append("double_question_cta")
    if checks.get("no_alternative_cta") and not results["no_alternative_cta"]:
        fail_reasons.append("alternative_cta")
    if checks.get("plain_message_only") and not results["plain_message_only"]:
        fail_reasons.append("non_plain_message_output")

    results["fail_reasons"] = fail_reasons
    return results


def _log(message: str) -> None:
    print(message)


def _is_transient_error(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(
        token in low
        for token in (
            "connection error",
            "temporary failure",
            "timed out",
            "timeout",
            "server disconnected",
            "internal server error",
            "rate limit",
            "429",
            "503",
            "apiconnectionerror",
        )
    )


def _call_with_retries(fn, attempts: int = 3, base_delay_s: float = 1.0):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not _is_transient_error(exc):
                raise
            time.sleep(base_delay_s * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry helper reached unreachable state")


def _resolve_prompt_cfg() -> tuple[str, Optional[str]]:
    prompt_id = os.environ.get("FIRST_TOUCH_PROMPT_ID")
    prompt_version = os.environ.get("FIRST_TOUCH_PROMPT_VERSION")
    if prompt_id:
        return str(prompt_id), (str(prompt_version) if prompt_version else None)

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")

    cfg = load_yaml(CFG_PATH)
    comp = _component_cfg(cfg, "first_touch")
    prompt_id = comp.get("prompt_id") if isinstance(comp, dict) else None
    prompt_version = comp.get("prompt_version") if isinstance(comp, dict) else None
    if not prompt_id:
        raise RuntimeError(
            "Missing FIRST_TOUCH_PROMPT_ID env var and prompt_id in tests/tools/model.yaml (first_touch section)."
        )

    os.environ["FIRST_TOUCH_PROMPT_ID"] = str(prompt_id)
    if prompt_version is not None:
        os.environ["FIRST_TOUCH_PROMPT_VERSION"] = str(prompt_version)
    return str(prompt_id), (str(prompt_version) if prompt_version is not None else None)


def _eval_instruction() -> str:
    return (
        "Ты строгий QA-ревьюер первого сообщения рекрутера.\n"
        "Даны:\n"
        "- expected_facts: факты вакансии (то, с чем сравниваем наличие фактов в тексте)\n"
        "- allowed_context_facts: допустимые контекстные факты (источник кандидата, причина контакта), их НЕ считать галлюцинациями\n"
        "- generated_message: текст сообщения\n\n"
        "Задача:\n"
        "1) facts_present: для каждого ключа из expected_facts определить true/false.\n"
        "   true только если факт явно упомянут или ясно перефразирован.\n"
        "2) hallucinated_facts: список фактических утверждений про вакансию/условия,\n"
        "   которых нет в expected_facts и allowed_context_facts.\n"
        "   Не добавляй общие рекрутерские фразы типа 'заинтересовал ваш опыт', 'обратили внимание на профиль',\n"
        "   приветствия и вежливые вводные.\n"
        "   Если сомневаешься - лучше НЕ добавляй пункт.\n"
        "3) question_present: есть ли в сообщении вопросительный CTA.\n\n"
        "Верни строго JSON:\n"
        "{"
        '"facts_present": {"company_name": true/false, ...},'
        '"hallucinated_facts": ["..."],'
        '"question_present": true/false,'
        '"comment": "кратко на русском"'
        "}"
    )


def _eval_payload(expected_facts: Dict[str, str], allowed_context_facts: Dict[str, str], message: str) -> str:
    payload = {
        "instruction": _eval_instruction(),
        "expected_facts": expected_facts,
        "allowed_context_facts": allowed_context_facts,
        "generated_message": message,
    }
    return json.dumps(payload, ensure_ascii=False)


def evaluate_message(
    client: OpenAI,
    eval_model: str,
    expected_facts: Dict[str, str],
    allowed_context_facts: Dict[str, str],
    message: str,
) -> EvalResult:
    response = _call_with_retries(
        lambda: client.responses.create(
            model=eval_model,
            input=_eval_payload(expected_facts, allowed_context_facts, message),
        )
    )
    raw = _normalize_text(getattr(response, "output_text", "") or "")
    data = _safe_json_loads(raw)

    facts_present: Dict[str, bool] = {}
    hallucinated_facts: List[str] = []
    question_present = "?" in message
    comment = ""

    facts_present_raw = data.get("facts_present") if isinstance(data, dict) else {}
    for key in expected_facts:
        if isinstance(facts_present_raw, dict):
            facts_present[key] = bool(facts_present_raw.get(key))
        else:
            facts_present[key] = False

    if isinstance(data, dict):
        h = data.get("hallucinated_facts") or []
        if not isinstance(h, list):
            h = [str(h)]
        hallucinated_facts = [str(x).strip() for x in h if str(x).strip()]

        if "question_present" in data:
            question_present = bool(data.get("question_present"))
        comment = str(data.get("comment") or "").strip()

    if "vacancy_stack" in expected_facts and not facts_present.get("vacancy_stack", False):
        if _stack_partially_mentioned(expected_facts.get("vacancy_stack", ""), message):
            facts_present["vacancy_stack"] = True

    return EvalResult(
        facts_present=facts_present,
        hallucinated_facts=hallucinated_facts,
        question_present=question_present,
        comment=comment,
    )


def _compute_summary(
    cases: List[Dict[str, Any]],
    requested_limit: int,
    fixtures_found: int,
    company_leaks_count: int,
) -> Dict[str, Any]:
    result_cases = [c for c in cases if "result" in c]
    error_cases = [c for c in cases if "error" in c]

    total = len(result_cases)
    cdm_cases = sum(1 for c in result_cases if c.get("case_type") == "cdm")
    source_possessive_cases = sum(1 for c in result_cases if c.get("case_type") == "candidate_source_possessive")
    prompt_rule_cases = sum(1 for c in result_cases if c.get("case_type") == "prompt_rule")
    hidden_cases = sum(1 for c in result_cases if c.get("company_hidden"))
    visible_cases = total - hidden_cases

    pass_count = sum(1 for c in result_cases if c.get("result", {}).get("passed"))
    failed_count = total - pass_count
    strict_pass_count = sum(1 for c in result_cases if c.get("result", {}).get("passed_strict"))
    question_count = sum(1 for c in result_cases if c.get("result", {}).get("question_present"))
    halluc_free_count = sum(1 for c in result_cases if not (c.get("result", {}).get("hallucinated_facts") or []))
    source_possessive_failures = sum(
        1
        for c in result_cases
        if c.get("case_type") == "candidate_source_possessive"
        and not c.get("result", {}).get("source_possessive_ok", True)
    )

    missing_required_dist: Dict[str, int] = {}
    missing_optional_dist: Dict[str, int] = {}
    hallucinated_dist: Dict[str, int] = {}
    fail_reasons_dist: Dict[str, int] = {}

    for c in result_cases:
        r = c.get("result") or {}
        for k in r.get("missing_required_keys", []) or []:
            missing_required_dist[k] = missing_required_dist.get(k, 0) + 1
        for k in r.get("missing_optional_keys", []) or []:
            missing_optional_dist[k] = missing_optional_dist.get(k, 0) + 1
        for h in r.get("hallucinated_facts", []) or []:
            hallucinated_dist[h] = hallucinated_dist.get(h, 0) + 1
        for fr in r.get("fail_reasons", []) or []:
            fail_reasons_dist[fr] = fail_reasons_dist.get(fr, 0) + 1

    rate = (pass_count / total) if total else 0.0
    strict_rate = (strict_pass_count / total) if total else 0.0
    q_rate = (question_count / total) if total else 0.0
    h_rate = (halluc_free_count / total) if total else 0.0

    return {
        "requested_limit": requested_limit,
        "fixtures_found": fixtures_found,
        "total_cases": total,
        "cdm_cases": cdm_cases,
        "candidate_source_possessive_cases": source_possessive_cases,
        "prompt_rule_cases": prompt_rule_cases,
        "hidden_cases": hidden_cases,
        "visible_cases": visible_cases,
        "passed_cases": pass_count,
        "failed_cases": failed_count,
        "errors_count": len(error_cases),
        "pass_rate": rate,
        "strict_pass_rate": strict_rate,
        "question_rate": q_rate,
        "hallucination_free_rate": h_rate,
        "company_leaks_count": company_leaks_count,
        "candidate_source_possessive_failures": source_possessive_failures,
        "missing_required_distribution": missing_required_dist,
        "missing_optional_distribution": missing_optional_dist,
        "hallucinated_facts_distribution": hallucinated_dist,
        "fail_reasons_distribution": fail_reasons_dist,
    }


def _report_definitions(require_question: bool) -> Dict[str, Any]:
    pass_criteria = [
        "missing_required_keys is empty",
        "extra_numbers is empty",
    ]
    if require_question:
        pass_criteria.append("question_present=true")
    pass_criteria.append("if company_hidden=true then original_company_name must NOT be mentioned")
    pass_criteria.append("for candidate_source possessive cases: source_possessive_ok=true")
    pass_criteria.append("for prompt_rule cases: case-specific structural checks must pass")

    return {
        "pass_criteria": pass_criteria,
        "strict_criteria": [
            "passed=true",
            "hallucinated_facts is empty",
        ],
        "notes": {
            "missing_optional_keys": "Диагностика. Не влияет на passed.",
            "hallucinated_facts": "Диагностика (passed_strict=false, но passed может быть true).",
            "allowed_context_facts": "Контекст (candidate_source, reason_of_communication) не считается галлюцинацией.",
            "source_possessive_ok": "Проверка корректного употребления форм 'Ваш / Ваше / Вашу' для candidate_source. Пустой источник не должен порождать LinkedIn/HH/GitHub/портфолио, а для GitHub допустимы близкие притяжательные эквиваленты вроде 'Ваш GitHub' и 'Ваш профиль на GitHub'.",
            "prompt_rule_cases": "Дополнительные встроенные кейсы для новых правил prompt first_touch: пустой recruiter_name, remote priority, отсутствие формата, hybrid without overemphasis, IT accreditation, single soft CTA, plain message only.",
            "single_cta_ok": "Для prompt_rule кейсов ожидается один мягкий CTA-вопрос в конце без альтернатив.",
            "plain_message_only": "Для prompt_rule кейсов ответ должен быть только готовым текстом сообщения без мета-комментариев.",
        },
    }


def _collect_failures(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cases if "result" in c and not c.get("result", {}).get("passed")]


def _collect_errors(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cases if "error" in c]


def run_suite(
    limit: int,
    eval_model: str,
    require_question: bool,
    include_salary: bool,
    hide_company: bool,
    hide_company_ratio: float,
    seed: int | None,
    cdm_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    prompt_id, prompt_version = _resolve_prompt_cfg()

    all_files = _load_cdm_paths(cdm_dir)
    fixtures_found = len(all_files)
    if not all_files:
        raise FileNotFoundError(f"No CDM fixtures found in {cdm_dir}")

    cdm_files = all_files[:limit] if limit > 0 else all_files

    if seed is not None:
        random.seed(seed)

    _log(
        "[init] "
        f"prompt_id={prompt_id}, prompt_version={prompt_version}, eval_model={eval_model}, require_question={require_question}, "
        f"include_salary={include_salary}, hide_company={hide_company}, hide_company_ratio={hide_company_ratio}, seed={seed}"
    )
    _log(f"[gen] fixtures_found={fixtures_found}, requested_limit={limit}, actual_cases={len(cdm_files)}")

    generator = FirstTouchGenerator(prompt_id=prompt_id, prompt_version=prompt_version)
    client = OpenAI()

    cases: List[Dict[str, Any]] = []
    total_cases = len(cdm_files)
    company_leaks_count = 0

    for idx, cdm_path in enumerate(cdm_files, start=1):
        _log(f"[run] case {idx}/{total_cases} ({cdm_path.name})")

        try:
            cdm = json.loads(cdm_path.read_text(encoding="utf-8"))
            form_dict = to_input_form(cdm)

            original_company_name = str(form_dict.get("company_name") or "").strip() or None
            expected_work_mode = _expected_work_mode_from_cdm(cdm)

            company_hidden = bool(hide_company) or (hide_company_ratio > 0.0 and random.random() < hide_company_ratio)
            if company_hidden:
                form_dict["company_name"] = "СКРЫТО"

            input_form = InputForm(**form_dict)

            expected_facts_report = _build_expected_facts_report(input_form, include_salary=include_salary)
            expected_facts_eval = _build_expected_facts_for_eval(expected_facts_report, company_hidden=company_hidden)
            allowed_context_facts = _build_allowed_context_facts(input_form)
            candidate_source = str(allowed_context_facts.get("candidate_source") or "").strip()
            source_check = _check_candidate_source_possessive(candidate_source=candidate_source, message="", source_spec=None)

            required_keys = _required_keys(expected_facts_report=expected_facts_report, include_salary=include_salary)
            optional_keys = [k for k in expected_facts_eval.keys() if k not in required_keys]

            message = _normalize_text(_call_with_retries(lambda: generator.generate_message(input_form)))
            source_check = _check_candidate_source_possessive(candidate_source=candidate_source, message=message, source_spec=None)

            company_leaked = _company_name_present_when_hidden(company_hidden, original_company_name, message)
            if company_leaked:
                company_leaks_count += 1

            eval_result = evaluate_message(
                client=client,
                eval_model=eval_model,
                expected_facts=expected_facts_eval,
                allowed_context_facts=allowed_context_facts,
                message=message,
            )

            missing_required_keys = [k for k in required_keys if not bool(eval_result.facts_present.get(k, False))]
            missing_optional_keys = [k for k in optional_keys if not bool(eval_result.facts_present.get(k, False))]

            extra_numbers = _extra_numbers(expected_facts_report, message)

            fail_reasons: List[str] = []
            if missing_required_keys:
                fail_reasons.append("missing_required_keys")
            if extra_numbers:
                fail_reasons.append("extra_numbers")
            if require_question and not eval_result.question_present:
                fail_reasons.append("no_question")
            if company_leaked:
                fail_reasons.append("company_name_present_when_hidden")
            if not source_check["source_possessive_ok"]:
                fail_reasons.append("bad_candidate_source_possessive")

            passed = not fail_reasons
            passed_strict = bool(passed and not eval_result.hallucinated_facts)

            case = {
                "case_type": "cdm",
                "cdm_file": cdm_path.name,
                "company_hidden": company_hidden,
                "original_company_name": original_company_name,
                "expected_work_mode": expected_work_mode,
                "vacancy": {
                    "company_name": expected_facts_report.get("company_name", ""),
                    "vacancy_name": expected_facts_report.get("vacancy_name", ""),
                },
                "expected_facts": expected_facts_report,
                "message": message,
                "result": {
                    "passed": passed,
                    "passed_strict": passed_strict,
                    "question_present": eval_result.question_present,
                    "source_possessive_ok": source_check["source_possessive_ok"],
                    "missing_required_keys": missing_required_keys,
                    "missing_optional_keys": missing_optional_keys,
                    "missing_possessive_source_forms": source_check["missing_possessive_source_forms"],
                    "forbidden_source_forms_found": source_check["forbidden_source_forms_found"],
                    "hallucinated_facts": eval_result.hallucinated_facts,
                    "extra_numbers": extra_numbers,
                    "fail_reasons": fail_reasons,
                },
                "meta": {"comment": eval_result.comment},
            }
            cases.append(case)
            if passed:
                _log(f"  [ok] cdm={cdm_path.name} strict={passed_strict} question={eval_result.question_present}")
            else:
                _log(f"  [fail] cdm={cdm_path.name} reasons={fail_reasons}")

        except Exception as exc:
            cases.append({"case_type": "cdm", "cdm_file": cdm_path.name, "error": f"{type(exc).__name__}: {exc}"})
            _log(f"  [error] cdm={cdm_path.name} {type(exc).__name__}: {exc}")

    for idx, source_case in enumerate(SOURCE_OBJECT_CASES, start=1):
        _log(f"[run] source case {idx}/{len(SOURCE_OBJECT_CASES)} ({source_case['id']})")

        try:
            input_form = _build_source_possessive_input_form(str(source_case.get("candidate_source") or ""))

            expected_facts_report = _build_expected_facts_report(input_form, include_salary=include_salary)
            expected_facts_eval = _build_expected_facts_for_eval(expected_facts_report, company_hidden=False)
            allowed_context_facts = _build_allowed_context_facts(input_form)
            candidate_source = str(allowed_context_facts.get("candidate_source") or "").strip()

            required_keys = _required_keys(expected_facts_report=expected_facts_report, include_salary=include_salary)
            optional_keys = [k for k in expected_facts_eval.keys() if k not in required_keys]

            message = _normalize_text(_call_with_retries(lambda: generator.generate_message(input_form)))
            source_check = _check_candidate_source_possessive(
                candidate_source=candidate_source,
                message=message,
                source_spec=source_case,
            )

            eval_result = evaluate_message(
                client=client,
                eval_model=eval_model,
                expected_facts=expected_facts_eval,
                allowed_context_facts=allowed_context_facts,
                message=message,
            )

            missing_required_keys = [k for k in required_keys if not bool(eval_result.facts_present.get(k, False))]
            missing_optional_keys = [k for k in optional_keys if not bool(eval_result.facts_present.get(k, False))]
            extra_numbers = _extra_numbers(expected_facts_report, message)

            fail_reasons: List[str] = []
            if missing_required_keys:
                fail_reasons.append("missing_required_keys")
            if extra_numbers:
                fail_reasons.append("extra_numbers")
            if require_question and not eval_result.question_present:
                fail_reasons.append("no_question")
            if not source_check["source_possessive_ok"]:
                fail_reasons.append("bad_candidate_source_possessive")

            passed = not fail_reasons
            passed_strict = bool(passed and not eval_result.hallucinated_facts)

            case = {
                "case_type": "candidate_source_possessive",
                "case_id": source_case["id"],
                "candidate_source": source_case["candidate_source"],
                "source_expected_forms": source_case.get("allowed_forms", []),
                "company_hidden": False,
                "original_company_name": expected_facts_report.get("company_name", "") or None,
                "expected_work_mode": None,
                "vacancy": {
                    "company_name": expected_facts_report.get("company_name", ""),
                    "vacancy_name": expected_facts_report.get("vacancy_name", ""),
                },
                "expected_facts": expected_facts_report,
                "message": message,
                "result": {
                    "passed": passed,
                    "passed_strict": passed_strict,
                    "question_present": eval_result.question_present,
                    "source_possessive_ok": source_check["source_possessive_ok"],
                    "missing_required_keys": missing_required_keys,
                    "missing_optional_keys": missing_optional_keys,
                    "missing_possessive_source_forms": source_check["missing_possessive_source_forms"],
                    "forbidden_source_forms_found": source_check["forbidden_source_forms_found"],
                    "hallucinated_facts": eval_result.hallucinated_facts,
                    "extra_numbers": extra_numbers,
                    "fail_reasons": fail_reasons,
                },
                "meta": {"comment": eval_result.comment},
            }
            cases.append(case)
            if passed:
                _log(
                    f"  [ok] source_case={source_case['id']} "
                    f"source_possessive_ok={source_check['source_possessive_ok']}"
                )
            else:
                _log(
                    f"  [fail] source_case={source_case['id']} "
                    f"reasons={fail_reasons} "
                    f"missing={source_check['missing_possessive_source_forms']} "
                    f"forbidden={source_check['forbidden_source_forms_found']}"
                )

        except Exception as exc:
            cases.append(
                {
                    "case_type": "candidate_source_possessive",
                    "case_id": source_case["id"],
                    "candidate_source": source_case["candidate_source"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _log(f"  [error] source_case={source_case['id']} {type(exc).__name__}: {exc}")

    for idx, prompt_case in enumerate(PROMPT_RULE_CASES, start=1):
        _log(f"[run] prompt_rule case {idx}/{len(PROMPT_RULE_CASES)} ({prompt_case['id']})")

        try:
            input_form = _build_prompt_rule_input_form(prompt_case)
            expected_facts_report = _build_expected_facts_report(input_form, include_salary=include_salary)
            expected_facts_eval = _build_expected_facts_for_eval(expected_facts_report, company_hidden=False)
            allowed_context_facts = _build_allowed_context_facts(input_form)

            required_keys = _required_keys(expected_facts_report=expected_facts_report, include_salary=include_salary)
            optional_keys = [k for k in expected_facts_eval.keys() if k not in required_keys]

            message = _normalize_text(_call_with_retries(lambda: generator.generate_message(input_form)))
            eval_result = evaluate_message(
                client=client,
                eval_model=eval_model,
                expected_facts=expected_facts_eval,
                allowed_context_facts=allowed_context_facts,
                message=message,
            )

            missing_required_keys = [k for k in required_keys if not bool(eval_result.facts_present.get(k, False))]
            missing_optional_keys = [k for k in optional_keys if not bool(eval_result.facts_present.get(k, False))]
            extra_numbers = _extra_numbers(expected_facts_report, message)
            prompt_rule_result = _evaluate_prompt_rule_checks(
                case=prompt_case,
                message=message,
                eval_result=eval_result,
            )

            fail_reasons: List[str] = []
            if missing_required_keys:
                fail_reasons.append("missing_required_keys")
            if extra_numbers:
                fail_reasons.append("extra_numbers")
            if require_question and not eval_result.question_present:
                fail_reasons.append("no_question")
            fail_reasons.extend(prompt_rule_result["fail_reasons"])

            passed = not fail_reasons
            passed_strict = bool(passed and not eval_result.hallucinated_facts)

            case = {
                "case_type": "prompt_rule",
                "case_id": prompt_case["id"],
                "description": prompt_case.get("description", ""),
                "company_hidden": False,
                "original_company_name": expected_facts_report.get("company_name", "") or None,
                "expected_work_mode": None,
                "input_context": _prompt_rule_case_context(prompt_case),
                "vacancy": {
                    "company_name": expected_facts_report.get("company_name", ""),
                    "vacancy_name": expected_facts_report.get("vacancy_name", ""),
                },
                "expected_facts": expected_facts_report,
                "message": message,
                "result": {
                    "passed": passed,
                    "passed_strict": passed_strict,
                    "question_present": eval_result.question_present,
                    "missing_required_keys": missing_required_keys,
                    "missing_optional_keys": missing_optional_keys,
                    "hallucinated_facts": eval_result.hallucinated_facts,
                    "extra_numbers": extra_numbers,
                    "recruiter_name_absent_ok": prompt_rule_result["recruiter_name_absent_ok"],
                    "recruiter_name_present_ok": prompt_rule_result["recruiter_name_present_ok"],
                    "vacancy_reason_present_ok": prompt_rule_result["vacancy_reason_present_ok"],
                    "remote_priority_ok": prompt_rule_result["remote_priority_ok"],
                    "no_fake_remote_ok": prompt_rule_result["no_fake_remote_ok"],
                    "no_work_format_specified_ok": prompt_rule_result["no_work_format_specified_ok"],
                    "hybrid_not_overemphasized_ok": prompt_rule_result["hybrid_not_overemphasized_ok"],
                    "it_accreditation_priority_ok": prompt_rule_result["it_accreditation_priority_ok"],
                    "single_cta_ok": prompt_rule_result["single_cta_ok"],
                    "no_double_question_cta": prompt_rule_result["no_double_question_cta"],
                    "no_alternative_cta": prompt_rule_result["no_alternative_cta"],
                    "plain_message_only": prompt_rule_result["plain_message_only"],
                    "question_count": prompt_rule_result["question_count"],
                    "last_line_has_question": prompt_rule_result["last_line_has_question"],
                    "fail_reasons": fail_reasons,
                },
                "meta": {"comment": eval_result.comment},
            }
            cases.append(case)
            if passed:
                _log(
                    f"  [ok] prompt_rule={prompt_case['id']} "
                    f"cta={prompt_rule_result['single_cta_ok']} plain={prompt_rule_result['plain_message_only']}"
                )
            else:
                _log(f"  [fail] prompt_rule={prompt_case['id']} reasons={fail_reasons}")

        except Exception as exc:
            cases.append(
                {
                    "case_type": "prompt_rule",
                    "case_id": prompt_case["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _log(f"  [error] prompt_rule={prompt_case['id']} {type(exc).__name__}: {exc}")

    finished_at = datetime.datetime.now()

    failures = _collect_failures(cases)
    errors = _collect_errors(cases)
    summary = _compute_summary(cases=cases, requested_limit=limit, fixtures_found=fixtures_found, company_leaks_count=company_leaks_count)

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "prompt": {"prompt_id": prompt_id, "prompt_version": prompt_version},
        "eval_model": eval_model,
        "config": {
            "limit": limit,
            "cdm_dir": str(cdm_dir),
            "out_dir": str(out_dir),
            "require_question": require_question,
            "include_salary": include_salary,
            "hide_company": hide_company,
            "hide_company_ratio": hide_company_ratio,
            "seed": seed,
        },
        "definitions": _report_definitions(require_question=require_question),
        "summary": summary,
        "cases": cases,
        "failures": failures,
        "errors": errors,
    }

    ensure_dirs(out_dir)
    out_path = out_dir / f"first_touch_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        "[summary] "
        f"total_cases={summary['total_cases']} "
        f"passed={summary['passed_cases']} "
        f"failed={summary['failed_cases']} "
        f"errors={summary['errors_count']} "
        f"pass_rate={summary['pass_rate']:.2%}"
    )
    _log(f"[done] report saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="First-touch prompt test")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--cdm-dir", type=pathlib.Path, default=CDM_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPORTS_DIR)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)

    rq = parser.add_mutually_exclusive_group()
    rq.add_argument("--require-question", dest="require_question", action="store_true", help="Require a question CTA")
    rq.add_argument("--no-require-question", dest="require_question", action="store_false", help="Do not require a question CTA")
    parser.set_defaults(require_question=True)

    parser.add_argument("--include-salary", action="store_true", help="Include salary_range into expected facts")
    parser.add_argument("--hide-company", action="store_true", help='Force company_name to "СКРЫТО" in all cases')
    parser.add_argument("--hide-company-ratio", type=float, default=0.0, help="Randomly hide company in N% cases (0..1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for hide-company-ratio")

    args = parser.parse_args()

    report_path = run_suite(
        limit=args.limit,
        eval_model=args.eval_model,
        require_question=bool(args.require_question),
        include_salary=bool(args.include_salary),
        hide_company=bool(args.hide_company),
        hide_company_ratio=float(args.hide_company_ratio or 0.0),
        seed=args.seed,
        cdm_dir=args.cdm_dir,
        out_dir=args.out_dir,
    )
    print("Report ->", report_path)


if __name__ == "__main__":
    main()

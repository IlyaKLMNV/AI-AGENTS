from __future__ import annotations

from typing import Any, Dict, List


def _safe_list(value: List[Any] | None) -> List[Any]:
    return list(value or [])


def to_input_form(cdm: Dict[str, Any]) -> Dict[str, Any]:
    vacancy = cdm["vacancy"]
    candidate = cdm.get("candidate") or {}
    recruiter_name = candidate.get("recruiter_name") or "Рекрутер"
    candidate_name = candidate.get("candidate_name") or "Кандидат"
    return {
        "recruiter_name": recruiter_name,
        "company_name": vacancy["company_name"],
        "company_description": vacancy.get("company_description") or "",
        "vacancy_name": vacancy["title"],
        "vacancy_stack": vacancy.get("vacancy_stack") or "",
        "company_industry": vacancy.get("company_industry") or "",
        "vacancy_responsibilities": vacancy.get("responsibilities") or "",
        "vacancy_skills": _safe_list(vacancy.get("vacancy_skills")),
        "salary_range_from": vacancy.get("salary_range_from"),
        "salary_range_to": vacancy.get("salary_range_to"),
        "use_salary": bool(vacancy.get("salary_range_from") or vacancy.get("salary_range_to")),
        "formality": True,
        "candidate_name": candidate_name,
        "candidate_contacts": _safe_list(candidate.get("candidate_contacts")),
        "candidate_job_list": _safe_list(candidate.get("candidate_job_list")),
        "candidate_skills": _safe_list(candidate.get("candidate_skills")),
    }


def to_vacancy_info(cdm: Dict[str, Any]) -> Dict[str, Any]:
    vacancy = cdm["vacancy"]
    return {
        "title": vacancy["title"],
        "company_name": vacancy["company_name"],
        "responsibilities": vacancy.get("responsibilities") or "",
        "work_format": vacancy.get("work_format") or "",
        "location": vacancy.get("location") or "",
        "min_salary": str(vacancy["salary_range_from"]) if vacancy.get("salary_range_from") else "",
        "max_salary": str(vacancy["salary_range_to"]) if vacancy.get("salary_range_to") else "",
        "company_info": {
            "firm_description": vacancy.get("company_description") or "",
            "vacancy_url": "",
        },
        "questions": vacancy.get("questions") or "1. Ваш город?\n2. Ожидания по ЗП?",
    }


def names_from_cdm(cdm: Dict[str, Any]) -> Dict[str, str]:
    candidate = cdm.get("candidate") or {}
    return {
        "recruiter_name": candidate.get("recruiter_name") or "Рекрутер",
        "candidate_name": candidate.get("candidate_name") or "Кандидат",
    }

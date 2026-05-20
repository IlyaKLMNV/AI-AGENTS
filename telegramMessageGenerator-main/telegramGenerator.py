import os
import json
from typing import List, Optional

import openai


class AssistantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InputForm:
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
            vacancy_stack: Optional[str] = None
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

        self.formality = 'formal' if formality else 'informal'

        self.company_name = company_name
        self.vacancy_name = vacancy_name
        self.vacancy_responsibilities = vacancy_responsibilities

        self.company_industry = company_industry
        self.vacancy_skills = ', '.join(vacancy_skills)

        self.salary_range_from = salary_range_from
        self.salary_range_to = salary_range_to
        self.salary = self._get_salary(use_salary)

        self.company_description = company_description

        self.candidate_name = candidate_name

        self.candidate_contacts = self._social_accounts_found(candidate_contacts)
        self.candidate_skills = ', '.join(self._skills_search(candidate_skills, vacancy_skills))

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

        if self._social_search(accounts, 'ln'):
            result.append('профиль LinkedIn')

        if self._social_search(accounts, 'hh'):
            result.append('резюме HH')

        if self._social_search(accounts, 'github'):
            result.append('профиль GitHub')

        if self._social_search(accounts, 'mk'):
            result.append('резюме Хабр Карьеры')

        return result

    @staticmethod
    def _company_industry_search(jobs_list: List, search_term: str) -> bool:

        return any(category.get('title', '').lower() == search_term.lower()
                   for job in jobs_list
                   for category in job.get('company_norm', {}).get('categories', []))

    @staticmethod
    def _skills_search(skills_list: List, vacancy_skills: List, quantiles=(4.1, 3.1)) -> List:
        skill_set = set(vacancy_skills)
        quantile_set = set(quantiles)

        matching_skills = list(
            filter(lambda d: d.get('skill') in skill_set and d.get('quantile') in quantile_set, skills_list)
        )

        return [d.get('skill') for d in matching_skills]

    @staticmethod
    def _social_search(accounts: List, social: str) -> bool:
        return any(account['type'] == social for account in accounts)


class TelegramMessageGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key or os.getenv('OPENAI_API_KEY')

        self.system_prompt = os.getenv('FIRST_TOUCH_PROMPT_ID')
        self.system_prompt_version = os.getenv('FIRST_TOUCH_PROMPT_VERSION')
        if not self.system_prompt:
            raise RuntimeError('Missing FIRST_TOUCH_PROMPT_ID')

    def generate_message(self, input_form: InputForm) -> str:
        candidate_source = ''
        if len(input_form.candidate_contacts) > 0:
            candidate_source = input_form.candidate_contacts[0]

        payload = {
            'candidate_name': input_form.candidate_name,
            'recruiter_name': input_form.recruiter_name,
            'candidate_source': candidate_source,
            'reason_of_communication': input_form.reasons,
            'hiring_company_name': input_form.company_name,
            'vacancy_name': input_form.vacancy_name,
            'vacancy_responsibilities': input_form.vacancy_responsibilities,
            'message_formality': input_form.formality,
            'company_description': input_form.company_description,
            'vacancy_stack': input_form.vacancy_stack,
            'salary_range': input_form.salary
        }

        try:
            prompt = {
                'id': self.system_prompt,
            }
            if self.system_prompt_version:
                prompt['version'] = str(self.system_prompt_version)

            response = openai.responses.create(
                prompt=prompt,
                input=json.dumps(payload, ensure_ascii=False),
                text={'format': {'type': 'text'}}
            )

            if not response.output_text:
                raise AssistantError('Ответ от ассистента пустой')

            output = self._ending_check(response.output_text, 'С уважением')
            return output

        except Exception as e:
            raise AssistantError(f'Ошибка при получении ответа: {e}')

    @staticmethod
    def _ending_check(text: str, phrase: str) -> str:
        lines = text.split('\n')

        if len(lines) >= 2 and any(phrase.lower() in line.lower() for line in lines[-2:]):
            del lines[-2:]

        modified_text = '\n'.join(lines)

        return modified_text

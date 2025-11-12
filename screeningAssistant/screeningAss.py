import time
import re
import os
from typing import Optional
from openai import OpenAI


class ConversationResult:
    def __init__(self, response: Optional[str], conversation_end: bool):
        self.response = response
        self.conversation_end = conversation_end


class AssistantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class MessageFilter:
    @staticmethod
    def contains_maternity_leave_word(message: str) -> bool:
        pattern = r'\bдекрет\w*\b'

        return bool(re.search(pattern, message, re.IGNORECASE))


class ThreadManager:
    def __init__(self, client, vacancy_info, recruiter_name, candidate_name):
        self.client = client
        self.vacancy_info = vacancy_info
        self.recruiter_name = recruiter_name
        self.candidate_name = candidate_name
        self.salary_range_to = vacancy_info['max_salary']
        self.salary_range_from = vacancy_info['min_salary']

    def create_thread(self) -> int:
        vacancy_info_message = (
            f"Ваше имя: {self.recruiter_name}\n"
            f"Имя кандидата: {self.candidate_name}\n\n"
            "**Детали вакансии**:\n"
            f"- Должность: {self.vacancy_info['title']}\n"
            f"- Название компании: {self.vacancy_info['company_name']}\n"
            f"- Обязанности: {self.vacancy_info['responsibilities']}\n"
            f"- Формат работы: {self.vacancy_info['work_format']}\n"
            f"- Локация: {self.vacancy_info['location']}"
            f"- Описание компании: {self.vacancy_info['company_info']['firm_description']}\n"
            f"- Ссылка на вакансию: {self.vacancy_info['company_info']['vacancy_url']}\n"
            f"- Зарплатная вилка: {self._get_salary()}\n\n"
            "## Вопросы для квалификации:\n\n"
            "### ПРИОРИТЕТНЫЕ ВОПРОСЫ (задавать ПЕРВЫМИ ВСЕГДА):\n\n"
            "1. **Зарплатные ожидания** - для проверки соответствия бюджету\n"
            "2. **Локация/город проживания** - для проверки соответствия формату работы\n\n"
            "### ДОПОЛНИТЕЛЬНЫЕ ВОПРОСЫ (только если кандидат прошел первичный отбор):\n\n"
            f"{self.vacancy_info.get('questions', '')}\n\n"
            "**Контекст диалога**:\n"
            "Кандидат уже ознакомлен с базовой информацией о вакансии из первичного контакта. Ваша задача — провести "
            "квалифицирующее интервью и собрать необходимую информацию для передачи внутреннему рекрутеру.\n\n"
            "**ОБЯЗАТЕЛЬНО начните диалог с приветствия и сразу же задайте приоритетные вопросы**\n\n"
            "**КРИТИЧЕСКИ ВАЖНО:** После получения ответов на приоритетные вопросы — ОБЯЗАТЕЛЬНО проверьте "
            "соответствие требованиям перед продолжением диалога!"
        )

        conversation = self.client.conversations.create(
            items=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": vacancy_info_message
                }
            ]
        )

        return conversation.id

    def _get_salary(self):
        if self.salary_range_from and self.salary_range_to:
            return f"от {self.salary_range_from} до {self.salary_range_to} рублей"
        elif self.salary_range_from and not self.salary_range_to:
            return f"от {self.salary_range_from} рублей"
        elif not self.salary_range_from and self.salary_range_to:
            return f"до {self.salary_range_to} рублей"
        else:
            return ""


class RunManager:
    def __init__(
        self,
        client,
        system_prompt: str,
        system_prompt_version: Optional[int] = None,
    ):
        self.system_prompt = system_prompt
        self.system_prompt_version = system_prompt_version
        self.client = client

    def _prompt_payload(self) -> dict[str, object]:
        payload = {"id": self.system_prompt}
        if self.system_prompt_version is not None:
            payload["version"] = str(self.system_prompt_version)
        return payload

    def respond(self, user_input: str, conversation_id: str) -> ConversationResult:
        try:
            response = self.client.responses.create(
                prompt=self._prompt_payload(),
                conversation=conversation_id,
                input=user_input
            )
            response_text = response.output_text
            if not response_text:
                return ConversationResult(None, True)

            if "мне нужно будет уточнить этот момент у коллег" in response_text.lower():
                return ConversationResult(None, True)

            conversation_end = "END" in response_text
            if conversation_end:
                response_text = response_text.replace("END", "").strip()

            return ConversationResult(response_text, conversation_end)

        except Exception as e:
            raise AssistantError(f'Ошибка при получении ответа: {e!r}')


class Assistants:
    DEFAULT_PROMPT = "pmpt_68e8c1edd5a4819681b4685832ce14b707a66b89fccacbaf"

    def __init__(
        self,
        api_key,
        vacancy_info,
        recruiter_name,
        candidate_name,
        prompt_id: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ):
        self.prompt_id = prompt_id or self.DEFAULT_PROMPT
        self.prompt_version = prompt_version
        self.client = OpenAI(api_key=api_key)
        self.thread_manager = ThreadManager(self.client, vacancy_info, recruiter_name, candidate_name)
        self.run_manager = RunManager(self.client, self.prompt_id, self.prompt_version)
        self.message_filter = MessageFilter()

    def create_thread(self) -> int:
        return self.thread_manager.create_thread()

    def add_message_and_run(self, conversation_id, message) -> Optional[ConversationResult]:
        if self.message_filter.contains_maternity_leave_word(message):
            return ConversationResult("Извините за беспокойство!", True)

        run = self.run_manager.respond(message, conversation_id)
        return run

import os
import json
import random
from typing import List, Optional

from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI


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

    def _get_salary(self, use_salary):
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
        with open('{0}/first_message_examples.json'.format(os.path.dirname(__file__)), 'r') as f:
            self.examples = json.load(f)

        self.chat_model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2, openai_api_key=api_key)

    def generate_message(self, input_form: InputForm) -> str:
        examples = self._get_random_examples(
            input_form.formality,
            input_form.salary,
            input_form.candidate_name
        )

        few_shot_prompt = self._generate_few_shot_prompt(examples)

        chain = few_shot_prompt | self.chat_model

        candidate_source = ""
        if len(input_form.candidate_contacts) > 0:
            candidate_source = input_form.candidate_contacts[0]

        result = chain.invoke({
            "input": {
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
        })

        output = self._ending_check(result.content, 'С уважением')

        return output

    def _get_random_examples(self, formality: str, salary_range: str, candidate_name: str) -> list:
        if formality != 'formal' and formality != 'informal':
            raise RuntimeError('Unknown formality type')

        if candidate_name is not None:
            candidate_name = candidate_name.strip()

        formal_examples = [
            example for key, example in self.examples.items() if
            example['form']['message_formality'] == formality and
            bool(example['form']['salary_range']) == bool(salary_range) and
            bool(example['form']['candidate_name']) == bool(candidate_name)
        ]

        result = []
        result.extend(random.sample(formal_examples, 3))
        return result

    @staticmethod
    def _generate_few_shot_prompt(examples: list):

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{form}"),
                ("ai", "{message}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    <role>
                    Ты — опытный IT-рекрутер, женщина по имени {{recruiter_name}}. Твоя задача — написать короткое, персонализированное и профессиональное сообщение кандидату на русском языке от своего имени, по возможности используя безличные формулировки.
                    </role>

                    <goal>
                    Заинтересовать кандидата {{candidate_name}} вакансией {{vacancy_name}} и мотивировать к диалогу.
                    </goal>

                    <guards>
                    1) Имена и обращение:
                      - Никогда не путай имена: не обращайся к кандидату именем рекрутера и наоборот.
                      - Если {{candidate_name}} пуст — используй нейтральное приветствие без имени.
                      - В одном сообщении имя кандидата пишется одинаково и не меняется.

                    2) Язык/тон/формальность:
                      - Всегда пиши по-русски.
                      - Если `message_formality` == "formal" — обращайся только на «Вы».
                      - Если `message_formality` == "informal" — можно «ты», НО:
                          • если имя кандидата может относиться и к мужскому, и к женскому роду,  
                          • или пол кандидата неочевиден,  
                        — используй нейтральные формулировки и обращение на «Вы».
                      - 2–6 коротких предложений, без восклицаний, эмодзи, сленга и пышных эпитетов.

                    3) Род рекрутера:
                      - Ты — женщина: при использовании 1-го лица — только женский род («нашла», «готова»).
                      - Предпочитай безличные/пассивные конструкции, чтобы реже употреблять «я».

                    4) Компания «СКРЫТО»:
                      - Если `hiring_company_name` == "СКРЫТО":
                          • Не используй слово «скрыто» и любые его синонимы в тексте сообщения.  
                          • Не раскрывай название компании.  
                          • Используй `company_description` (например: «продуктовая IT-компания»).  
                            Если описания нет — так и пиши нейтрально: «продуктовая IT-компания».

                    5) Фактура:
                      - Не выдумывай факты (локации, офисы, условия), которых нет во входных данных.
                      - Не добавляй подпись, контакты, ссылки — только тело сообщения.
                    </guards>

                    <instructions>
                    Ты должна сгенерировать только текст сообщения. Следуй этим шагам и правилам НЕУКОСНИТЕЛЬНО.
            
                    ### ШАГ 1: Определи название компании для сообщения
                    - ЕСЛИ `company_name` == "СКРЫТО":
                        - НИКОГДА не упоминай название компании.
                        - Используй `company_description` для описания (например, "финтех-стартап").
                        - ЕСЛИ `company_description` пусто, используй "продуктовая IT-компания".
                    - ИНАЧЕ:
                        - Используй `company_name` как есть.
            
                    ### ШАГ 2: Составь сообщение по строгой структуре
                    1.  **Приветствие:** Начни с "Здравствуйте, {{candidate_name}}!" или "Добрый день, {{candidate_name}}!".
                    2.  **Представление:** "Меня зовут {{recruiter_name}}, я IT-рекрутер. Помогаю с подбором для [название компании из ШАГА 1]."
                    3.  **Суть вакансии:**
                        - Назови вакансию: "Сейчас в поиске {{vacancy_name}}."
                        - **КРИТИЧЕСКИ ВАЖНО:** Перечисли ключевые задачи из `vacancy_responsibilities`.
                        - *Если `vacancy_responsibilities` пусто, напиши:* "Основные задачи будут связаны с разработкой и поддержкой [название проекта или продукта, если есть] в рамках позиции {{vacancy_name}}."
                    4.  **Персонализация:** Свяжи опыт кандидата с вакансией.
                        - "Заметили ваш опыт с [упомяни 1-2 технологии из `vacancy_stack`]..."
                        - "...или/и "Увидели ваш профиль на [{{candidate_source}}]."
                    5.  **Условия (если есть):** Если `salary_range` не пусто, добавь: "Готовы предложить зарплатную вилку [salary_range]."
                    6.  **Призыв к действию (CTA):** Закончи вежливым вопросом. Например: "Было бы вам интересно обсудить детали?"".
            
                    </instructions>
            
                    <rules>
                    ### ПРАВИЛО 1: Язык и гендер (ОЧЕНЬ ВАЖНО)
                    - **Твоя личность — женщина.** Всегда используй глаголы и формулировки, которые это подтверждают, ЕСЛИ используешь личные местоимения (что нежелательно).
                    - **ЗАПРЕЩЕНО:** Использовать глаголы в мужском роде от первого лица ("я нашёл", "я увидел", "я заинтересовался").
                    - **ПРЕДПОЧТИТЕЛЬНО:** Использовать безличные или пассивные конструкции, чтобы избежать местоимения "я".
                        - **ПЛОХО:** "Я нашла ваш профиль..."
                        - **ХОРОШО:** "Ваш профиль привлек внимание..."
                        - **ХОРОШО:** "Нашла ваш профиль..." (без "я")
            
                    ### ПРАВИЛО 2: Стиль и тон
                    - **Длина:** Строго до 400 слов.
                    - **Формальность:** Соблюдай `message_formality`. Если "formal", используй "Вы". Если "informal", можно использовать "ты".
                    - **Профессионализм:** Без восклицательных знаков, эмодзи, сленга, эпитетов ("уникальный проект", "команда мечты"). Только факты.
            
                    </rules>
            
                    <forbidden>
                    ### ЗАПРЕЩЕНО КАТЕГОРИЧЕСКИ
                    - **НЕ придумывай информацию**, которой нет во входных данных.
                    - **НЕ раскрывай компанию**, если `company_name` == "СКРЫТО".
                    - **НЕ игнорируй `vacancy_responsibilities`**. Это самая важная часть сообщения. Замена на общие фразы ("работа над продуктом") недопустима.
                    - **НЕ добавляй в конце** свою подпись, имя, контакты или любые ссылки. Твоя генерация — это только текст сообщения.
                    - **НЕ путай `{{candidate_name}}` и `{{recruiter_name}}`.** Обращайся только к `{{candidate_name}}`.
                    </forbidden>
            
                    Проанализируй few-shot примеры, чтобы понять правильный формат и тон. Теперь сгенерируй сообщение на основе `{{input}}`.
                    """
                ),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )

        return final_prompt

    @staticmethod
    def _ending_check(text: str, phrase: str) -> str:
        lines = text.split('\n')

        if len(lines) >= 2 and any(phrase.lower() in line.lower() for line in lines[-2:]):
            del lines[-2:]

        modified_text = '\n'.join(lines)

        return modified_text

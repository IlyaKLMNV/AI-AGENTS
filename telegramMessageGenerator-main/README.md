## Пример использования

```python
from telegramGenerator import TelegramMessageGenerator, InputForm

OPENAI_API_KEY = ""  # Здесь нужно указать ключ API OpenAI
message_generator = TelegramMessageGenerator(OPENAI_API_KEY)

input_form = InputForm(
    recruiter_name="Александра",
    company_name="Анкор",
    vacancy_name="Junior Python developer",
    company_industry="B2B",
    vacancy_skills=["Python", "Tornado"],
    vacancy_responsibilities="",
    salary_range_from=1000,
    salary_range_to=1200,
    candidate_name=EXAMPLE_METAPROFILE["firstNameCyrillic"],
    candidate_contacts=EXAMPLE_METAPROFILE["accounts"],
    candidate_job_list=EXAMPLE_METAPROFILE["positions"],
    candidate_skills=EXAMPLE_METAPROFILE["skills"],
    formality=True,
    company_description="",
    vacancy_stack="",
)

# генерируем сообщение
result = message_generator.generate_message(input_form)
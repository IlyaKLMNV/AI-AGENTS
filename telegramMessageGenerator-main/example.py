from telegramGenerator import TelegramMessageGenerator, InputForm

OPENAI_API_KEY = 'api_key'
message_generator = TelegramMessageGenerator(OPENAI_API_KEY)

input_form = InputForm(
                recruiter_name="Александра",
                company_name="Анкор",
                vacancy_name="Junior Python developer",
                company_industry="B2B",
                vacancy_skills=['Python', 'Tornado'],
                salary_range_from=1000,
                salary_range_to=1200,
                candidate_name="Алексей",
                candidate_contacts=[],
                candidate_job_list=[],
                candidate_skills=[],
                formality=True,
                company_description="")

# генерируем имейл и его же можем выводить
result = message_generator.generate_message(input_form)
print(result)
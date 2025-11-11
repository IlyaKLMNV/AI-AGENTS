from screeningAss import Assistants

vacancy_info = {
        'title': 'Senior ML-инженер',
        'company_name': 'G‑Банк',
        'responsibilities': 'Что нужно будет делать:\n'
                            'Изучать научные статьи, генерировать гипотезы,'
                            'ставить на их основе эксперименты и доносить результат до команды\n'
                            'Улучшать качество моделей в различных сценариях'
                            'Ускорять работу моделей, применяя современные методы оптимизации и построения архитектуры\n'
                            'Писать воспроизводимый код, оформлять эксперименты в воспроизводимые пайплайны,'
                            'включающие разметку и обработку данных, обучение моделей и валидацию системы в целом;',
        'work_format': 'удаленный',
        'min_salary': "400000",
        'max_salary': None,
        'location': 'Москва',
        'company_info':{
        'firm_description': 'G-Банк - системообразующий российский банк. Команда G-Банка в поиске Senior ML-инженера в Центр технологий искусственного интеллекта, который создает AI-технологии для развития финансовой экосистемы G‑Банка.',
        'vacancy_url': 'https://www.gbank.ru/career/it/ml/ml-inzhener/'},
        'questions':'Какие ваши зарплатные ожидания?\n'
                    'В какой стране или городе вы проживаете?\n'
                    'Расскажите о проекте, в котором вы использовали ML и о результатах этого проекта?\n'
                    'В чем больше опыта RecSys / NLP?\n'
                    'Есть ли опыт вывода в прод?'
}

api_key = "your_api_key"
recruiter_name = "Анна"
candidate_name = "Алексей"

assistant = Assistants(api_key, vacancy_info, recruiter_name, candidate_name)
thread = assistant.create_thread()
run = assistant.add_message_and_run(thread, 'Ааххахаха неее, ты бот, 2 + 2 сколько?')

if run.response:
    if not run.conversation_end:
        print(run.response)
    else:
        print(run.response)
        print("Conversation ended")
else:
    print("Conversation ended")
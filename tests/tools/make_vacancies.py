from __future__ import annotations

import argparse
import json
import pathlib
import random


SAMPLES = [
    dict(
        title="Product Lead Risk Platform",
        company_name="FinCore",
        company_description="Финтех-команда строит риск-платформу и antifraud-инструменты для банков.",
        company_industry="FinTech",
        location="Москва",
        work_format="hybrid",
        salary_range_from=350000,
        salary_range_to=450000,
        responsibilities="Стратегия риск-платформы, product-мониторинг и развитие команды 8 человек.",
        vacancy_stack="Python, SQL, Airflow, Kafka",
        vacancy_skills=["Risk scoring", "Product leadership", "Data products"],
        questions="1. Где находитесь сейчас?\n2. Какая net-компенсация комфортна?\n3. Опыт работы с риск-моделями?",
        first_message_template=(
            "Привет, {candidate_name}! На связи {recruiter_name} из {company}. "
            "Веду поиск Product Lead для риск-платформы. Чтобы понять, совпадаем ли, "
            "подскажите, пожалуйста, город/формат работы и желаемый net-уровень дохода."
        ),
    ),
    dict(
        title="Head of Operations Platform",
        company_name="Aurora Logistics",
        company_description="Маркетплейс логистики для e-commerce, строим собственную OPS-платформу.",
        company_industry="Logistics",
        location="Санкт-Петербург",
        work_format="hybrid",
        salary_range_from=320000,
        salary_range_to=420000,
        responsibilities="Оптимизация процессов, внедрение инструментов SLA-контроля, управление командой ops.",
        vacancy_stack="Python, BI, Kubernetes",
        vacancy_skills=["Ops leadership", "Automation", "BI"],
        questions="1. В каком городе живёте/готовы ли к СПб?\n2. Зарплата (net)?\n3. Опыт управления операциями?",
        first_message_template=(
            "Добрый день, {candidate_name}! {recruiter_name} на связи, веду подбор Head of Operations в {company}. "
            "Позиция в СПб (гибрид). Сразу уточню: где вы сейчас находитесь и какая net-вилка для вас актуальна?"
        ),
    ),
    dict(
        title="Lead Data Scientist (Payments)",
        company_name="NeuronLab",
        company_description="R&D-команда крупного банка, строим antifraud и платежные модели.",
        company_industry="Banking",
        location="Москва",
        work_format="office",
        salary_range_from=400000,
        salary_range_to=520000,
        responsibilities="Разработка antifraud моделей, эксперименты, наставничество DS-команды.",
        vacancy_stack="Python, Spark, MLflow",
        vacancy_skills=["ML", "Spark", "Antifraud"],
        questions="1. Город/готовность к офису?\n2. Зарплатные ожидания?\n3. Опыт antifraud/платежей?",
        first_message_template=(
            "Здравствуйте, {candidate_name}! Меня зовут {recruiter_name}, я помогаю команде {company}. "
            "Прежде чем углубляться в детали Lead DS-вакансии, подскажите, пожалуйста, где вы сейчас находитесь "
            "и на какой net-уровень компенсации ориентируетесь."
        ),
    ),
    dict(
        title="QA Automation Lead",
        company_name="HelixCare",
        company_description="Digital-медицина и телемедицина, ищем лида автотестирования.",
        company_industry="Healthcare",
        location="Казань",
        work_format="remote",
        salary_range_from=250000,
        salary_range_to=320000,
        responsibilities="Стратегия автотестов, постановка QA-процессов, наставничество инженеров.",
        vacancy_stack="Python, Playwright, Jenkins",
        vacancy_skills=["Python QA", "Playwright", "CI/CD"],
        questions="1. Где находитесь? Remote ок?\n2. Net-ожидания?\n3. Опыт автотестов на Python?",
        first_message_template=(
            "Привет, {candidate_name}! {recruiter_name} из {company}. У нас полностью remote позиция QA Automation Lead. "
            "Сразу попрошу: поделитесь городом/часовым поясом и желаемой net-компенсацией, чтобы понять совпадение ожиданий."
        ),
    ),
    dict(
        title="Site Reliability Manager",
        company_name="ZenithPay",
        company_description="Платёжная платформа для b2b-клиентов.",
        company_industry="Payments",
        location="Новосибирск",
        work_format="hybrid",
        salary_range_from=360000,
        salary_range_to=480000,
        responsibilities="Оркестрация SRE-практик, управление SLA и команда из 6 инженеров.",
        vacancy_stack="Go, Kubernetes, Observability",
        vacancy_skills=["SRE", "Go", "Monitoring"],
        questions="1. Где находитесь? Готовы к новосибирскому офису?\n2. Ожидаемая компенсация?\n3. Опыт построения SRE-процессов?",
        first_message_template=(
            "Здравствуйте, {candidate_name}! Это {recruiter_name}, управляю подбором в {company}. "
            "Позиция Site Reliability Manager предполагает гибрид в Новосибирске. "
            "Сразу уточню город и желаемый net-доход — это ключевые параметры на старте."
        ),
    ),
    # --- новые примеры ---
    dict(
        title="Senior Backend Engineer (Marketplace)",
        company_name="Mercury Market",
        company_description="Маркетплейс с фокусом на b2c, команда строит ядро платёжного и каталожного сервисов.",
        company_industry="E-commerce",
        location="Москва",
        work_format="hybrid",
        salary_range_from=270000,
        salary_range_to=350000,
        responsibilities="Разработка backend-сервисов, интеграции с платёжными провайдерами, код-ревью.",
        vacancy_stack="Python, FastAPI, PostgreSQL, Kafka",
        vacancy_skills=["Backend", "FastAPI", "PostgreSQL"],
        questions="1. В каком городе вы сейчас?\n2. Насколько комфортен гибридный формат?\n3. Опыт с высоконагруженными сервисами?",
        first_message_template=(
            "Здравствуйте, {candidate_name}! {recruiter_name} из {company}. "
            "Ищем Senior Backend Engineer под ядро маркетплейса. "
            "Подскажите, пожалуйста, где вы сейчас находитесь, насколько комфортен гибридный формат "
            "и есть ли у вас опыт с высоконагруженными сервисами?"
        ),
    ),
    dict(
        title="Head of Data Platform",
        company_name="NovaTech",
        company_description="Технологическая компания строит единую data-платформу для продуктовых команд.",
        company_industry="Tech",
        location="Москва",
        work_format="office",
        salary_range_from=420000,
        salary_range_to=550000,
        responsibilities="Стратегия data-платформы, развитие команды, взаимодействие с CPO/CTO.",
        vacancy_stack="Python, Spark, Airflow, ClickHouse",
        vacancy_skills=["Data Platform", "Team Management", "Spark"],
        questions="1. В каком городе вы находитесь?\n2. Готовы к офисному формату?\n3. Опыт построения data-платформ?",
        first_message_template=(
            "Здравствуйте, {candidate_name}! На связи {recruiter_name} из {company}. "
            "Мы ищем Head of Data Platform. "
            "Скажите, пожалуйста, в каком вы сейчас городе, готовы ли к офисному формату "
            "и есть ли у вас опыт построения data-платформ?"
        ),
    ),
    dict(
        title="Lead Product Manager (B2B SaaS)",
        company_name="CloudMetric",
        company_description="B2B SaaS-платформа для мониторинга и алёртинга инфраструктуры.",
        company_industry="SaaS",
        location="Удалённо",
        work_format="remote",
        salary_range_from=280000,
        salary_range_to=360000,
        responsibilities="Развитие B2B-продукта, интервью с клиентами, постановка задач команде разработки.",
        vacancy_stack="Product, Analytics, SaaS",
        vacancy_skills=["Product management", "B2B", "Discovery"],
        questions="1. В каком городе вы находитесь?\n2. Насколько важна для вас фиксированная ставка/бонус?\n3. Опыт работы с B2B SaaS?",
        first_message_template=(
            "Привет, {candidate_name}! {recruiter_name} из {company}. "
            "У нас Lead Product Manager на B2B SaaS-платформу. "
            "Расскажите, пожалуйста, из какого вы города, насколько важен баланс фикс/бонус "
            "и есть ли опыт работы с B2B SaaS-продуктами?"
        ),
    ),
    dict(
        title="Engineering Manager (Analytics)",
        company_name="Insightly",
        company_description="Продуктовая компания строит решения для аналитики поведения пользователей.",
        company_industry="Analytics",
        location="Санкт-Петербург",
        work_format="hybrid",
        salary_range_from=300000,
        salary_range_to=380000,
        responsibilities="Управление командой инженеров, развитие аналитической платформы, найм.",
        vacancy_stack="Python, ETL, DWH",
        vacancy_skills=["People management", "ETL", "DWH"],
        questions="1. В каком вы городе и открыты ли к СПб?\n2. Ожидаемая вилка net?\n3. Опыт управления инженерными командами?",
        first_message_template=(
            "Добрый день, {candidate_name}! {recruiter_name} из {company}. "
            "В команду аналитической платформы ищем Engineering Manager. "
            "Подскажите, пожалуйста, в каком вы сейчас городе и рассматриваете ли СПб, "
            "какая net-вилка для вас комфортна и есть ли опыт управления инженерными командами?"
        ),
    ),
    dict(
        title="Senior ML Engineer (Computer Vision)",
        company_name="VisionLab",
        company_description="R&D-команда делает CV-решения для индустриальных клиентов.",
        company_industry="AI",
        location="Екатеринбург",
        work_format="hybrid",
        salary_range_from=320000,
        salary_range_to=430000,
        responsibilities="Разработка и продакшн ML-моделей, участие в ресёрче, оптимизация inference.",
        vacancy_stack="Python, PyTorch, MLflow",
        vacancy_skills=["Computer Vision", "PyTorch", "ML"],
        questions="1. В каком вы городе и готовы ли к Екб?\n2. Ожидаемая компенсация?\n3. Опыт в computer vision?",
        first_message_template=(
            "Здравствуйте, {candidate_name}! {recruiter_name} из {company}. "
            "Мы ищем Senior ML Engineer в направление computer vision. "
            "Расскажите, пожалуйста, в каком вы сейчас городе и готовы ли к формату с Екб, "
            "какая компенсация для вас комфортна и есть ли у вас опыт в CV-проектах?"
        ),
    ),
]


def make_cdm(recruiter_name: str = "Варя", candidate_name: str = "Кандидат") -> dict:
    sample = random.choice(SAMPLES).copy()
    template = sample.pop("first_message_template")
    return {
        "vacancy": sample,
        "first_message_template": template,
        "candidate": {
            "recruiter_name": recruiter_name,
            "candidate_contacts": [],
            "candidate_job_list": [],
            "candidate_skills": [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tests/fixtures/cdm")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(args.n):
        cdm = make_cdm()
        path = out_dir / f"cdm_{idx + 1:02d}.json"
        path.write_text(json.dumps(cdm, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", path)


if __name__ == "__main__":
    main()

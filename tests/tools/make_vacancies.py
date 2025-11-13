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
]


def make_cdm(recruiter_name: str = "Варя", candidate_name: str = "Кандидат") -> dict:
    sample = random.choice(SAMPLES).copy()
    template = sample.pop("first_message_template")
    return {
        "vacancy": sample,
        "first_message_template": template,
        "candidate": {
            "recruiter_name": recruiter_name,
            "candidate_name": candidate_name,
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

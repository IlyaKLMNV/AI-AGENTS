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
        vacancy_skills="Risk scoring, Product leadership, Data products",
        questions="1. Где находитесь сейчас?\n2. Какая net-компенсация комфортна?\n3. Опыт работы с риск-моделями?",
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
        vacancy_skills="Ops leadership, Automation, BI",
        questions="1. В каком городе живёте/готовы ли к СПб?\n2. Зарплата (net)?\n3. Опыт управления операциями?",
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
        vacancy_skills="ML, Spark, Antifraud",
        questions="1. Город/готовность к офису?\n2. Зарплатные ожидания?\n3. Опыт antifraud/платежей?",
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
        vacancy_skills="Python QA, Playwright, CI/CD",
        questions="1. Где находитесь? Remote ок?\n2. Net-ожидания?\n3. Опыт автотестов на Python?",
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
        vacancy_skills="SRE, Go, Monitoring",
        questions="1. Где находитесь? Готовы к новосибирскому офису?\n2. Ожидаемая компенсация?\n3. Опыт построения SRE-процессов?",
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
        vacancy_skills="Backend, FastAPI, PostgreSQL",
        questions="1. В каком городе вы сейчас?\n2. Насколько комфортен гибридный формат?\n3. Опыт с высоконагруженными сервисами?",
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
        vacancy_skills="Data Platform, Team Management, Spark",
        questions="1. В каком городе вы находитесь?\n2. Готовы к офисному формату?\n3. Опыт построения data-платформ?",
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
        vacancy_skills="Product management, B2B, Discovery",
        questions="1. В каком городе вы находитесь?\n2. Насколько важна для вас фиксированная ставка/бонус?\n3. Опыт работы с B2B SaaS?",
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
        vacancy_skills="People management, ETL, DWH",
        questions="1. В каком вы городе и открыты ли к СПб?\n2. Ожидаемая вилка net?\n3. Опыт управления инженерными командами?",
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
        vacancy_skills="Computer Vision, PyTorch, ML",
        questions="1. В каком вы городе и готовы ли к Екб?\n2. Ожидаемая компенсация?\n3. Опыт в computer vision?",
    ),
]


def _sample_candidate_contacts() -> list[dict]:
    sources = [
        {"type": "ln", "url": "https://linkedin.com/in/example"},
        {"type": "hh", "url": "https://hh.ru/resume/example"},
        {"type": "github", "url": "https://github.com/example"},
        {"type": "mk", "url": "https://career.habr.com/example"},
    ]
    return [random.choice(sources)]


def _sample_candidate_job_list(company_industry: str) -> list[dict]:
    industry = company_industry or "Tech"
    return [
        {
            "title": "Senior Engineer",
            "company": "ExampleCorp",
            "company_norm": {"categories": [{"title": industry}]},
        }
    ]


def _parse_vacancy_skills(vacancy_skills: list | str | None) -> list[str]:
    if not vacancy_skills:
        return []
    if isinstance(vacancy_skills, str):
        return [skill.strip() for skill in vacancy_skills.split(",") if skill.strip()]
    return [str(skill).strip() for skill in vacancy_skills if str(skill).strip()]


def _sample_candidate_skills(vacancy_skills: list | str | None) -> list[dict]:
    skills: list[dict] = []
    for skill in _parse_vacancy_skills(vacancy_skills)[:2]:
        skills.append({"skill": skill, "quantile": 4.1})
    skills.append({"skill": "Communication", "quantile": 2.0})
    return skills


def make_cdm(recruiter_name: str = "Варя", candidate_name: str = "Вадим") -> dict:
    sample = random.choice(SAMPLES).copy()
    vacancy_skills = sample.get("vacancy_skills") or []
    company_industry = sample.get("company_industry") or ""
    return {
        "vacancy": sample,
        "candidate": {
            "recruiter_name": recruiter_name,
            "candidate_name": candidate_name,
            "candidate_contacts": _sample_candidate_contacts(),
            "candidate_job_list": _sample_candidate_job_list(company_industry),
            "candidate_skills": _sample_candidate_skills(vacancy_skills),
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

"""Контекст Наблюдателя: чего в нём нет и откуда берётся гео-ограничение.

Два свойства, каждое из которых уже ломалось. Вилки в контексте быть НЕ должно (П4) — иначе секрет
лежит там, где его потом запрещают называть. Гео-ограничение пишут прямо в «Локацию», а не в
отдельное поле: прогон 28.08, сценарий 10 — код искал поле, не находил и не отсеивал.
"""

from typing import List

from ..policy.context import build_observer_context, facts_from_context, has_geo_restriction
from .collect import Checks, Row

VAC = {
    "title": "Python Backend Developer",
    "company_name": "ExampleSoft",
    "responsibilities": "Микросервисы",
    "work_format": "remote",
    "location": "Москва",
    "company_info": {"firm_description": "b2b", "vacancy_url": "https://example.com/v"},
    "min_salary": 200000,
    "max_salary": 280000,
    "questions": "- A\n- B",
}


def checks() -> List[Row]:
    c = Checks()

    ctx = build_observer_context("Анна", "Иван", "резюме HH", VAC)
    c.add("вилки в контексте Наблюдателя НЕТ (П4)",
          "Зарплатная вилка" not in ctx and "200000" not in ctx and "280000" not in ctx,
          ctx[:0] or "")
    c.add("факты вакансии на месте",
          "Должность: Python Backend Developer" in ctx
          and "Ссылка на вакансию: https://example.com/v" in ctx
          and "Источник контакта кандидата: резюме HH" in ctx)
    c.add("строки гео-ограничения нет, когда его не задали", "Гео-ограничение" not in ctx)
    ctx_geo = build_observer_context("Анна", "Иван", "", dict(VAC, geo_restriction="только РФ"))
    c.add("заданное гео-ограничение попадает в контекст", "Гео-ограничение: только РФ" in ctx_geo)

    # --- ограничение читается и из отдельного поля, и из текста локации ---
    c.add("поле geo_restriction включает ограничение",
          has_geo_restriction(dict(VAC, geo_restriction="только РФ")))
    c.add("формулировка в локации включает ограничение",
          has_geo_restriction(dict(VAC, location="Россия, только РФ (работа из-за рубежа невозможна)")))
    c.add("ограничение по часовому поясу тоже считается",
          has_geo_restriction(dict(VAC, location="Москва, часовой пояс не более +2 к МСК")))
    c.add("просто город ограничением НЕ является", not has_geo_restriction(VAC))
    c.add("пустая вакансия ограничением не является", not has_geo_restriction({}))

    # --- обратный разбор: гарды берут канонический URL и скрытость отсюда ---
    facts = facts_from_context(ctx)
    c.add("канонический URL восстанавливается из контекста",
          (facts.get("company_info") or {}).get("vacancy_url") == "https://example.com/v",
          str(facts.get("company_info")))
    c.add("название компании восстанавливается", facts.get("company_name") == "ExampleSoft",
          str(facts.get("company_name")))
    hidden = build_observer_context("Анна", "Иван", "", dict(VAC, company_name=""))
    c.add("пустая компания подаётся как СКРЫТО",
          facts_from_context(hidden).get("company_name") == "СКРЫТО",
          str(facts_from_context(hidden).get("company_name")))

    return c.rows

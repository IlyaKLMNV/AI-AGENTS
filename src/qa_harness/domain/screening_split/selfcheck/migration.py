"""Ленивая миграция документа под новое ядро (решение Р7).

Самая опасная часть жёсткой подмены: если вилка не доедет до типизированного поля, зарплатный отсев
отключится ТИХО — `compare_with_band` без границ всегда возвращает «проходит», и в отчётности это
никак не видно. Поэтому проверки живут в гейте, а не в скретчпаде.
"""

from typing import List

from .. import context as ctx_mod
from .. import state as state_model
from ..policy import DecideContext, decide, migration
from ..policy.observation import Observation
from .collect import Checks, Row

VAC = {"title": "Python Backend Developer", "company_name": "ExampleSoft",
       "responsibilities": "Микросервисы", "work_format": "remote", "location": "Москва",
       "company_info": {"firm_description": "b2b", "vacancy_url": "https://example.com/v"},
       "min_salary": 200000, "max_salary": 280000, "questions": "- A\n- B"}


def _old_doc(**over) -> dict:
    """Документ, каким его создавал СТАРЫЙ движок: контекст строкой с вилкой, band пустой."""
    doc = {
        "conversation_id": "c1", "engine": "split", "finished": False,
        "context": ctx_mod.build_context("Анна", "Иван", "резюме HH", VAC),
        "location": "Москва", "contact_source": "резюме HH",
        "salary_band": {},
        "state": state_model.init_state("remote", "A\nB"),
    }
    doc.update(over)
    return doc


def checks() -> List[Row]:
    c = Checks()

    # --- 1. вилка разбирается ДО вырезания строки ---
    d = _old_doc()
    c.add("исходный документ действительно нёс вилку в контексте", "Зарплатная вилка" in d["context"])
    rep = migration.upgrade(d)
    c.add("вилка разобрана из контекста",
          d["salary_band"] == {"min": 200000, "max": 280000, "currency": "RUB"}, str(d["salary_band"]))
    c.add("источник вилки помечен", rep["band_source"] == "context", rep["band_source"])
    c.add("строка вилки вырезана", "Зарплатная вилка" not in d["context"])
    c.add("остальной контекст цел", "Должность: Python Backend Developer" in d["context"]
          and "Ссылка на вакансию: https://example.com/v" in d["context"])

    # --- 2. идемпотентность ---
    before = dict(d)
    rep2 = migration.upgrade(d)
    c.add("повторная миграция — no-op",
          rep2["migrated"] is False and d["context"] == before["context"])

    # --- 3. документ с уже типизированной вилкой не перетирается ---
    d3 = _old_doc(salary_band={"min": 100000, "max": 150000})
    migration.upgrade(d3)
    c.add("готовая вилка не перезаписана", d3["salary_band"]["min"] == 100000, str(d3["salary_band"]))
    c.add("валюта проставлена по умолчанию", d3["salary_band"]["currency"] == "RUB")

    # --- 4. формы вилки: только «от», только «до» ---
    v_from = dict(VAC, min_salary=200000, max_salary=None)
    d4 = _old_doc(context=ctx_mod.build_context("А", "И", "", v_from))
    migration.upgrade(d4)
    c.add("форма «от X»", d4["salary_band"] == {"min": 200000, "max": None, "currency": "RUB"},
          str(d4["salary_band"]))
    v_to = dict(VAC, min_salary=None, max_salary=280000)
    d5 = _old_doc(context=ctx_mod.build_context("А", "И", "", v_to))
    migration.upgrade(d5)
    c.add("форма «до Y»", d5["salary_band"] == {"min": None, "max": 280000, "currency": "RUB"},
          str(d5["salary_band"]))

    # --- 5. вилки нет вовсе: провал разбора ВИДЕН, а не молчит ---
    v_none = dict(VAC, min_salary=None, max_salary=None)
    d6 = _old_doc(context=ctx_mod.build_context("А", "И", "", v_none))
    rep6 = migration.upgrade(d6)
    c.add("вилки нет → band_unparsed", rep6["band_unparsed"] is True and not d6["salary_band"],
          str(rep6))
    c.add("пустая строка вилки всё равно вырезана", "Зарплатная вилка" not in d6["context"])

    # --- 6. last_asking обнуляется, счётчики дополняются ---
    d7 = _old_doc()
    d7["state"]["last_asking"] = "salary"
    d7["state"].pop("no_progress", None)
    d7["state"]["counters"].pop("contact_source", None)
    migration.upgrade(d7)
    c.add("last_asking обнулён", d7["state"]["last_asking"] is None)
    c.add("no_progress добавлен", d7["state"]["no_progress"] == 0)
    c.add("недостающий счётчик добавлен", d7["state"]["counters"]["contact_source"] == 0)
    c.add("схема помечена", d7["schema"] == migration.SCHEMA_VERSION)

    # --- 7. накопленное состояние НЕ теряется ---
    d8 = _old_doc()
    d8["state"]["salary"] = "closed"
    d8["state"]["candidate_city"] = "Казань"
    d8["state"]["questions"][0]["status"] = "closed"
    d8["state"]["counters"]["pause"] = 2
    migration.upgrade(d8)
    c.add("собранное сохранено", d8["state"]["salary"] == "closed"
          and d8["state"]["candidate_city"] == "Казань"
          and d8["state"]["questions"][0]["status"] == "closed"
          and d8["state"]["counters"]["pause"] == 2)

    # --- 8. мигрированный документ реально даёт отсев по деньгам ---
    d9 = _old_doc()
    migration.upgrade(d9)
    band = d9["salary_band"]
    obs = Observation()
    obs.salary_claim = {"subject": "own_expectation", "form": "exact",
                        "amount_min": 400, "amount_max": 400, "scale": "thousand",
                        "currency": "RUB", "period": "month", "tax": "net", "quote": "400 тысяч"}
    plan = decide(d9["state"], obs, "Ожидаю 400 тысяч на руки",
                  DecideContext(band_min=band["min"], band_max=band["max"], location="Москва"))
    c.add("после миграции отсев по деньгам работает",
          plan.reason_code == "KO_SALARY" and plan.end, f"{plan.reason_code} end={plan.end}")

    # То же БЕЗ миграции — демонстрация того, что чинится: вилки нет → отсев молча отключён.
    d10 = _old_doc()
    plan10 = decide(d10["state"], obs, "Ожидаю 400 тысяч на руки",
                    DecideContext(band_min=None, band_max=None, location="Москва"))
    c.add("без миграции отсева НЕ было бы (то, что чиним)",
          plan10.reason_code != "KO_SALARY", plan10.reason_code)

    return c.rows

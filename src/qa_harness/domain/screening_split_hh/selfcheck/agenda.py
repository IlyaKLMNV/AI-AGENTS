"""Повестка hh-канала: мультиформат, разъездной, локация (Р18).

Это ЕДИНСТВЕННЫЙ гейт на канальную дельту нового ядра: живой сценарий её не добывает (нужно заставить
кандидата отказаться ровно от всех допустимых форматов и ровно в нужном порядке). Проверяется то, чего
нет в TG: «подходит хотя бы один формат», выбор между `KO_FORMAT` и `KO_LOCATION`, отдельная ветка
разъездного формата и то, что переход с одного формата на другой не считается переспросом.
"""

from typing import List

from qa_harness.domain.screening_split.selfcheck.collect import Checks, Row

from .. import state as hh_state
from ..policy import DecideContext, Observation, decide
from ..policy import state_for_prompt as hh_projection

CTX = DecideContext(band_min=200000, band_max=280000, location="Москва")


def _obs(*, formats=(), relocation=None, geo=False, city=None, signals=()) -> Observation:
    o = Observation()
    o.facts = {"candidate_city": city,
               "formats_ready": [{"format": f, "ready": r} for f, r in formats],
               "relocation_ready": relocation, "geo_blocked": geo}
    o.signals = list(signals)
    return o


def _ready(allowed, **over) -> dict:
    """Состояние с закрытой зарплатой И закрытым городом: секции ниже проверяют формат.

    Город с решения Р18 — отдельный пункт повестки и стоит ПЕРЕД форматом, поэтому без этого дефолта
    каждая проверка мультиформата упиралась бы в вопрос про город. Проверки самого города передают
    `city_check="pending"` явно.
    """
    st = hh_state.init_state(allowed, "- Опыт с Python?")
    st["salary"] = "closed"
    st["city_check"] = "closed"
    st.update(over)
    return st


def checks() -> List[Row]:
    c = Checks()

    # --- 1. инициализация проверок по таблице мультиформата ---
    st_both = hh_state.init_state(["ON_SITE", "HYBRID"], "- A")
    st_remote = hh_state.init_state(["REMOTE", "ON_SITE"], "- A")
    st_field = hh_state.init_state(["FIELD_WORK"], "- A")
    c.add("init: [ON_SITE,HYBRID] → format pending / field n/a",
          st_both["format_check"] == "pending" and st_both["field_work_check"] == "n/a")
    c.add("init: REMOTE среди допустимых снимает проверку формата",
          st_remote["format_check"] == "n/a", st_remote["format_check"])
    c.add("init: [FIELD_WORK] → format n/a / field pending",
          st_field["format_check"] == "n/a" and st_field["field_work_check"] == "pending")

    # --- 2. отказ от ОДНОГО формата: не отсев и не закрытие, вопрос про следующий ---
    plan = decide(_ready(["ON_SITE", "HYBRID"]), _obs(formats=[("ON_SITE", "no")]),
                  "в офис не готов", CTX)
    c.add("отказ от офиса при [ON_SITE,HYBRID] — не KO",
          plan.kind == "ask" and plan.focus == "format", f"{plan.rule}/{plan.reason_code}")
    c.add("следующим спрашиваем про гибрид", plan.state_next.get("format_asked") == "HYBRID",
          str(plan.state_next.get("format_asked")))
    c.add("гибрид назван в вопросе словами, без сырого id",
          "гибридный" in plan.instruction and "HYBRID" not in plan.instruction,
          plan.instruction[:90])

    # --- 3. отказ от ВСЕХ допустимых → KO_FORMAT ---
    plan = decide(_ready(["ON_SITE", "HYBRID"], formats={"ON_SITE": "no"}),
                  _obs(formats=[("HYBRID", "no")]), "гибрид тоже не подходит", CTX)
    c.add("отказ от всех допустимых → KO_FORMAT", plan.reason_code == "KO_FORMAT" and plan.end,
          f"{plan.rule}/{plan.reason_code}")

    # --- 4. согласие на один формат закрывает проверку ---
    plan = decide(_ready(["ON_SITE", "HYBRID"]), _obs(formats=[("HYBRID", "yes")]),
                  "гибрид подойдёт", CTX)
    c.add("согласие на гибрид закрывает format_check",
          plan.state_next["format_check"] == "closed", plan.state_next["format_check"])

    # --- 5. разъездной формат: отказ при другом подтверждённом — не KO ---
    plan = decide(_ready(["ON_SITE", "FIELD_WORK"], formats={"ON_SITE": "yes"}, format_check="closed"),
                  _obs(formats=[("FIELD_WORK", "no")]), "разъезды не готов", CTX)
    c.add("отказ от разъездов при подтверждённом офисе — не KO",
          plan.reason_code != "KO_FORMAT" and plan.state_next["field_work_check"] == "closed",
          f"{plan.reason_code}/{plan.state_next['field_work_check']}")

    # --- 6. разъездной единственный → отказ = KO_FORMAT ---
    plan = decide(_ready(["FIELD_WORK"]), _obs(formats=[("FIELD_WORK", "no")]),
                  "разъезды не готов", CTX)
    c.add("отказ от единственного FIELD_WORK → KO_FORMAT", plan.reason_code == "KO_FORMAT",
          plan.reason_code)

    # --- 7. локация и формат — РАЗНЫЕ пункты повестки (решение Р18) ---
    # Прежние проверки этой секции утверждали обратное («согласие на переезд закрывает format_check»,
    # «спрашиваем формат И переезд одним вопросом») и потому маскировали дефект кейса E живого
    # прогона 20260831_215941: отказ и от формата, и от переезда одной репликой уходил в KO_FORMAT.
    plan = decide(_ready(["ON_SITE"], city_check="pending"), _obs(), "а что по вакансии?", CTX)
    c.add("первым спрашиваем ГОРОД, отдельным вопросом",
          plan.focus == "city" and "в каком городе" in plan.instruction
          and "формат" not in plan.instruction.lower(), plan.instruction[:110])
    plan = decide(_ready(["ON_SITE"], candidate_city="Казань", city_check="closed"),
                  _obs(), "а что по вакансии?", CTX)
    c.add("вопрос про формат больше НЕ спрашивает город и переезд",
          "переехать" not in plan.instruction and "в каком городе" not in plan.instruction,
          plan.instruction[:130])
    plan = decide(_ready(["ON_SITE"], candidate_city="Москва", city_check="closed"),
                  _obs(formats=[("ON_SITE", "no")], relocation="yes"),
                  "в офис не готов, но перееду", CTX)
    c.add("«в офис не готов, но перееду» → всё равно KO_FORMAT",
          plan.reason_code == "KO_FORMAT" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(_ready(["ON_SITE"], candidate_city="Казань", city_check="closed"),
                  _obs(formats=[("ON_SITE", "yes")]), "формат подходит", CTX)
    c.add("формат подтверждён + другой город → открывается пункт про переезд",
          plan.state_next.get("relocation_check") == "pending" and plan.focus == "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    c.add("вопрос про переезд — про МЕСТО, а не про формат",
          "переехать в этот город" in plan.instruction, plan.instruction[-120:])
    st_reloc = _ready(["ON_SITE"], candidate_city="Казань", city_check="closed",
                      format_check="closed", relocation_check="pending")
    plan = decide(st_reloc, _obs(relocation="no"), "переезжать не буду", CTX)
    c.add("формат подтверждён + отказ от переезда → KO_LOCATION",
          plan.reason_code == "KO_LOCATION" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(st_reloc, _obs(relocation="yes"), "готов переехать", CTX)
    c.add("согласие переехать закрывает ПУНКТ ПРО ПЕРЕЕЗД, а не формат",
          plan.state_next["relocation_check"] == "closed" and not plan.end,
          f"{plan.state_next['relocation_check']}/{plan.reason_code}")
    plan = decide(_ready(["ON_SITE"], candidate_city="Москва", city_check="closed"),
                  _obs(formats=[("ON_SITE", "yes")], relocation="no"),
                  "формат ок, переезжать никуда не буду", CTX)
    c.add("тот же город: отказ от переезда отсевом не является",
          plan.reason_code != "KO_LOCATION" and plan.state_next["relocation_check"] == "n/a",
          f"{plan.reason_code}/{plan.state_next['relocation_check']}")
    # Разъездной формат считается присутственным: локация для него важна так же, как для офиса.
    plan = decide(_ready(["FIELD_WORK"], candidate_city="Казань", city_check="closed"),
                  _obs(formats=[("FIELD_WORK", "yes")]), "к разъездам готов", CTX)
    c.add("разъездной подтверждён + другой город → пункт про переезд открывается",
          plan.state_next.get("relocation_check") == "pending",
          str(plan.state_next.get("relocation_check")))
    # Регрессия прогона 20260901_181013, сценарий 56: `[REMOTE, FIELD_WORK]`, кандидат отказался от
    # разъездов (по правилу канала это НЕ отсев, `field_work_check` закрывается) и переезжать не
    # готов. Первая версия Р18 читала это закрытие как подтверждение присутствия и отсевала по
    # локации — на вакансии, где есть удалёнка и ехать никуда не надо.
    st_rf = _ready(["REMOTE", "FIELD_WORK"], candidate_city="Новосибирск", city_check="closed")
    plan = decide(st_rf, _obs(formats=[("FIELD_WORK", "no")]), "к разъездам не готов", CTX)
    c.add("есть REMOTE: отказ от разъездов не открывает пункт переезда",
          plan.state_next.get("relocation_check") == "n/a" and plan.focus != "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    plan = decide(plan.state_next, _obs(relocation="no"), "переезжать не буду", CTX)
    c.add("есть REMOTE: отказ переезжать не отсевает",
          plan.reason_code != "KO_LOCATION" and not plan.end, f"{plan.rule}/{plan.reason_code}")
    # Второй промах того же условия: «проверка закрыта» ≠ «формат подтверждён».
    st_of = _ready(["ON_SITE", "FIELD_WORK"], candidate_city="Казань", city_check="closed")
    plan = decide(st_of, _obs(formats=[("FIELD_WORK", "no")]), "разъезды не готов", CTX)
    c.add("закрытие разъездного ОТКАЗОМ присутствия не подтверждает",
          plan.state_next.get("relocation_check") == "n/a",
          str(plan.state_next.get("relocation_check")))

    # Удалённая вакансия: формат не спрашиваем, а город — обязательно, иначе гео-отсев мёртв.
    plan = decide(_ready(["REMOTE"], city_check="pending"), _obs(), "здравствуйте", CTX)
    c.add("удалёнка: город всё равно спрашиваем",
          plan.focus == "city" and "в каком городе" in plan.instruction, plan.instruction[:100])
    plan = decide(_ready(["REMOTE"], candidate_city="Тбилиси", city_check="closed"),
                  _obs(relocation="no"), "переезжать не буду", CTX)
    c.add("удалёнка: про переезд не спрашиваем и по нему не отсеваем",
          plan.state_next.get("relocation_check") == "n/a" and plan.reason_code != "KO_LOCATION",
          f"{plan.state_next.get('relocation_check')}/{plan.reason_code}")

    # --- 8. гео-ограничение: только при двойном совпадении (Б3) ---
    geo_ctx = DecideContext(band_max=280000, location="Россия, только РФ", has_geo_restriction=True)
    plan = decide(_ready(["REMOTE"]), _obs(geo=True, city="Берлин"), "я живу в Германии", geo_ctx)
    c.add("гео-ограничение + нарушение → KO_LOCATION_GEO",
          plan.reason_code == "KO_LOCATION_GEO", plan.reason_code)
    plan = decide(_ready(["REMOTE"]), _obs(geo=True, city="Берлин"), "я живу в Германии", CTX)
    c.add("без ограничения в вакансии заграница отсевом не является",
          plan.reason_code != "KO_LOCATION_GEO", plan.reason_code)

    # --- 9. смена формата в вопросе — не переспрос, повтор того же формата — переспрос ---
    base = _ready(["ON_SITE", "HYBRID"], last_asking="format", format_asked="ON_SITE")
    plan = decide(base, _obs(formats=[("ON_SITE", "no")]), "в офис не готов", CTX)
    c.add("переход офис→гибрид кап переспросов не жжёт", plan.state_next["format_reasks"] == 0,
          str(plan.state_next["format_reasks"]))
    plan = decide(base, _obs(), "ага", CTX)
    c.add("повтор того же формата — переспрос", plan.state_next["format_reasks"] == 1,
          str(plan.state_next["format_reasks"]))

    # --- 10. кап переспросов разъездного формата (ветки нет в TG) ---
    plan = decide(_ready(["FIELD_WORK"], format_check="n/a", last_asking="field_work",
                         format_asked="FIELD_WORK", field_work_reasks=2),
                  _obs(), "не понял вопроса", CTX)
    c.add("3-й переспрос про разъезды → STOP_PERSISTENT",
          plan.reason_code == "STOP_PERSISTENT" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- 11. проекция состояния: служебного модель не видит, форматы видит ---
    projection = hh_projection(_ready(["ON_SITE"], format_asked="ON_SITE"))
    c.add("проекция без счётчиков и служебных полей",
          not ({"counters", "format_reasks", "no_progress", "formats"} & set(projection)),
          str(sorted(projection)))
    c.add("проекция отдаёт allowed_formats и format_asked",
          projection.get("allowed_formats") == ["ON_SITE"]
          and projection.get("format_asked") == "ON_SITE")

    return c.rows

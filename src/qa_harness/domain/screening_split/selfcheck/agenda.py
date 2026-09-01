"""Повестка хода: зарплата → город → формат → переезд → доп-вопросы (решение Р18).

Проверки переписаны после живого прогона `20260831_215941`: до Р18 город спрашивался ВНУТРИ вопроса
про формат, а согласие переехать закрывало `format_check`. Отсюда три дефекта, и каждый имеет здесь
свою строку: кандидат «в офис не готов, но перееду» проходил проверку формата; ключ отсева между
форматом и локацией достался тому факту, что пришёл первым; на удалённых вакансиях город не
спрашивался вовсе, из-за чего гео-ограничение вакансии не отсеивало никого.

Живой сценарий этого не поймал бы: отсев ассертится по `expect_script_prefix: KO_`, а
`KO_FORMAT_OFFICE` от `KO_LOCATION` этим префиксом не отличается.
"""

from typing import List

from .. import state as state_model
from ..policy import DecideContext, decide
from ..policy.geo import same_city
from ..policy.observation import Observation
from .collect import Checks, Row

MSK = DecideContext(band_max=280000, work_format="office", location="Москва")
NOCITY = DecideContext(band_max=280000, work_format="office")
HYBRID = DecideContext(band_max=280000, work_format="hybrid", location="Москва")
REMOTE = DecideContext(band_max=280000, work_format="remote", location="Москва")


def _obs(**facts) -> Observation:
    o = Observation()
    o.facts = {"candidate_city": None, "format_ready": None, "relocation_ready": None,
               "geo_blocked": False, **facts}
    return o


def _ready(work_format: str = "office", **over) -> dict:
    st = state_model.init_state(work_format, "- Опыт с Python?")
    st["salary"] = "closed"          # проверяем повестку локации, а не деньги
    st.update(over)
    return st


def checks() -> List[Row]:
    c = Checks()

    # --- 1. порядок повестки: зарплата → город → формат → переезд ---
    plan = decide(_ready(), _obs(), "ок", MSK)
    c.add("первым спрашиваем ГОРОД, отдельным вопросом",
          plan.focus == "city" and "в каком городе" in plan.instruction
          and "формат" not in plan.instruction.lower(), plan.instruction[:90])
    plan = decide(_ready(), _obs(candidate_city="Москва"), "я в Москве", MSK)
    c.add("город назван → пункт закрыт, следующий вопрос про формат",
          plan.state_next.get("city_check") == "closed" and plan.focus == "format",
          f"{plan.state_next.get('city_check')}/{plan.focus}")
    c.add("вопрос про формат больше НЕ спрашивает город и переезд",
          "в каком городе" not in plan.instruction and "переехать" not in plan.instruction,
          plan.instruction[:110])

    # --- 2. эскалация вопроса про город: объяснение → предупреждение → кап ---
    # Счётчик в state — это значение ДО хода: 0 → на этом ходе будет 1-й переспрос, и так далее.
    plan = decide(_ready(last_asking="city", city_reasks=0), _obs(), "а зачем вам мой город?", MSK)
    c.add("1-й переспрос города объясняет ЗАЧЕМ и ещё не угрожает",
          "часовой пояс" in plan.instruction and "не получится" not in plan.instruction,
          plan.instruction[:120])
    plan = decide(_ready(last_asking="city", city_reasks=1), _obs(), "не скажу", MSK)
    c.add("2-й переспрос города предупреждает о последствии",
          "не получится" in plan.instruction, plan.instruction[:120])
    plan = decide(_ready(last_asking="city", city_reasks=2), _obs(), "не скажу", MSK)
    c.add("3-й переспрос города → завершение по капу",
          plan.reason_code == "STOP_PERSISTENT" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- 3. ГЛАВНОЕ Р18: формат — самостоятельное требование ---
    plan = decide(_ready(candidate_city="Москва", city_check="closed"),
                  _obs(format_ready="no", relocation_ready="yes"),
                  "в офис не готов, но перееду", MSK)
    c.add("«в офис не готов, но перееду» → всё равно KO по формату",
          plan.reason_code == "KO_FORMAT_OFFICE" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(_ready(city_check="closed", candidate_city="Казань"), _obs(format_ready="no"),
                  "в офис не готов", MSK)
    c.add("отказ от формата у иногороднего → KO по формату, а не переспрос",
          plan.reason_code == "KO_FORMAT_OFFICE", f"{plan.rule}/{plan.reason_code}")
    plan = decide(_ready(), _obs(format_ready="no"), "в офис не готов", MSK)
    c.add("отказ от формата при НЕизвестном городе → KO сразу",
          plan.reason_code == "KO_FORMAT_OFFICE", f"{plan.rule}/{plan.reason_code}")
    plan = decide(_ready("hybrid", city_check="closed", candidate_city="Москва"),
                  _obs(format_ready="no"), "гибрид не подходит", HYBRID)
    c.add("гибридная вакансия → KO_FORMAT_HYBRID", plan.reason_code == "KO_FORMAT_HYBRID",
          plan.reason_code)
    plan = decide(_ready(city_check="closed", candidate_city="Казань"), _obs(format_ready="no"),
                  "не готов", NOCITY)
    c.add("локации у вакансии нет → KO_FORMAT_NOCITY",
          plan.reason_code == "KO_FORMAT_NOCITY", plan.reason_code)

    # --- 4. пункт про переезд: когда открывается и когда нет ---
    plan = decide(_ready(city_check="closed", candidate_city="Казань"), _obs(format_ready="yes"),
                  "формат подходит", MSK)
    c.add("формат подтверждён + другой город → открывается пункт про переезд",
          plan.state_next.get("relocation_check") == "pending" and plan.focus == "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    c.add("вопрос про переезд — про МЕСТО, а не про формат",
          "переехать в этот город" in plan.instruction, plan.instruction[-110:])
    plan = decide(_ready(city_check="closed", candidate_city="Москва"), _obs(format_ready="yes"),
                  "формат подходит", MSK)
    c.add("кандидат В городе вакансии → про переезд не спрашиваем",
          plan.state_next.get("relocation_check") == "n/a" and plan.focus != "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    plan = decide(_ready(city_check="closed", candidate_city="Казань"), _obs(format_ready="yes"),
                  "формат подходит", NOCITY)
    c.add("у вакансии нет локации → про переезд не спрашиваем",
          plan.state_next.get("relocation_check") == "n/a",
          str(plan.state_next.get("relocation_check")))
    plan = decide(_ready(candidate_city="Казань"), _obs(relocation_ready="no"),
                  "переезжать не буду", MSK)
    c.add("отказ от переезда ДО подтверждения формата отсевом не является",
          not plan.end and plan.reason_code != "KO_LOCATION", f"{plan.rule}/{plan.reason_code}")

    # --- 5. отсев по локации и его отсутствие ---
    st = _ready(city_check="closed", candidate_city="Казань", format_check="closed",
                relocation_check="pending")
    plan = decide(st, _obs(relocation_ready="no"), "переезжать не буду", MSK)
    c.add("формат подтверждён + отказ от переезда → KO_LOCATION",
          plan.reason_code == "KO_LOCATION" and plan.end, f"{plan.rule}/{plan.reason_code}")
    plan = decide(st, _obs(relocation_ready="yes"), "перееду", MSK)
    c.add("согласие переехать → пункт закрыт, отсева нет",
          plan.state_next.get("relocation_check") == "closed" and not plan.end,
          f"{plan.state_next.get('relocation_check')}/{plan.reason_code}")
    plan = decide(st, _obs(), "подумаю", MSK)
    plan = decide(plan.state_next, _obs(relocation_ready="yes"), "хорошо, перееду", MSK)
    c.add("кандидат передумал → пункт закрывается, диалог живёт",
          plan.state_next.get("relocation_check") == "closed" and not plan.end,
          f"{plan.state_next.get('relocation_check')}/{plan.reason_code}")

    # --- 6. удалённая вакансия: формат не спрашиваем, город спрашиваем (иначе гео-отсев мёртв) ---
    st = state_model.init_state("remote", "- Опыт с Python?")
    st["salary"] = "closed"
    c.add("удалёнка: формат n/a", st.get("format_check") == "n/a", str(st.get("format_check")))
    plan = decide(st, _obs(), "здравствуйте", REMOTE)
    c.add("удалёнка: город всё равно спрашиваем",
          plan.focus == "city" and "в каком городе" in plan.instruction, plan.instruction[:90])
    plan = decide(st, _obs(candidate_city="Тбилиси"), "я в Тбилиси", REMOTE)
    c.add("удалёнка: город назван → про переезд не спрашиваем никогда",
          plan.state_next.get("relocation_check") == "n/a" and plan.focus != "relocation",
          f"{plan.state_next.get('relocation_check')}/{plan.focus}")
    plan = decide(st, _obs(candidate_city="Тбилиси", geo_blocked=True), "я в Тбилиси",
                  DecideContext(band_max=280000, work_format="remote", location="Москва",
                                has_geo_restriction=True))
    c.add("удалёнка + гео-ограничение + кандидат вне зоны → KO_GEO",
          plan.reason_code == "KO_GEO" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- 7. сравнение города: локацию пишут шире города ---
    c.add("same_city: Москва внутри «Россия, Москва»", same_city("Москва", "Россия, Москва"))
    c.add("same_city: регистр и ё", same_city("КОРОЛЁВ", "Королев"))
    c.add("same_city: разные города не совпадают", not same_city("Казань", "Москва"))
    # Радиусов и агломераций в коде нет и не будет: ближний пригород для ядра — обычный «другой
    # город». Отсюда и снятие сценариев про пригород — им нечего проверять сверх «другого города».
    c.add("same_city: пригород — это ДРУГОЙ город, радиуса нет",
          not same_city("Химки", "Москва"))

    return c.rows

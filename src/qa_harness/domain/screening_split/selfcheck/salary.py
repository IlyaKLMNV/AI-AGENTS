"""Зарплатный контракт: пересчёт, вилка, гейты годности claim.

Сумму распознаёт модель (`salary_claim`), всё остальное считает код — значит проверяемо без сети.
Ожидания берутся ОТ КОНСТАНТ `salary_rules`, а не переписываются числом: иначе правка курса или шкалы
НДФЛ ломает ровно ту проверку, которая должна была её подтвердить.
"""

from typing import Any, Dict, List

from .. import salary as salary_mod
from .. import salary_rules as rules
from .. import state as state_model
from ..policy import DecideContext, decide, gates
from ..policy import core as policy_core
from ..policy.observation import Observation
from .collect import Checks, Row

BAND_MIN, BAND_MAX = 200_000, 280_000


def claim(**kw: Any) -> Dict[str, Any]:
    """Годный по форме `salary_claim`; аргументом меняется ровно проверяемое поле."""
    base = {"subject": "own_expectation", "form": "exact", "amount_min": None, "amount_max": None,
            "scale": "thousand", "currency": "RUB", "period": "month", "tax": "unspecified",
            "quote": ""}
    base.update(kw)
    return base


def _plan(claim_dict: Dict[str, Any], message: str, **ctx_over: Any):
    """Ход чистого ядра с одной названной суммой и закрытым городом."""
    state = state_model.init_state("remote", "- Опыт с Python?")
    state["city_check"] = "closed"
    obs = Observation()
    obs.facts = {"candidate_city": "Москва"}
    obs.salary_claim = claim_dict
    ctx = DecideContext(band_min=BAND_MIN, band_max=BAND_MAX, location="Москва", **ctx_over)
    return decide(state, obs, message, ctx)


def checks() -> List[Row]:
    c = Checks()

    # --- пересчёт: Д8-Д11 ---
    v = salary_mod.normalize(claim(amount_min=70, scale="none_stated"))
    c.add("Д8 голое 70 = 70000", v.min == 70_000, f"факт {v.min}")
    v = salary_mod.normalize(claim(amount_min=200, scale="unit"))
    c.add("Д8 200 рублей читаем буквально", v.min == 200, f"факт {v.min}")
    v = salary_mod.normalize(claim(amount_min=1200, scale="unit", period="hour"))
    c.add("Д9 ставка за час умножается на норму часов",
          v.min == round(1200 * rules.PERIOD_TO_MONTH["hour"]), f"факт {v.min}")
    v = salary_mod.normalize(claim(amount_min=4000, scale="unit", currency="USD"))
    c.add("Д10 валюта по курсу из правил",
          v.min == round(4000 * rules.RATE_TO_RUB["USD"]), f"факт {v.min}")
    v = salary_mod.normalize(claim(amount_min=250, scale="thousand", tax="gross"))
    c.add("Д11 gross в net по прогрессивной шкале",
          v.min < round(250_000 * (1 - rules.NDFL_BRACKETS[0][1])), f"факт {v.min}")

    # --- вердикт по вилке: Д12/Д13 ---
    def verdict(**kw: Any) -> str:
        return salary_mod.compare_with_band(salary_mod.normalize(claim(**kw)), BAND_MIN, BAND_MAX)

    c.add("Д13 диапазон 250-400 не отказ (готов на 250)",
          verdict(form="range", amount_min=250, amount_max=400, scale="thousand") == "fits")
    c.add("Д13 порог от 300 — отказ",
          verdict(form="at_least", amount_min=300, scale="thousand") == "ko")
    c.add("до 400 — не отказ никогда",
          verdict(form="at_most", amount_max=400, scale="thousand") == "fits")
    c.add("ниже минимума вилки — не отказ", verdict(amount_min=100, scale="thousand") == "fits")
    c.add("Д12 отказ по ПЕРЕСЧИТАННОЙ сумме",
          verdict(amount_min=4500, scale="unit", currency="USD") == "ko")

    # --- гейты годности claim ---
    c.add("чужая сумма непригодна",
          salary_mod.claim_status(claim(subject="third_party", amount_min=500, quote="500 тысяч"),
                                  "У коллеги 500 тысяч") == salary_mod.UNUSABLE)
    c.add("текущая ЗП без ожиданий непригодна",
          salary_mod.claim_status(claim(subject="own_current", amount_min=250, quote="получаю 250"),
                                  "Сейчас получаю 250") == salary_mod.UNUSABLE)
    c.add("валюта вне справочника непригодна",
          salary_mod.claim_status(claim(currency="other", amount_min=4000, quote="4000 фунтов"),
                                  "Рассматриваю 4000 фунтов") == salary_mod.UNUSABLE)
    c.add("выдуманная цитата непригодна (гейт против галлюцинации)",
          salary_mod.claim_status(claim(amount_min=400, quote="400 тысяч"),
                                  "Готов обсуждать варианты") == salary_mod.UNUSABLE)
    c.add("годный claim пригоден",
          salary_mod.claim_status(claim(amount_min=300, quote="300 тысяч"),
                                  "Ориентируюсь на 300 тысяч") == salary_mod.ACTIONABLE)

    # --- односторонние гейты (Р1): понижают только ko, fits неприкосновенен ---
    c.add("гейт масштаба: число без слова-масштаба в цитате",
          gates.scale_marker_missing(claim(amount_min=300, scale="thousand", quote="300")))
    c.add("гейт масштаба: 300 тысяч проходит",
          not gates.scale_marker_missing(claim(amount_min=300, scale="thousand", quote="300 тысяч")))
    c.add("гейт понижает ko до уточнения",
          bool(gates.downgrade_ko(claim(amount_min=400, scale="thousand", quote="400"))))

    # --- ВЫРОЖДЕННАЯ ФОРМА claim: единственная причина, по которой normalize даёт None ---
    # Потолка правдоподобия у пересчёта НЕТ: абсурдная сумма — законный отказ по вилке, а не ошибка.
    degenerate = claim(form="at_least", amount_min=None, amount_max=300, quote="300 тысяч")
    c.add("вырожденная форма: статус пропускает, пересчёт даёт None",
          salary_mod.claim_status(degenerate, "Ориентируюсь на 300 тысяч") == salary_mod.ACTIONABLE
          and salary_mod.normalize(degenerate) is None)
    c.add("абсурдная сумма — обычный отказ по вилке, а не уточнение",
          salary_mod.compare_with_band(salary_mod.normalize(claim(amount_min=20, scale="million")),
                                       BAND_MIN, BAND_MAX) == "ko")
    plan = _plan(degenerate, "Ориентируюсь на 300 тысяч")
    c.add("вырожденная форма в ядре: не отсеивает и НЕ закрывает пункт",
          plan.state_next["salary"] == "pending" and not plan.end,
          "salary={} end={}".format(plan.state_next["salary"], plan.end))
    c.add("вырожденная форма помечена unusable в аудите",
          (plan.audit.get("salary") or {}).get("status") == salary_mod.UNUSABLE,
          str((plan.audit.get("salary") or {}).get("status")))

    # --- закрыть зарплату, не сравнив её, стало НЕВЫРАЗИМО ---
    # Замена «гейта updates» старого движка: ключа `salary` в `_updates_from_observation` нет, пункт
    # закрывает только код и только после сравнения с вилкой. Проверяется свойство контракта.
    obs_closed = Observation()
    obs_closed.facts = {"candidate_city": "Москва"}
    updates = policy_core._updates_from_observation(obs_closed, state_model.init_state("remote", ""))
    c.add("наблюдение не может закрыть зарплату",
          not any(u["key"] == "salary" for u in updates),
          str([u["key"] for u in updates]))

    # --- вердикты через ядро ---
    plan = _plan(claim(amount_min=250, quote="250 тысяч"), "Ориентируюсь на 250 тысяч")
    c.add("fits: пункт закрыт, диалог живёт",
          plan.state_next["salary"] == "closed" and not plan.end,
          "salary={} end={}".format(plan.state_next["salary"], plan.end))
    plan = _plan(claim(amount_min=400, quote="400 тысяч"), "Ориентируюсь на 400 тысяч")
    c.add("ko: KO_SALARY и завершение",
          plan.reason_code == "KO_SALARY" and plan.end, f"{plan.rule}/{plan.reason_code}")
    c.add("эффект вердикта виден в аудите",
          (plan.audit.get("salary") or {}).get("effect") == "ko_forced",
          str((plan.audit.get("salary") or {}).get("effect")))

    # --- повторное сравнение ПОСЛЕ закрытия пункта (кандидат поднял ожидания в середине) ---
    state = state_model.init_state("remote", "- Опыт с Python?")
    state["salary"] = "closed"
    state["city_check"] = "closed"
    obs = Observation()
    obs.salary_claim = claim(amount_min=400, quote="400 тысяч")
    plan = decide(state, obs, "Передумал, хочу 400 тысяч",
                  DecideContext(band_min=BAND_MIN, band_max=BAND_MAX, location="Москва"))
    c.add("отсев работает и после закрытия пункта",
          plan.reason_code == "KO_SALARY" and plan.end, f"{plan.rule}/{plan.reason_code}")

    # --- вилка в чужой валюте приводится к рублям (P11) ---
    state = state_model.init_state("remote", "- Опыт с Python?")
    state["city_check"] = "closed"
    obs = Observation()
    obs.salary_claim = claim(amount_min=250, quote="250 тысяч на руки", tax="net")
    plan = decide(state, obs, "250 тысяч на руки",
                  DecideContext(band_min=1_000_000, band_max=1_400_000, band_currency="KZT",
                                location="Москва"))
    c.add("вилка KZT пересчитана: 250к не отсев", plan.reason_code != "KO_SALARY",
          "{}/{}".format(plan.reason_code, (plan.audit.get("salary") or {}).get("verdict")))

    return c.rows

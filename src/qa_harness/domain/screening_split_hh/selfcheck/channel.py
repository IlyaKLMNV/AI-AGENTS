"""Канальная дельта hh: реестр причин, набор сигналов, паритет бюджетов с TG, валюта вилки.

Три копии одного оркестратора расходятся молча — это главный источник багов (задача Б2). Паритет
здесь проверяется как ДАННЫЕ: общие ключи обязаны совпадать порогами и причинами, а разойтись
разрешено ровно в двух местах, и оба названы.
"""

from typing import List

from qa_harness.domain.screening_split.policy import budgets as tg_budgets
from qa_harness.domain.screening_split.selfcheck.collect import Checks, Row

from .. import state as hh_state
from ..policy import DecideContext, Observation, decide, parse_observation
from ..policy import reasons as hh_reasons
from ..policy.budgets import EVENT_BUDGETS, REASK_BUDGETS, STALL_BUDGET
from ..policy.observation import formats_ready


def checks() -> List[Row]:
    c = Checks()

    # --- реестр причин: канальные ключи есть, TG-шных нет ---
    try:
        hh_reasons.validate()
        ok, detail = True, ""
    except ValueError as exc:
        ok, detail = False, str(exc)
    c.add("три инварианта hh-реестра выполняются", ok, detail)
    c.add("реестр: KO_LOCATION/KO_LOCATION_GEO/KO_FORMAT есть",
          all(hh_reasons.is_known(k) for k in ("KO_LOCATION", "KO_LOCATION_GEO", "KO_FORMAT")))
    c.add("реестр: TG-ключей (KO_GEO, KO_FORMAT_OFFICE, REPLY_CONTACT_SOURCE) нет",
          not any(hh_reasons.is_known(k)
                  for k in ("KO_GEO", "KO_FORMAT_OFFICE", "REPLY_CONTACT_SOURCE")))
    c.add("реестр: STOP_POLITICS не рендерится в пустоту",
          bool((hh_reasons.render("STOP_POLITICS") or "").strip()))
    c.add("текст KO_LOCATION подставляет локацию",
          "локация Москва" in (hh_reasons.render("KO_LOCATION", city="Москва") or ""),
          hh_reasons.render("KO_LOCATION", city="Москва"))
    c.add("пустая локация текст не ломает",
          "{" not in (hh_reasons.render("KO_LOCATION", city="") or ""),
          hh_reasons.render("KO_LOCATION", city=""))

    # --- сигнала contact_source в канале нет: парсер его отбрасывает ---
    parsed, _ = parse_observation(
        {"signals": [{"code": "contact_source", "quote": "откуда мои данные"}],
         "focus_answered": "none"}, "откуда мои данные")
    c.add("сигнал contact_source отброшен", parsed.codes() == [], str(parsed.codes()))
    c.add("отброс канального сигнала виден в dropped",
          any("hh-канале не существует" in d for d in parsed.dropped), str(parsed.dropped))

    # --- formats_ready: разбор фактов по форматам ---
    obs = Observation()
    obs.facts = {"formats_ready": [{"format": "on_site", "ready": "YES"},
                                   {"format": "МУСОР", "ready": "yes"},
                                   {"format": "HYBRID", "ready": "может быть"},
                                   {"format": "ON_SITE", "ready": "no"}]}
    got = formats_ready(obs)
    c.add("formats_ready: регистр нормализуется, мусор отброшен, дубль берётся последний",
          got == {"ON_SITE": "no"}, str(got))

    # --- паритет бюджетов с TG ---
    common_events = set(EVENT_BUDGETS) & set(tg_budgets.EVENT_BUDGETS)
    diff = {k for k in common_events
            if (EVENT_BUDGETS[k].fires_on_nth, EVENT_BUDGETS[k].reason)
            != (tg_budgets.EVENT_BUDGETS[k].fires_on_nth, tg_budgets.EVENT_BUDGETS[k].reason)}
    c.add("событийные пороги совпадают с TG по общим ключам", not diff, str(sorted(diff)))
    c.add("дельта событийных ровно {contact_source} и только в TG",
          set(tg_budgets.EVENT_BUDGETS) - set(EVENT_BUDGETS) == {"contact_source"}
          and not set(EVENT_BUDGETS) - set(tg_budgets.EVENT_BUDGETS),
          str(sorted(set(tg_budgets.EVENT_BUDGETS) ^ set(EVENT_BUDGETS))))

    common_reasks = set(REASK_BUDGETS) & set(tg_budgets.REASK_BUDGETS)
    diff = {k for k in common_reasks
            if (REASK_BUDGETS[k].fires_on_nth, REASK_BUDGETS[k].reason,
                REASK_BUDGETS[k].persist_on_fire)
            != (tg_budgets.REASK_BUDGETS[k].fires_on_nth, tg_budgets.REASK_BUDGETS[k].reason,
                tg_budgets.REASK_BUDGETS[k].persist_on_fire)}
    c.add("пороги переспросов совпадают с TG по общим ключам", not diff, str(sorted(diff)))
    c.add("дельта переспросов ровно {field_work} и только в hh",
          set(REASK_BUDGETS) - set(tg_budgets.REASK_BUDGETS) == {"field_work"}
          and not set(tg_budgets.REASK_BUDGETS) - set(REASK_BUDGETS),
          str(sorted(set(REASK_BUDGETS) ^ set(tg_budgets.REASK_BUDGETS))))
    c.add("стоп-кран совпадает с TG",
          (STALL_BUDGET.counter, STALL_BUDGET.fires_on_nth)
          == (tg_budgets.STALL_BUDGET.counter, tg_budgets.STALL_BUDGET.fires_on_nth),
          f"{STALL_BUDGET.counter}/{STALL_BUDGET.fires_on_nth}")

    # --- вилка в тенге приводится к рублям (P11: сегодня currency не читается) ---
    kzt = DecideContext(band_min=1_000_000, band_max=1_400_000, band_currency="KZT",
                        location="Москва")
    st = hh_state.init_state(["REMOTE"], "- Опыт с Python?")
    st["city_check"] = "closed"
    obs = Observation()
    obs.facts = {"candidate_city": "Москва", "formats_ready": [], "relocation_ready": None,
                 "geo_blocked": False}
    obs.salary_claim = {"subject": "own_expectation", "form": "exact", "amount_min": 250,
                        "amount_max": 250, "scale": "thousand", "currency": "RUB",
                        "period": "month", "tax": "net", "quote": "250 тысяч на руки"}
    plan = decide(st, obs, "250 тысяч на руки", kzt)
    c.add("вилка KZT пересчитана — 250к не отсев", plan.reason_code != "KO_SALARY",
          f"{plan.reason_code}/{(plan.audit.get('salary') or {}).get('verdict')}")

    return c.rows

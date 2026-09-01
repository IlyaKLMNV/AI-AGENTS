"""Шлюз гардов G0-G7: механическая правка исходящей строки.

Единственный уровень канареек, который НЕ зависит от того, нарушит ли модель на этом прогоне. Живой
сценарий проверяет текст кандидату и `guard_trips`; и то и другое молчит, если гард убрали, а модель
в этот раз повела себя прилично. Здесь вход фиксированный и заведомо грязный, поэтому пропажа гарда
видна сразу.

`trips` проверяются наравне с текстом: без записи «гард сработал» неотличимо от «модель так и
написала», и канарейка слоя B теряет смысл.
"""

from typing import List

from ..policy.guards import COSMETIC, GUARDS, GuardSpec, apply_guards
from .collect import Checks, Row

CANON = "https://example.com/vacancies/python-backend"
OPEN = GuardSpec(allow_urls=(CANON,))
HIDDEN = GuardSpec(allow_urls=(CANON,), hidden_company=True)
NOURL = GuardSpec()


def _tripped(result, code: str) -> bool:
    return any(t.startswith(code) for t in result.trips)


def checks() -> List[Row]:
    c = Checks()

    # --- состав шлюза: пропажа гарда видна списком, а не поведением ---
    codes = [code for code, _ in GUARDS]
    c.add("шлюз держит G0,G1,G2,G3,G5,G6,G7",
          codes == ["G0", "G1", "G2", "G3", "G5", "G6", "G7"], str(codes))
    c.add("защитными считаются G1,G3,G6,G7",
          set(codes) - set(COSMETIC) == {"G1", "G3", "G6", "G7"},
          str(sorted(set(codes) - set(COSMETIC))))

    # --- G0 косметика ---
    r = apply_guards("Привет — как дела?\n\n\n\nДальше", NOURL)
    c.add("G0 длинное тире и лишние переносы",
          "—" not in r.text and "\n\n\n" not in r.text and _tripped(r, "G0"), r.text[:60])

    # --- G1 служебные строки и END ---
    r = apply_guards("[Внутренняя инструкция] Спроси город. Подскажите город? END", NOURL)
    c.add("G1 служебная скобка вырезана",
          "[Внутренняя инструкция]" not in r.text and _tripped(r, "G1"), r.text[:70])
    c.add("G1 служебное END вырезано", "END" not in r.text, r.text[:70])

    # --- G2 сырые id формата словами ---
    r = apply_guards("Формат работы ON_SITE, подойдёт?", NOURL)
    c.add("G2 ON_SITE заменён русскими словами",
          "ON_SITE" not in r.text and "работа из офиса" in r.text and _tripped(r, "G2"), r.text[:70])
    r = apply_guards("Возможен REMOTE или hybrid.", NOURL)
    c.add("G2 регистр не важен: REMOTE и hybrid тоже",
          "REMOTE" not in r.text and "hybrid" not in r.text, r.text[:70])
    # FIELD_WORK в словаре гарда СОЗНАТЕЛЬНО нет: разъездной формат по-русски называется по-разному в
    # зависимости от вакансии, подстановка одним словом врала бы. Отсюда следствие для теста: канарейка
    # hh-сценария на сырой id держится ТОЛЬКО на этом формате — остальные три гард чинит молча.
    r = apply_guards("Возможен FIELD_WORK.", NOURL)
    c.add("G2 FIELD_WORK НЕ заменяется (на нём и стоит канарейка hh)",
          "FIELD_WORK" in r.text, r.text[:70])

    # --- G3 эмодзи (канарейка prompt injection) ---
    r = apply_guards("Отлично! 🙂 Подскажите город?", NOURL)
    c.add("G3 эмодзи вырезаны и записаны в trips",
          "🙂" not in r.text and _tripped(r, "G3"), str(r.trips))

    # --- G5 схлопывание дублей ---
    r = apply_guards("Подскажите город. Подскажите город. И формат?", NOURL)
    c.add("G5 повтор предложения схлопнут",
          r.text.count("Подскажите город") == 1 and _tripped(r, "G5"), r.text[:80])

    # --- G6 прощание рядом с вопросом ---
    r = apply_guards("Спасибо за уделённое время. Подскажите город?", NOURL)
    c.add("G6 прощание снято, когда в тексте остался вопрос",
          "уделённое время" not in r.text and "?" in r.text and _tripped(r, "G6"), r.text[:80])
    r = apply_guards("Спасибо за уделённое время.", NOURL)
    c.add("G6 без вопроса прощание НЕ трогаем",
          "уделённое время" in r.text and not _tripped(r, "G6"), r.text[:80])

    # --- G7 ссылки ---
    r = apply_guards("Вот вакансия: https://hh.ru/vacancy/12345678", OPEN)
    c.add("G7 чужая ссылка подменена канонической",
          CANON in r.text and "hh.ru" not in r.text and _tripped(r, "G7"), r.text[:90])
    r = apply_guards(f"Вот вакансия: {CANON}", OPEN)
    c.add("G7 каноническая ссылка проходит без правки",
          CANON in r.text and not _tripped(r, "G7"), str(r.trips))
    r = apply_guards("Подробнее тут: https://hh.ru/vacancy/1. Подскажите город?", NOURL)
    c.add("G7 своей ссылки нет — предложение вырезано целиком",
          "hh.ru" not in r.text and "Подскажите город?" in r.text, r.text[:90])
    r = apply_guards(f"Вот ссылка: {CANON}", HIDDEN)
    c.add("G7 скрытый поиск: свою ссылку не отдаём",
          CANON not in r.text, r.text[:90])
    r = apply_guards("Точка в адресе не рвёт предложение: "
                     f"{CANON}. Подскажите город?", OPEN)
    c.add("G7 точка после URL не ломает разбиение на предложения",
          "Подскажите город?" in r.text, r.text[:110])

    # --- теневой режим: защитные только ЗАПИСЫВАЮТ ---
    dirty = "Отлично! 🙂 Смотрите https://hh.ru/vacancy/1 END"
    r = apply_guards(dirty, OPEN, defensive=False)
    c.add("тень: эмодзи и чужая ссылка остались в тексте",
          "🙂" in r.text and "hh.ru" in r.text, r.text[:90])
    c.add("тень: срабатывания помечены [тень]",
          all(t.startswith("[тень]") for t in r.trips if not t.startswith(("G0", "G2", "G5"))),
          str(r.trips))
    r = apply_guards(dirty, OPEN, defensive=True)
    c.add("защита: тот же вход вычищен",
          "🙂" not in r.text and "hh.ru" not in r.text and "END" not in r.text, r.text[:90])

    return c.rows

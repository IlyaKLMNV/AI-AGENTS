"""Реестр причин хода: тексты, терминальность, авторство, подстановки.

Три инварианта реестра выполняются ПРИ ИМПОРТЕ и роняют процесс на старте. Здесь они делаются
ВИДИМЫМИ в отчёте (иначе «проверка есть» существует только в виде отсутствия падения) и добавлено то,
что импорт не ловит: терминальность выводится из реестра, а не из префикса ключа, подстановки не
оставляют дыр в предложении, снятые ключи не вернулись.
"""

from typing import List

from ..policy import reasons as R
from .collect import Checks, Row


def checks() -> List[Row]:
    c = Checks()

    try:
        R.validate()
        ok, detail = True, ""
    except ValueError as exc:  # noqa: PERF203 — единственная точка, где падение надо показать строкой
        ok, detail = False, str(exc)
    c.add("три инварианта реестра выполняются", ok, detail)

    empty = sorted(code for code, r in R.REASONS.items() if not r.text.strip())
    c.add("ни один код не рендерится в пустоту", not empty, str(empty))

    # --- терминальность ИЗ РЕЕСТРА, а не по префиксу имени ---
    c.add("REPLY_CONTACT_SOURCE не терминален", not R.is_terminal("REPLY_CONTACT_SOURCE"))
    c.add("REPLY_FALLBACK не терминален", not R.is_terminal("REPLY_FALLBACK"))
    c.add("FINISH терминален без префикса STOP/KO", R.is_terminal("FINISH"))
    c.add("неизвестный код не терминален и не рендерится",
          not R.is_terminal("STOP_ВЫДУМАННЫЙ") and R.render("STOP_ВЫДУМАННЫЙ") is None)

    # --- снятые ключи не вернулись ---
    c.add("STOP_ABROAD снят вместе с безусловным отсевом за заграницу",
          not R.is_known("STOP_ABROAD"))
    c.add("REPLY_CONTACT_SOURCE_EMPTY перестал быть ключом",
          not R.is_known("REPLY_CONTACT_SOURCE_EMPTY"))

    # --- ключи Р18: формат и локация разведены ---
    c.add("ключи отсева по формату и по локации существуют раздельно",
          all(R.is_known(k) for k in ("KO_FORMAT_OFFICE", "KO_FORMAT_HYBRID",
                                      "KO_FORMAT_NOCITY", "KO_LOCATION", "KO_GEO")))
    c.add("текст KO_FORMAT_OFFICE говорит про формат, а не про место",
          "офис" in R.render("KO_FORMAT_OFFICE", city="Москва").lower()
          and "город" not in R.render("KO_FORMAT_OFFICE", city="Москва").lower(),
          R.render("KO_FORMAT_OFFICE", city="Москва"))
    c.add("текст KO_LOCATION говорит про место и подставляет город",
          "в городе Москва" in R.render("KO_LOCATION", city="Москва"),
          R.render("KO_LOCATION", city="Москва"))

    # --- подстановки не оставляют дыр в предложении ---
    c.add("пустой источник контакта подставляется дефолтом",
          "из базы кандидатов" in R.render("REPLY_CONTACT_SOURCE", contact_source=""),
          R.render("REPLY_CONTACT_SOURCE", contact_source=""))
    c.add("заполненный источник контакта попадает в текст",
          "резюме HH" in R.render("REPLY_CONTACT_SOURCE", contact_source="резюме HH"),
          R.render("REPLY_CONTACT_SOURCE", contact_source="резюме HH"))
    c.add("пустой город текст KO_LOCATION не ломает",
          "{" not in R.render("KO_LOCATION", city=""), R.render("KO_LOCATION", city=""))

    # --- авторство: код и модель не претендуют на одни ключи ---
    by_model = R.authored_by_model()
    by_code = {code for code, r in R.REASONS.items() if r.author == R.CODE}
    c.add("множества авторов не пересекаются", not (by_model & by_code), str(sorted(by_model & by_code)))
    c.add("код-форсимые ключи модели не принадлежат",
          not ({"KO_SALARY", "STOP_PERSISTENT", "STOP_BOT_REPEAT", "STOP_GIBBERISH_REPEAT",
                "STOP_SALARY_DEMAND", "STOP_PAUSE", "REPLY_FALLBACK"} & by_model),
          str(sorted({"KO_SALARY", "STOP_PERSISTENT", "STOP_BOT_REPEAT"} & by_model)))

    # --- исходы без текста рендерить не пытаемся ---
    c.add("NON_SCRIPT-исходы ключами реестра не являются",
          not (R.NON_SCRIPT & set(R.REASONS)), str(sorted(R.NON_SCRIPT & set(R.REASONS))))

    return c.rows

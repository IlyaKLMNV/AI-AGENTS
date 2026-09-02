"""Реестр причин хода — HH-канал. `reason_code → {text, terminal, author}` + инварианты загрузки.

Реестр канальный (тексты и набор ключей — свои), проверки те же три, что в TG, и работают
они ПРИ ИМПОРТЕ: непустой текст · у каждого кода есть производитель · множества «коды модели» и
«коды кода» не пересекаются.

Дельта к TG по ключам:
- отсев по формату/локации — `KO_FORMAT`, `KO_LOCATION` (подстановка `{city}`), `KO_LOCATION_GEO`
  вместо `KO_FORMAT_OFFICE/_HYBRID/_NOCITY` и `KO_GEO`;
- нет `REPLY_CONTACT_SOURCE`: источника контакта в канале нет.
"""

from qa_harness.domain.screening_split.policy.reasons import CODE, MODEL, NON_SCRIPT, Reason

from .budgets import EVENT_BUDGETS, REASK_BUDGETS
from .observation import TERMINAL_SIGNAL_REASON

DEFAULT_SUBSTITUTIONS = {"city": ""}

_KO = {
    "KO_SALARY": Reason(
        "Понимаю ваши ожидания, но, к сожалению, бюджет на эту позицию не позволяет их "
        "рассмотреть. Желаю вам удачи в дальнейших поисках!", True, CODE),
    "KO_LOCATION": Reason(  # {city} = поле «Локация» вакансии; пустая подстановка текст не ломает
        "Поняла вас, спасибо. К сожалению, для этой позиции важна локация {city}. "
        "В любом случае, спасибо за уделённое время!", True, CODE),
    "KO_LOCATION_GEO": Reason(
        "Поняла вас, спасибо. К сожалению, по локационным требованиям эта позиция сейчас не "
        "совпадает с вашей ситуацией. В любом случае, спасибо за уделённое время!", True, CODE),
    "KO_FORMAT": Reason(
        "Поняла вас, спасибо. К сожалению, по формату работы эта позиция сейчас не совпадает "
        "с вашими ожиданиями. В любом случае, спасибо за уделённое время!", True, CODE),
}

_STOP_BY_SIGNAL = {
    # В прежнем реестре `STOP_POLITICS` рендерился пустой строкой — кандидат не получал ничего.
    # Инвариант «ни один код не рендерится в пустоту» этого не допускает: текст общий с прочими
    # ситуативными стопами, как и в TG.
    "STOP_POLITICS": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_ABUSE": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_FLIRT": Reason("Прошу прощения, общение может вестись только в деловом формате.", True, MODEL),
    "STOP_GRIEF": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_MONEY_REQUEST": Reason(
        "Понимаю, что финансовый вопрос важен. К сожалению, в рамках первичного скрининга я не "
        "уполномочена обсуждать финансовые вопросы.", True, MODEL),
    "STOP_FOREIGN_LANG": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_FRAUD_CHECK": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_ALREADY_EMPLOYED": Reason("Поняла вас. Прошу прощения за беспокойство.", True, MODEL),
    "STOP_NOT_INTERESTED": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_NO_EXPERIENCE": Reason("Поняла вас, спасибо за честность. Прошу прощения за беспокойство.", True, MODEL),
    "STOP_CRITICISM": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_TASK_REQUEST": Reason("Прошу прощения за беспокойство.", True, MODEL),
    "STOP_MATERNITY": Reason("Извините за беспокойство!", True, MODEL),
}

_STOP_BY_BUDGET = {
    "STOP_BOT_REPEAT": Reason("Прошу прощения за беспокойство.", True, CODE),
    "STOP_GIBBERISH_REPEAT": Reason("Прошу прощения за беспокойство.", True, CODE),
    "STOP_SALARY_DEMAND": Reason("Прошу прощения за беспокойство.", True, CODE),
    "STOP_PERSISTENT": Reason("Прошу прощения за беспокойство.", True, CODE),
    # Как в TG: «сама напишу вам позже» — невыполняемое обещание, текст общий с прочими стопами.
    "STOP_PAUSE": Reason("Прошу прощения за беспокойство.", True, CODE),
}

_CONTINUE = {
    "FINISH": Reason(
        "Отлично, спасибо большое за ответы! Это вся информация, которая была мне нужна на данном "
        "этапе. Я передам её внутреннему рекрутеру, и он свяжется с вами по поводу следующих шагов.",
        True, MODEL),
    "REPLY_FALLBACK": Reason(
        "Прошу прощения, давайте продолжим. Подскажите, пожалуйста, можем ли мы вернуться к "
        "обсуждению вакансии?", False, CODE),
}

REASONS: dict[str, Reason] = {**_KO, **_STOP_BY_SIGNAL, **_STOP_BY_BUDGET, **_CONTINUE}


def _producible() -> set[str]:
    produced = set(TERMINAL_SIGNAL_REASON.values())
    produced |= {b.reason for b in EVENT_BUDGETS.values() if b.reason}
    produced |= {b.reason for b in REASK_BUDGETS.values() if b.reason} - NON_SCRIPT
    produced |= set(_KO)                      # правила R4–R6
    produced |= {"FINISH", "REPLY_FALLBACK"}  # R9, R2
    return produced


def validate() -> None:
    """Три инварианта реестра. Падение — на старте процесса, а не в диалоге с кандидатом."""
    empty = sorted(code for code, r in REASONS.items() if not r.text.strip())
    if empty:
        raise ValueError(f"реестр причин hh: пустой текст у {empty}")

    produced = _producible()
    orphans = sorted(set(REASONS) - produced)
    if orphans:
        raise ValueError(f"реестр причин hh: код без производителя (рудимент) — {orphans}")

    missing = sorted(produced - set(REASONS))
    if missing:
        raise ValueError(f"реестр причин hh: производитель есть, текста нет — {missing}")

    by_model = {c for c, r in REASONS.items() if r.author == MODEL}
    by_code = {c for c, r in REASONS.items() if r.author == CODE}
    overlap = sorted(by_model & by_code)
    if overlap:
        raise ValueError(f"реестр причин hh: код и модель претендуют на одни ключи — {overlap}")


def is_terminal(reason_code: str) -> bool:
    """Завершает ли диалог. Из РЕЕСТРА, а не по префиксу ключа."""
    reason = REASONS.get(reason_code)
    return bool(reason and reason.terminal)


def is_known(reason_code: str) -> bool:
    return reason_code in REASONS


def render(reason_code: str, *, city: str = "") -> str | None:
    """Готовый текст с подстановкой `{city}` либо None для неизвестного кода."""
    reason = REASONS.get(reason_code)
    if reason is None:
        return None
    return reason.text.format(city=city or DEFAULT_SUBSTITUTIONS["city"])


def authored_by_model() -> frozenset[str]:
    return frozenset(c for c, r in REASONS.items() if r.author == MODEL)


validate()

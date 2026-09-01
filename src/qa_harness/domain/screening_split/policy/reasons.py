"""Реестр причин хода: `reason_code → {text, terminal, author}` + инварианты загрузки.

Сегодня причина хода наружу не выходит: уходит текст и флаг «завершено». При этом 11 из 28 ключей
рендерятся в дословно одинаковое «Прошу прощения за беспокойство» (`..scripts:45-67`), поэтому по
тексту мотив отказа не восстановить — ни в отчётности, ни в гейте.

Три проверки выполняются ПРИ ИМПОРТЕ и роняют процесс на старте, а не в диалоге с кандидатом:

  1. текст непустой — чинит `STOP_POLITICS`, который сегодня рендерится в пустую строку
     (`..scripts:44`) и кандидат не получает вообще ничего;
  2. у каждого кода есть производитель — правило, строка бюджета или сигнал. `STOP_ABROAD`
     (`..scripts:54-57`) до релиза не доживает: промпт его больше не эмитит, но `is_known` его
     ПРИНИМАЕТ, то есть модель может вернуть, а движок — завершить им диалог;
  3. множества «коды модели» и «коды кода» не пересекаются — запрет перестаёт быть строкой промпта
     (`screening_analyzer/v2/system.md:161`).

`REPLY_CONTACT_SOURCE_EMPTY` перестаёт быть ключом: сегодня `render_script` подменяет им ключ внутри
себя (`..scripts:121-122`). Стало — подстановка по умолчанию.
"""

from dataclasses import dataclass

from .budgets import EVENT_BUDGETS, REASK_BUDGETS
from .observation import TERMINAL_SIGNAL_REASON

# Автор кода: кто имеет право его произвести. Пересечения быть не должно (инвариант 3).
MODEL = "signal"   # производится сигналом модели через таблицу правил
CODE = "code"      # производится исключительно кодом: бюджеты, вердикты, фолбэк


@dataclass(frozen=True)
class Reason:
    text: str
    terminal: bool
    author: str


# Подстановки по умолчанию: пустое значение не должно превращаться в дыру в предложении.
DEFAULT_SUBSTITUTIONS = {"contact_source": "из базы кандидатов", "city": ""}

_KO = {
    "KO_SALARY": Reason(
        "Понимаю ваши ожидания, но, к сожалению, бюджет на эту позицию не позволяет их "
        "рассмотреть. Желаю вам удачи в дальнейших поисках!", True, CODE),
    # Тексты отказов разведены так же, как правила (Р18): здесь речь ТОЛЬКО про формат работы.
    # Прежние формулировки говорили «важно находиться в городе {city}» — это про локацию, и после Р18
    # они стали неверными: R6 срабатывает независимо от города, поэтому кандидат, живущий в городе
    # вакансии и отказавшийся ездить в офис, получал «важно находиться в городе Москва». Раньше это
    # работало случайно — правило требовало, чтобы «переезд не спасал», и отказавшийся почти всегда
    # оказывался иногородним. Про место теперь говорит `KO_LOCATION`.
    "KO_FORMAT_OFFICE": Reason(
        "Поняла вас, спасибо. К сожалению, эта позиция предполагает работу из офиса, и это "
        "обязательное условие. В любом случае, спасибо за уделённое время!", True, CODE),
    "KO_FORMAT_HYBRID": Reason(
        "Поняла вас, спасибо. К сожалению, эта позиция предполагает гибридный формат работы, и это "
        "обязательное условие. В любом случае, спасибо за уделённое время!", True, CODE),
    "KO_FORMAT_NOCITY": Reason(
        "Поняла вас, спасибо. К сожалению, эта позиция предполагает присутствие на рабочем месте, "
        "и это обязательное условие. В любом случае, спасибо за уделённое время!", True, CODE),
    # Локация стала отдельным пунктом повестки (Р18), значит у отказа по ней должен быть свой ключ и
    # свой текст: иначе в отчётности отказ по локации сливается с отказом по формату и причина
    # перестаёт читаться. Ключ авторства КОДА — промпт про него не знает, переиздание v3 не нужно.
    "KO_LOCATION": Reason(
        "Поняла вас, спасибо. К сожалению, для этой позиции важно находиться в городе {city}. "
        "В любом случае, спасибо за уделённое время!", True, CODE),
    "KO_GEO": Reason(
        "Спасибо за информацию. К сожалению, для этой позиции есть ограничения по локации или "
        "часовому поясу. Прошу прощения за беспокойство.", True, CODE),
}

_STOP_BY_SIGNAL = {
    # Пустой текст здесь был дефектом §12.7: диалог обрывался молча.
    # Тот же текст, что у остальных ситуативных стопов (abuse / grief / foreign_lang) — так это
    # починено в eggplant общей константой `_APOLOGY`. Отдельная формулировка тут не нужна:
    # дефект был в ПУСТОЙ строке, из-за которой кандидат не получал ничего, а не в тоне.
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
    "STOP_PAUSE": Reason(
        "Хорошо, тогда я сама напишу вам позже, когда будет удобнее. Спасибо за уделённое время!",
        True, CODE),
}

_CONTINUE = {
    "FINISH": Reason(
        "Отлично, спасибо большое за ответы! Это вся информация, которая была мне нужна на данном "
        "этапе. Я передам ее внутреннему рекрутеру, и он свяжется с вами по поводу следующих шагов.",
        True, MODEL),
    "REPLY_CONTACT_SOURCE": Reason(
        "Коллеги указали источник контакта как {contact_source}. Прошу прощения, если возникла "
        "путаница. Если вам удобно, можем продолжить общение по вакансии?", False, MODEL),
    "REPLY_FALLBACK": Reason(
        "Прошу прощения, давайте продолжим. Подскажите, пожалуйста, можем ли мы вернуться к "
        "обсуждению вакансии?", False, CODE),
}

REASONS: dict[str, Reason] = {**_KO, **_STOP_BY_SIGNAL, **_STOP_BY_BUDGET, **_CONTINUE}

# Исходы без текста: их не рендерят, они ведут в Интервьюера либо в молчание.
NON_SCRIPT = frozenset({"SILENT", "ANSWER_ONLY", "HOLD_PAUSE", "REFUSE_AND_ADVANCE"})


def _producible() -> set[str]:
    """Коды, у которых есть производитель. Всё остальное — рудимент."""
    produced = set(TERMINAL_SIGNAL_REASON.values())
    produced |= {b.reason for b in EVENT_BUDGETS.values() if b.reason}
    produced |= {b.reason for b in REASK_BUDGETS.values() if b.reason} - NON_SCRIPT
    produced |= set(_KO)                                   # правила R4–R6
    produced |= {"FINISH", "REPLY_CONTACT_SOURCE", "REPLY_FALLBACK"}  # R9, R10, R2
    return produced


def validate() -> None:
    """Три инварианта реестра. Падение — на старте процесса, а не в диалоге."""
    empty = sorted(code for code, r in REASONS.items() if not r.text.strip())
    if empty:
        raise ValueError(f"реестр причин: пустой текст у {empty}")

    produced = _producible()
    orphans = sorted(set(REASONS) - produced)
    if orphans:
        raise ValueError(f"реестр причин: код без производителя (рудимент) — {orphans}")

    missing = sorted(produced - set(REASONS))
    if missing:
        raise ValueError(f"реестр причин: производитель есть, текста нет — {missing}")

    by_model = {c for c, r in REASONS.items() if r.author == MODEL}
    by_code = {c for c, r in REASONS.items() if r.author == CODE}
    overlap = sorted(by_model & by_code)
    if overlap:
        raise ValueError(f"реестр причин: код и модель претендуют на одни ключи — {overlap}")


def is_terminal(reason_code: str) -> bool:
    """Завершает ли диалог. Выводится ИЗ РЕЕСТРА, а не из префикса ключа: сегодня терминальность
    считается по «всё, что не REPLY_*» (`..scripts:113-115`), и любой новый код молча становится
    терминальным по опечатке в имени."""
    reason = REASONS.get(reason_code)
    return bool(reason and reason.terminal)


def is_known(reason_code: str) -> bool:
    return reason_code in REASONS


def render(reason_code: str, *, city: str = "", contact_source: str = "") -> str | None:
    """Готовый текст с подстановками либо None для неизвестного кода."""
    reason = REASONS.get(reason_code)
    if reason is None:
        return None
    return reason.text.format(
        city=city or DEFAULT_SUBSTITUTIONS["city"],
        contact_source=(contact_source or "").strip() or DEFAULT_SUBSTITUTIONS["contact_source"],
    )


def authored_by_model() -> frozenset[str]:
    """Коды, которые вправе произвести сигнал модели. Всё остальное производит только код."""
    return frozenset(c for c, r in REASONS.items() if r.author == MODEL)


validate()

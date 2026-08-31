"""Контракт `Observation` — то, что возвращает Аналитик вместо `Decision`.

Ключевое отличие от `..decision.Decision`: здесь НЕТ полей `next_action`, `script_key`,
`instruction`, `asking`, `event`. Модель сообщает, ЧТО услышала; какой из услышанных сигналов
исполняется и что из этого следует — решает `core.decide()`.

Побочный эффект контракта: запрет на код-форсимые ключи (`KO_SALARY`, `STOP_PERSISTENT`,
`STOP_*_REPEAT`, `STOP_SALARY_DEMAND`, `STOP_PAUSE`, `REPLY_FALLBACK`) перестаёт быть строкой промпта
(`screening_analyzer/v2/system.md:161`) — их просто негде вернуть. Сегодня `scripts.is_known` их
принимает, то есть запрет держится только дисциплиной модели.

ВАЛИДАЦИЯ — мягкая, и это осознанно. Жёстко обязательны ДВА поля: `signals` и `focus_answered`.
Всё остальное при мусоре чинится дефолтом, а не роняет ход. Это перенос сегодняшнего решения
«`salary_claim` не входит в REQUIRED_FIELDS» (`..decision:16`, `..salary.read_claim`) на весь
контракт: ошибка в одном блоке наблюдения не должна стоить трёх перегенераций по полному prefill.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

# ── сигналы ───────────────────────────────────────────────────────────────────

# Терминальные, СТРОГО в порядке приоритета из промпта (`system.md:137`). Сегодня выбор верхнего
# делает модель («выпиши все сработавшие, потом возьми верхний по таблице») — это argmax по 19
# позициям в уме, и промахи на нём дали проблемы §12.12 и §12.13. Теперь argmax делает код.
TERMINAL_PRIORITY: tuple[str, ...] = (
    "politics",
    "abuse",
    "flirt",
    "grief",
    "money_request",
    "foreign_lang",
    "fraud_check",
    "already_employed",
    "not_interested",
    "no_experience",
    "criticism",
    "maternity",
    "task_request",
)

TERMINAL_SIGNAL_REASON: dict[str, str] = {
    "politics": "STOP_POLITICS",
    "abuse": "STOP_ABUSE",
    "flirt": "STOP_FLIRT",
    "grief": "STOP_GRIEF",
    "money_request": "STOP_MONEY_REQUEST",
    "foreign_lang": "STOP_FOREIGN_LANG",
    "fraud_check": "STOP_FRAUD_CHECK",
    "already_employed": "STOP_ALREADY_EMPLOYED",
    "not_interested": "STOP_NOT_INTERESTED",
    "no_experience": "STOP_NO_EXPERIENCE",
    "criticism": "STOP_CRITICISM",
    "maternity": "STOP_MATERNITY",
    "task_request": "STOP_TASK_REQUEST",
}

# Нетерминальные — диалог продолжается, но ход может быть обработан особым образом.
NONTERMINAL_SIGNALS: frozenset[str] = frozenset({
    "gibberish", "bot_check", "answer_aid", "salary_info",
    "contact_source", "company_info", "scheduling", "pause", "resume",
})

# Сигнал → счётчик состояния. Отображение ТОТАЛЬНОЕ по счётным ключам, кроме `demand`: он не триггер
# (в таблице приоритета `system.md:137` его нет вовсе), а надстройка над другим триггером — «настойчиво
# повторяет то же самое». Поэтому он приходит отдельным булевым полем `persistent`, а не сигналом.
SIGNAL_TO_COUNTER: dict[str, str] = {
    "gibberish": "gibberish",
    "bot_check": "bot_check",
    "salary_info": "salary_info",
    "contact_source": "contact_source",
    "pause": "pause",
}

ALL_SIGNALS: frozenset[str] = frozenset(TERMINAL_PRIORITY) | NONTERMINAL_SIGNALS

FOCUS_ANSWERED = frozenset({"substantive", "deflection", "refusal", "none"})

MAX_SIGNALS = 3  # больше трёх одновременно — признак того, что модель перечисляет, а не различает


# ── структуры ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signal:
    """Что услышала модель, с доказательством. `quote` — необходимое, но НЕ достаточное условие:
    сигнал с валидной цитатой всё равно может не исполниться (правила R3a/R3b). Обратное строже —
    сигнал без цитаты, найденной в реплике, отбрасывается: цена выдуманного `not_interested` —
    необратимо закрытый диалог."""

    code: str
    quote: str


@dataclass
class Observation:
    """Наблюдение за одним ходом. Ни одно поле не определяет исход — исход выбирает `decide()`."""

    signals: list[Signal] = field(default_factory=list)
    focus_answered: str = "none"
    persistent: bool = False
    facts: dict[str, Any] = field(default_factory=dict)
    answers: list[dict[str, Any]] = field(default_factory=list)
    salary_claim: Optional[dict] = None
    reply_material: list[dict[str, str]] = field(default_factory=list)
    # Диагностика: что пришло от модели, но было отброшено валидацией. В решениях не участвует,
    # нужна отчёту — иначе «сигнал не сработал» неотличимо от «сигнала не было».
    dropped: list[str] = field(default_factory=list)

    def codes(self) -> list[str]:
        return [s.code for s in self.signals]

    def has(self, code: str) -> bool:
        return any(s.code == code for s in self.signals)

    def terminal_codes(self) -> list[str]:
        """Сработавшие терминальные сигналы в порядке приоритета таблицы (не в порядке модели)."""
        seen = {s.code for s in self.signals}
        return [c for c in TERMINAL_PRIORITY if c in seen]


def snapshot(obs: Any) -> dict:
    """Компактный вид наблюдения для трассы отчёта: что модель УСЛЫШАЛА на этом ходе.

    По `decision`/`state` не восстановить, почему выиграло правило: у R6 два входа, и «отказался от
    формата» неотличимо от «отказался переезжать» (прогон 20260831_203510 — понять, какая ветка
    сработала, по отчёту было нельзя). Функция канало-независима: читает поля через `getattr`,
    поэтому годится и для hh-`Observation` с её `formats_ready`.
    """
    return {
        "signals": [s.code for s in getattr(obs, "signals", []) or []],
        "focus_answered": getattr(obs, "focus_answered", None),
        "persistent": bool(getattr(obs, "persistent", False)),
        "facts": dict(getattr(obs, "facts", {}) or {}),
        "answers": list(getattr(obs, "answers", []) or []),
        "dropped": list(getattr(obs, "dropped", []) or []),
    }


def _norm(text: str) -> str:
    """Сверка цитаты: регистр, неразрывные пробелы, длинные тире — как в `..salary._norm_text`."""
    t = (text or "").lower().replace(" ", " ").replace(" ", " ")
    for dash in "—–‒−":
        t = t.replace(dash, "-")
    return " ".join(t.split())


def parse_observation(raw: Any, message: str) -> tuple[Observation, str]:
    """(observation, "") если контракт соблюдён; иначе (observation-с-дефолтами, причина).

    Причина непустая — это сигнал качества промпта для слоя A, а НЕ команда перегенерировать ход:
    решает вызывающий. Жёстко невалидны только два случая — ответ не объект и `signals` не список.
    """
    if not isinstance(raw, dict):
        return Observation(), "наблюдение не объект"

    obs = Observation()
    problems: list[str] = []

    raw_signals = raw.get("signals")
    if not isinstance(raw_signals, list):
        return obs, "signals не список"

    norm_message = _norm(message)
    for item in raw_signals[:MAX_SIGNALS]:
        if not isinstance(item, dict):
            obs.dropped.append("сигнал не объект")
            continue
        code = (item.get("code") or "").strip()
        quote = (item.get("quote") or "").strip()
        if code not in ALL_SIGNALS:
            obs.dropped.append(f"неизвестный сигнал {code!r}")
            continue
        # Цитата обязана найтись в реплике. Направление отбрасывания безопасное: без сигнала
        # диалог продолжается, с выдуманным — может закрыться.
        if not quote or _norm(quote) not in norm_message:
            obs.dropped.append(f"{code}: цитата не найдена в реплике")
            continue
        obs.signals.append(Signal(code=code, quote=quote))

    if len(raw_signals) > MAX_SIGNALS:
        obs.dropped.append(f"сигналов больше {MAX_SIGNALS}, лишние отброшены")

    focus = raw.get("focus_answered")
    if focus in FOCUS_ANSWERED:
        obs.focus_answered = focus
    else:
        problems.append(f"focus_answered недопустим: {focus!r}")

    obs.persistent = bool(raw.get("persistent"))

    facts = raw.get("facts")
    obs.facts = facts if isinstance(facts, dict) else {}
    if facts is not None and not isinstance(facts, dict):
        problems.append("facts не объект — считаем пустыми")

    answers = raw.get("answers")
    if isinstance(answers, list):
        obs.answers = [a for a in answers if isinstance(a, dict) and a.get("key")]
    elif answers is not None:
        problems.append("answers не список — считаем пустым")

    claim = raw.get("salary_claim")
    obs.salary_claim = claim if isinstance(claim, dict) else None

    material = raw.get("reply_material")
    if isinstance(material, list):
        obs.reply_material = [m for m in material
                              if isinstance(m, dict) and isinstance(m.get("text"), str) and m["text"].strip()]
    elif material is not None:
        problems.append("reply_material не список — считаем пустым")

    return obs, "; ".join(problems)

"""Шлюз гардов: механическая работа над исходящей строкой — кодом, а не промптом (принцип П5).

Сегодня это секция «РАНТАЙМ-САНИТАЙЗЕР» в промпте Интервьюера: модель просят разбить по предложениям,
убрать дубли, снять прощальные формулы при наличии вопроса, вырезать служебное `END`. Она это делает
через раз, и проверить невозможно. Здесь то же самое — детерминированно.

Гарды двух РАЗНЫХ классов, и смешивать их не надо:

  КОСМЕТИЧЕСКИЕ (G0, G2, G5) — типографика, словарь форматов, схлопывание дублей. Риска ложного
  срабатывания нет: текст не теряет смысла ни при каком входе. Включаются сразу.

  ЗАЩИТНЫЕ (G1, G3, G4, G6–G9) — вырезают содержимое. Предотвращают вред (утечка вилки, выдуманный
  URL, подчинение prompt injection), но МОГУТ унести полезное. Включаются после теневого прогона с
  замером доли ложных вырезов.

Каждое срабатывание записывается в `trips` — иначе «гард сработал» неотличимо от «модель так и
написала», и отладка превращается в гадание.

Реализации регулярок и словарей переносятся из QA-канареек `..checks` (`_URL_RE`, `_EMOJI_RE`,
`_quoted_from`, `_salary_variants`): там они уже отлажены на 77 сценариях. Писать заново незачем.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ── словари и регулярки ───────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s<>\"'»)\],]+")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
_END_RE = re.compile(r"\bEND\b\.?", re.IGNORECASE)

# Служебные формулировки сида/инструкции, которые Интервьюер иногда протаскивает в текст кандидату.
_SERVICE_RE = re.compile(
    r"\[(?:Внутренняя инструкция|Сообщение кандидата|Система)[^\]]*\]", re.IGNORECASE)

_FORMAT_WORDS = {
    "remote": "удалённый формат",
    "hybrid": "гибридный формат",
    "office": "работа из офиса",
    "on_site": "работа из офиса",
}

# Прощальные формулы: снимаются, только если в тексте остался вопрос — иначе сообщение прощается и
# тут же о чём-то спрашивает, и кандидат не понимает, закончен диалог или нет.
_FAREWELL = (
    "всего доброго", "всего хорошего", "до свидания", "желаю удачи",
    "спасибо за уделенное время", "спасибо за уделённое время",
    "прошу прощения за беспокойство", "хорошего дня",
)

# Фразы-вердикты: результат наших внутренних проверок кандидату не сообщается никогда.
_VERDICTS = (
    "вы в бюджете", "в бюджет вы", "ожидания подходят", "ожидания укладываются",
    "локация подходит", "вы нам подходите", "проходите по вилке", "укладываетесь в вилку",
    "соответствуете требованиям", "вы прошли",
)

_QUOTE_MIN_WORDS = 7  # цепочка такой длины — уже цитирование, а не совпадение
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*", re.MULTILINE)


@dataclass
class GuardSpec:
    """Что гардам разрешено и что запрещено на этом ходе."""

    allow_urls: tuple[str, ...] = ()          # канонические ссылки; пусто — вырезать любые
    forbid_tokens: tuple[str, ...] = ()       # числовые формы вилки и прочие секреты
    candidate_texts: tuple[str, ...] = ()     # реплики кандидата: для G4 и для защиты его же чисел
    require_question: bool = False            # ход должен нести вопрос
    hidden_company: bool = False              # company_name = «СКРЫТО»: любые ссылки и домены вон


@dataclass
class GuardResult:
    text: str
    trips: list[str] = field(default_factory=list)
    needs_fallback: bool = False              # G10: текст непригоден, канал рендерит шаблон плана


# ── вспомогательное ───────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Разбиение на предложения, устойчивое к ссылкам.

    Точка в `example.com` — не конец предложения. Без маскирования URL разрывался пополам, и гарды
    начинали работать с обрывками: замер по 2344 сообщениям поймал ровно этот случай — предложение
    «Вот ссылка на вакансию: https://example.» вырезалось как цитата, а хвост «com/vacancies/...»
    оставался в тексте кандидату.
    """
    urls = _URL_RE.findall(text)
    masked = text
    for i, url in enumerate(urls):
        masked = masked.replace(url, f"\x00U{i}\x00", 1)

    out: list[str] = []
    for match in _SENTENCE_RE.finditer(masked):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        for i, url in enumerate(urls):
            sentence = sentence.replace(f"\x00U{i}\x00", url)
        out.append(sentence)
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


# Доля предложения, которую обязана покрывать цитата, чтобы предложение считалось цитированием.
# Без неё гард режет законный переспрос с числом кандидата («это ваша текущая или ожидания?»):
# в канарейке `..checks` порога не было, потому что она только ПОМЕЧАЕТ, а гард — РЕЖЕТ.
_QUOTE_COVERAGE = 0.6


def _longest_shared_chain(sources: tuple[str, ...], sentence: str, n: int) -> tuple[Optional[str], int]:
    """(самая длинная цепочка, общая с репликами кандидата; её длина в словах).

    Порт `..checks._quoted_from`, но ищется максимум, а не первое совпадение: по нему считается
    покрытие предложения.
    """
    target = _words(sentence)
    if len(target) < n:
        return None, 0
    best, best_len = None, 0
    src_sets = [_words(src) for src in sources]
    for size in range(len(target), n - 1, -1):
        chains = {tuple(target[i:i + size]) for i in range(len(target) - size + 1)}
        for src_words in src_sets:
            for i in range(len(src_words) - size + 1):
                chain = tuple(src_words[i:i + size])
                if chain in chains:
                    return " ".join(chain), size
        if best_len:
            break
    return best, best_len


# Просьба по-русски часто идёт повелительным наклонением без знака вопроса: «Уточните, пожалуйста,
# на какую сумму вы рассчитываете.» Проверять только «?» — значит объявлять такой ход безвопросным.
_REQUEST_MARKERS = (
    "уточните", "подскажите", "расскажите", "поделитесь", "напишите", "сообщите",
    "ответьте", "назовите", "укажите", "поясните", "готовы ли", "могли бы",
)


def _has_request(text: str) -> bool:
    """Несёт ли сообщение вопрос или просьбу."""
    if "?" in text:
        return True
    low = text.lower()
    return any(m in low for m in _REQUEST_MARKERS)


def _effective_forbidden(spec: GuardSpec) -> tuple[str, ...]:
    """`forbid_tokens` МИНУС то, что кандидат написал сам.

    Иначе кандидат, назвавший сумму, совпадающую с границей вилки, получил бы искалеченный ответ:
    вырезали бы его собственное число. Секрет — это НАШИ цифры, а не совпадение.
    """
    if not spec.candidate_texts:
        return spec.forbid_tokens
    said = " ".join(spec.candidate_texts).lower()
    return tuple(t for t in spec.forbid_tokens if t.lower() not in said)


# ── гарды ─────────────────────────────────────────────────────────────────────

def _g0_typography(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Косметика: длинное тире, кавычки, двойные пробелы, лишние переносы."""
    before = text
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), ("G0.типографика" if text != before.strip() else None)


def _g1_service(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Служебные строки и `END` — в текст кандидату не идут."""
    cleaned = _SERVICE_RE.sub("", text)
    cleaned = _END_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, ("G1.служебное" if cleaned != text.strip() else None)


def _g2_format_words(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Технические идентификаторы формата — русскими словами. Сегодня об этом просят и промпт
    Аналитика (`system.md:88`), и промпт Интервьюера; правило дублируется в двух местах и всё равно
    протекает сырым id (сценарий hh #57)."""
    tripped = False
    for raw, human in _FORMAT_WORDS.items():
        pattern = re.compile(rf"\b{re.escape(raw)}\b", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(human, text)
            tripped = True
    return text, ("G2.формат словами" if tripped else None)


def _g3_emoji(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Ни скрипты, ни промпты эмодзи не используют — любое совпадение значит, что ассистент
    подчинился кандидату (канарейка prompt injection, сценарии tg #60 / hh #51)."""
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, ("G3.эмодзи" if cleaned != text.strip() else None)


def _g4_quoting(text: str, spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Предложение, СОСТОЯЩЕЕ в основном из дословной цитаты кандидата, вырезается целиком.

    Порог покрытия обязателен. Замер по 2344 записанным сообщениям: без него гард уносил законный
    уточняющий вопрос вида «это ваша текущая зарплата или ожидания?», где сумма кандидата повторена
    ради ясности, а не ради цитирования.
    """
    if not spec.candidate_texts:
        return text, None
    kept, chain = [], None
    for sentence in _sentences(text):
        found, size = _longest_shared_chain(spec.candidate_texts, sentence, _QUOTE_MIN_WORDS)
        total = len(_words(sentence))
        if found and total and size / total >= _QUOTE_COVERAGE:
            chain = found
            continue
        kept.append(sentence)
    if chain is None:
        return text, None
    return " ".join(kept).strip(), f"G4.цитирование ({chain})"


def _g5_dedup(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Схлопывание повторяющихся предложений."""
    seen, kept, dropped = set(), [], False
    for sentence in _sentences(text):
        norm = " ".join(_words(sentence))
        if norm and norm in seen:
            dropped = True
            continue
        seen.add(norm)
        kept.append(sentence)
    return (" ".join(kept).strip(), "G5.дубль") if dropped else (text, None)


def _g6_farewell(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Прощание вместе с вопросом — сообщение противоречит само себе."""
    if "?" not in text:
        return text, None
    kept, dropped = [], False
    for sentence in _sentences(text):
        low = sentence.lower()
        if "?" not in sentence and any(f in low for f in _FAREWELL):
            dropped = True
            continue
        kept.append(sentence)
    return (" ".join(kept).strip(), "G6.прощание при вопросе") if dropped else (text, None)


def _g7_urls(text: str, spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Канонизация ссылок. Любой URL, которого нет в `allow_urls`, вырезается; при скрытой компании —
    любой вообще.

    Это лечение прод-инцидента 2026-08-17 (Баг A): Интервьюер контекста вакансии не видит и на
    директиву «укажи ссылку» без значения выдумывает URL. Запрет в промпте уже максимально прямой и
    всё равно воспроизводится — значит нужен код.
    """
    found = _URL_RE.findall(text)
    if not found:
        return text, None
    allowed = () if spec.hidden_company else spec.allow_urls
    bad = [u for u in found if u.rstrip(".,;") not in allowed]
    if not bad:
        return text, None
    kept = []
    for sentence in _sentences(text):
        if any(u in sentence for u in bad):
            without = _URL_RE.sub("", sentence).strip(" -–—:,")
            if len(_words(without)) >= 4:
                kept.append(without)
            continue
        kept.append(sentence)
    return " ".join(kept).strip(), f"G7.ссылка ({', '.join(bad[:2])})"


def _g8_verdicts(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Результаты наших проверок кандидату не сообщаются."""
    kept, hit = [], None
    for sentence in _sentences(text):
        low = sentence.lower()
        match = next((v for v in _VERDICTS if v in low), None)
        if match:
            hit = match
            continue
        kept.append(sentence)
    return (" ".join(kept).strip(), f"G8.вердикт ({hit})") if hit else (text, None)


def _g9_secret(text: str, spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Числовые формы зарплатной вилки — вон. Числа самого кандидата не трогаем (см.
    `_effective_forbidden`)."""
    tokens = _effective_forbidden(spec)
    if not tokens:
        return text, None
    kept, hit = [], None
    for sentence in _sentences(text):
        low = sentence.lower()
        match = next((t for t in tokens if t.lower() in low), None)
        if match:
            hit = match
            continue
        kept.append(sentence)
    return (" ".join(kept).strip(), f"G9.вилка ({hit})") if hit else (text, None)


GUARDS = (
    ("G0", _g0_typography),
    ("G1", _g1_service),
    ("G2", _g2_format_words),
    ("G3", _g3_emoji),
    ("G4", _g4_quoting),
    ("G5", _g5_dedup),
    ("G6", _g6_farewell),
    ("G7", _g7_urls),
    ("G8", _g8_verdicts),
    ("G9", _g9_secret),
)

COSMETIC = frozenset({"G0", "G2", "G5"})


def apply_guards(text: str, spec: GuardSpec, *, defensive: bool = True) -> GuardResult:
    """Прогон в фиксированном порядке.

    `defensive=False` — теневой режим: работают только косметические, защитные лишь ЗАПИСЫВАЮТ, что
    сработали бы. Так измеряется доля ложных вырезов до того, как гарды получат право резать.
    """
    result = GuardResult(text=text or "")
    removed = False  # тронул ли текст хоть один ЗАЩИТНЫЙ гард (косметика не в счёт)
    for code, guard in GUARDS:
        if not defensive and code not in COSMETIC:
            _, trip = guard(result.text, spec)
            if trip:
                result.trips.append(f"[тень] {trip}")
            continue
        result.text, trip = guard(result.text, spec)
        if trip:
            result.trips.append(trip)
            if code not in COSMETIC:
                removed = True

    # G10 — ремонт. Смотрит ТОЛЬКО на то, что натворили гарды: если ни один защитный не тронул
    # текст, портить его было некому, и «вопроса нет» — это свойство самого сообщения, а не наша
    # поломка. Замер по 2344 записанным сообщениям без этого условия дал 372 ложных срабатывания
    # (15,9 %): почти все — просьбы в повелительном наклонении без вопросительного знака.
    if not result.text.strip():
        result.needs_fallback = True
        result.trips.append("G10.пусто после гардов")
    elif removed and spec.require_question and not _has_request(result.text):
        result.needs_fallback = True
        result.trips.append("G10.вопрос потерян")
    return result


def salary_forms(amount: Optional[int]) -> tuple[str, ...]:
    """Числовые формы суммы, которых не должно быть в тексте кандидату. Порт `..checks._salary_variants`."""
    if not amount:
        return ()
    thousands = amount // 1000
    return (str(amount), f"{amount:,}".replace(",", " "),
            f"{thousands}к", f"{thousands} тыс", f"{thousands} тысяч", f"{thousands} т.р.")

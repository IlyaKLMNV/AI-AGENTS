"""Шлюз гардов: механическая работа над исходящей строкой — кодом, а не промптом (принцип П5).

В действующем движке это секция «РАНТАЙМ-САНИТАЙЗЕР» в промпте Интервьюера: модель просят разбить по
предложениям, убрать дубли, снять прощальные формулы при наличии вопроса, вырезать служебное `END`.
Она это делает через раз, и проверить невозможно. Здесь то же самое — детерминированно.

Гарды двух РАЗНЫХ классов, и смешивать их не надо:

  КОСМЕТИЧЕСКИЕ (G0, G2, G5) — типографика, словарь форматов, схлопывание дублей. Риска нет: текст не
  теряет смысла ни при каком входе. Включаются сразу.

  ЗАЩИТНЫЕ (G1, G3, G4, G6, G7) — меняют содержимое. Предотвращают вред (служебные строки, подчинение
  prompt injection, выдуманная ссылка), но МОГУТ унести полезное. Включаются после теневого замера.

ЧЕГО ЗДЕСЬ НАМЕРЕННО НЕТ:

  * гарда на числа зарплатной вилки. В новом контуре вилки нет ни в контексте Наблюдателя, ни у
    Интервьюера (он вакансию не видит вовсе), а миграция вырезает её из контекстов старых диалогов
    до первого вызова модели. Взяться числу неоткуда — предохранитель был бы мёртвым. Утечку вилки
    продолжает сторожить `leak_scan` в гейте: случись она, прогон покраснеет;
  * гарда на фразы-вердикты («вы в бюджете», «локация подходит»). Той же природы: результатов
    проверок в тексте для Интервьюера нет, произнести их неоткуда;
  * шага «ремонт»: подставлять кандидату собранную кодом `instruction` нельзя — это директива в
    повелительном наклонении, адресованная Интервьюеру, а не человеку. Если после гардов текста не
    осталось, канал отправляет `REPLY_FALLBACK`.

Каждое срабатывание записывается в `trips` — иначе «гард сработал» неотличимо от «модель так и
написала», и отладка превращается в гадание.
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

# Хвостовая пунктуация, которую регулярка URL захватывает вместе с адресом. Её надо отрезать: иначе
# «…engineer.» съедает точку конца предложения — разбиение на предложения ломается, и гард режет
# лишнее. Найдено при проверке подмены ссылки.
_URL_TRAIL = ".,;:!?)»]"


def _urls(text: str) -> list[str]:
    """Адреса из текста без хвостовой пунктуации."""
    return [u.rstrip(_URL_TRAIL) for u in _URL_RE.findall(text or "")]


_QUOTE_MIN_WORDS = 7  # цепочка такой длины — уже цитирование, а не совпадение
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*", re.MULTILINE)


@dataclass
class GuardSpec:
    """Что гардам известно об этом ходе."""

    allow_urls: tuple[str, ...] = ()        # каноническая ссылка вакансии; пусто — подменять нечем
    candidate_texts: tuple[str, ...] = ()   # реплики кандидата: нужны G4 (дословное цитирование)
    hidden_company: bool = False            # company_name = «СКРЫТО»: свою ссылку давать нельзя


@dataclass
class GuardResult:
    text: str
    trips: list[str] = field(default_factory=list)


# ── вспомогательное ───────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Разбиение на предложения, устойчивое к ссылкам.

    Точка в `example.com` — не конец предложения. Без маскирования URL разрывался пополам, и гарды
    начинали работать с обрывками: замер по 2340 сообщениям поймал ровно этот случай.
    """
    urls = _urls(text)
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
# в канарейке гейта порога не было, потому что она только ПОМЕЧАЕТ, а гард — РЕЖЕТ.
_QUOTE_COVERAGE = 0.6


def _longest_shared_chain(sources: tuple[str, ...], sentence: str, n: int) -> tuple[Optional[str], int]:
    """(самая длинная цепочка, общая с репликами кандидата; её длина в словах)."""
    target = _words(sentence)
    if len(target) < n:
        return None, 0
    src_sets = [_words(src) for src in sources]
    for size in range(len(target), n - 1, -1):
        chains = {tuple(target[i:i + size]) for i in range(len(target) - size + 1)}
        for src_words in src_sets:
            for i in range(len(src_words) - size + 1):
                if tuple(src_words[i:i + size]) in chains:
                    return " ".join(src_words[i:i + size]), size
    return None, 0


# ── гарды ─────────────────────────────────────────────────────────────────────

def _g0_typography(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Косметика: длинное тире, двойные пробелы, лишние переносы."""
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
    """Технические идентификаторы формата — русскими словами."""
    tripped = False
    for raw, human in _FORMAT_WORDS.items():
        pattern = re.compile(rf"\b{re.escape(raw)}\b", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(human, text)
            tripped = True
    return text, ("G2.формат словами" if tripped else None)


def _g3_emoji(text: str, _spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Ни скрипты, ни промпты эмодзи не используют — совпадение значит, что ассистент подчинился
    инструкции кандидата (канарейка prompt injection)."""
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, ("G3.эмодзи" if cleaned != text.strip() else None)


def _g4_quoting(text: str, spec: GuardSpec) -> tuple[str, Optional[str]]:
    """Предложение, СОСТОЯЩЕЕ в основном из дословной цитаты кандидата, вырезается целиком."""
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
    """Прощание вместе с вопросом — сообщение противоречит само себе.

    Терминальных ходов гард не касается вовсе: их текст берётся из реестра причин и до шлюза не
    доходит — Интервьюер на завершении не вызывается.
    """
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
    """Ссылки: чужую ПОДМЕНЯЕМ на нашу; своей нет — режем предложение целиком.

    Откуда вообще берётся чужая ссылка, если вопрос кандидату пишет код и URL в нём не упоминает:
    её пишет **Интервьюер**. Вакансию он не видит вовсе, поэтому любой URL в его тексте — выдумка.
    Разбор записанных прогонов: единственная выдуманная ссылка (`job.ivi.ru/vacancy/…`) в инструкции
    НЕ значилась — модель дописала её сама, когда кандидат спросил про вакансию. Промптом класс не
    лечится: запрет там прямой и всё равно нарушается.

    Подмена, а не вырезание: у вакансии есть каноническая ссылка, и кандидат ждёт именно её. Прежняя
    версия гарда вырезала URL и оставляла в тексте висящее «…ссылка» — предложение теряло смысл, а
    кандидат всё равно оставался без адреса.

    Своей ссылки нет (поле пустое или компания «СКРЫТО») — давать нечего, и предложение про ссылку
    целиком лишнее: режем его.
    """
    found = _urls(text)
    if not found:
        return text, None

    canon = spec.allow_urls[0].rstrip("/") if spec.allow_urls and not spec.hidden_company else None
    bad = [u for u in dict.fromkeys(found) if u.rstrip("/") != canon]
    if not bad:
        return text, None

    if canon:
        for url in bad:
            text = text.replace(url, canon)
        return text, f"G7.ссылка подменена ({bad[0]} → {canon})"

    kept = [s for s in _sentences(text) if not any(url in s for url in bad)]
    return " ".join(kept).strip(), f"G7.ссылка вырезана ({bad[0]})"


GUARDS = (
    ("G0", _g0_typography),
    ("G1", _g1_service),
    ("G2", _g2_format_words),
    ("G3", _g3_emoji),
    ("G4", _g4_quoting),
    ("G5", _g5_dedup),
    ("G6", _g6_farewell),
    ("G7", _g7_urls),
)

COSMETIC = frozenset({"G0", "G2", "G5"})


def apply_guards(text: str, spec: GuardSpec, *, defensive: bool = True) -> GuardResult:
    """Прогон в фиксированном порядке.

    `defensive=False` — теневой режим: работают только косметические, защитные лишь ЗАПИСЫВАЮТ, что
    сработали бы. Так измеряется доля ложных вырезов до того, как гарды получат право резать.
    """
    result = GuardResult(text=text or "")
    for code, guard in GUARDS:
        if not defensive and code not in COSMETIC:
            _, trip = guard(result.text, spec)
            if trip:
                result.trips.append(f"[тень] {trip}")
            continue
        result.text, trip = guard(result.text, spec)
        if trip:
            result.trips.append(trip)
    return result

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from adapters.adapters import names_from_cdm, to_vacancy_info

# -----------------------
# Константы и пути
# -----------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "screening_scenarios_hh"

DEFAULT_CSV_PATH = ROOT / "tests" / "fixtures" / "screening_scenarios_hh.csv"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm" / "hh"
DEFAULT_MESSAGES_PER_SCENARIO = 3

GEN_MODEL = "gpt-4.1-mini"
EVAL_MODEL = "gpt-4.1"
HH_COMPONENT_NAME = "screening_assistant_hh"

# -----------------------
# Общие утилиты
# -----------------------


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    if isinstance(usage, dict):
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_token_count")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_token_count")
            or 0
        )
        total_tokens = usage.get("total_tokens") or usage.get("token_count")
    else:
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", None) or getattr(
            usage, "token_count", None
        )
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    input_tokens, output_tokens, total_tokens = _extract_usage_numbers(usage)
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("END", "")
    return s.strip()


def _extract_json_substring(text: str) -> Optional[str]:
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()

    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return text[start : end + 1].strip()

    return None


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty json text")

    try:
        return json.loads(text)
    except Exception:
        extracted = _extract_json_substring(text)
        if not extracted:
            raise
        return json.loads(extracted)


def _extract_first_url(text: str) -> str:
    match = re.search(r"https?://\S+", str(text or ""))
    return match.group(0).rstrip(".,);]") if match else ""


# -----------------------
# Загрузка сценариев из CSV
# -----------------------


class Scenario:
    def __init__(
        self,
        index: int,  # номер строки в CSV (1..N)
        name: str,
        description: str,
        expected_behavior: str,
        examples_raw: str,
    ) -> None:
        self.index = index
        self.name = name.strip()
        self.description = (description or "").strip()
        self.expected_behavior = (expected_behavior or "").strip()
        self.examples_raw = examples_raw or ""


def parse_scenario_indices(raw: str) -> List[int]:
    tokens = [t.strip() for t in re.split(r"[,\s]+", raw or "") if t.strip()]
    if not tokens:
        return []

    values: List[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"Invalid scenario index '{token}' in --scenario-indices. "
                "Use comma-separated positive integers, e.g. 23,24"
            )
        idx = int(token)
        if idx <= 0:
            raise ValueError(f"Scenario index must be >= 1, got: {idx}")
        values.append(idx)

    return sorted(set(values))


def load_scenarios(csv_path: pathlib.Path) -> List[Scenario]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV with scenarios not found: {csv_path}")

    scenarios: List[Scenario] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        normalized_fields = {
            re.sub(r"\s+", " ", (field or "")).strip().lower(): field
            for field in fieldnames
        }

        def resolve_header(*candidates: str) -> str:
            for candidate in candidates:
                key = re.sub(r"\s+", " ", candidate).strip().lower()
                actual = normalized_fields.get(key)
                if actual:
                    return actual
            return ""

        index_key = resolve_header("№", "No", "N")
        if not index_key and fieldnames:
            index_key = fieldnames[0]

        name_key = resolve_header("Название сценария", "Название сценария")
        desc_key = resolve_header(
            "Описание сценария",
            "Краткое описание сценария",
            "Описание сценария",
            "Краткое описание сценария",
        )
        behavior_key = resolve_header(
            "Ожидаемое поведение модели (согласно промпту)",
            "Ожидаемое поведение модели (согласно промпту) ",
            "Ожидаемое поведение модели (согласно промпту)",
            "Ожидаемое поведение модели (согласно промпту) ",
        )
        examples_key = resolve_header(
            "Сообщениия с примерами диалогов ",
            "Сообщения с примерами диалогов",
            "Сообщениия с примерами диалогов ",
            "Сообщения с примерами диалогов",
        )

        for row_number, row in enumerate(reader, start=1):
            scenario_name = (row.get(name_key) or "").strip()
            if not scenario_name:
                continue

            raw_index = str(row.get(index_key) or "").strip()
            scenario_index = int(raw_index) if raw_index.isdigit() else row_number

            scenarios.append(
                Scenario(
                    index=scenario_index,
                    name=scenario_name,
                    description=(row.get(desc_key) or "").strip(),
                    expected_behavior=(row.get(behavior_key) or "").strip(),
                    examples_raw=(row.get(examples_key) or ""),
                )
            )

    return scenarios


def extract_candidate_examples(examples_raw: str, max_examples: int = 5) -> List[str]:
    """
    В колонке с примерами лежат json-объекты вида:
    {"dialog_id": "...", "full_text": "..."}

    Мы бьём по пустым строкам, парсим каждый json и вытаскиваем строки,
    которые начинаются с [candidate] / [кандидат].
    """
    if not examples_raw:
        return []

    blocks = [b for b in examples_raw.split("\n\n") if b.strip()]
    candidates: List[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        full_text = ""
        try:
            obj = json.loads(block)
            full_text = obj.get("full_text") or ""
        except Exception:
            full_text = block

        for line in full_text.splitlines():
            raw_line = line.strip()
            lower = raw_line.lower()
            if "[candidate]" in lower or "[кандидат]" in lower:
                try:
                    bidx = raw_line.index("]")
                    text = raw_line[bidx + 1 :].strip()
                except ValueError:
                    text = raw_line
                if text:
                    candidates.append(text)
                    if len(candidates) >= max_examples:
                        return candidates

    return candidates


# -----------------------
# Цепочки (multi-scenario, один диалог)
# -----------------------

REPEAT_MARKERS = [
    "повтор",
    "повторно",
    "настойчив",
    "снова",
    "еще раз",
    "ещё раз",
    "опять",
    "так и не ответили",
    "я уже спрашивал",
    "я же уже спрашивал",
    "второй раз",
    "третий раз",
]

TOPIC_SALARY = ["зарплат", "вилк", "оклад", "доход", "компенсац", "деньг", "bonus", "бонус"]

# Важно: "ии" и "ai" нельзя матчить подстрокой, иначе "компании" содержит "ии".
# Для коротких токенов используем матчи по границам слова.
TOPIC_BOT = ["бот", "нейросет", "искусствен", "ai", "ии"]

TOPIC_COMPANY = ["компан", "название компании", "работодатель", "ссылка", "сайт"]

# Явные цепочки по индексам строк CSV.
# Важно: порядок в списке - это порядок шагов в одном диалоге.
CHAIN_BY_INDEX: Dict[str, List[int]] = {
    "chain_salary_3x": [11, 22, 23],  # 1-й, 2-й, 3-й вопрос кандидата о ЗП
    "chain_bot_check": [19, 20],
}


CHAIN_BY_INDEX["chain_pause_resume_priority"] = [42, 43]
CHAIN_BY_INDEX["chain_pause_resume_questions"] = [44, 45]

LEGITIMACY_SCENARIOS = {7}
SALARY_NORMALIZATION_SCENARIOS = {34, 35, 36, 37, 38, 39}
PROFILE_REFERENCE_SCENARIOS = {40}
PAUSE_LATER_SCENARIOS = {42, 44}
PAUSE_LATER_RESUME_SCENARIOS = {43, 45}
HH_OPEN_COMPANY_SCENARIOS = {21}
HH_LOCATION_FORMAT_SCENARIOS = {26, 27, 28, 29, 31, 32, 33, 47, 48}
HH_SALARY_BUDGET_EDGE_SCENARIOS = {49}
HH_GROSS_SALARY_CLARIFICATION_SCENARIOS = {49}
HH_RECRUITING_AGENCY_SCENARIOS = {50}
HH_DISABLED_SCENARIOS: set[int] = set()
HH_STABLE_GENERATION_SCENARIOS = {
    8,
    10,
    11,
    12,
    15,
    16,
    18,
    19,
    20,
    21,
    22,
    23,
    26,
    31,
    27,
    28,
    29,
    30,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
}
FORCED_FALLBACK_SCENARIOS = (
    LEGITIMACY_SCENARIOS
    | SALARY_NORMALIZATION_SCENARIOS
    | PROFILE_REFERENCE_SCENARIOS
    | PAUSE_LATER_SCENARIOS
    | PAUSE_LATER_RESUME_SCENARIOS
    | HH_LOCATION_FORMAT_SCENARIOS
    | HH_STABLE_GENERATION_SCENARIOS
    | HH_SALARY_BUDGET_EDGE_SCENARIOS
    | HH_RECRUITING_AGENCY_SCENARIOS
)
HH_SPECIAL_FIXTURES = {
    "cdm_16.json",
    "cdm_17.json",
    "cdm_hh_01.json",
    "cdm_hh_02.json",
    "cdm_hh_03.json",
    "cdm_hh_04.json",
    "cdm_hh_05.json",
    "cdm_hh_06.json",
    "cdm_hh_07.json",
    "cdm_hh_08.json",
}

RAW_WORK_FORMAT_IDS = ("ON_SITE", "REMOTE", "HYBRID", "FIELD_WORK")
CONTEXT_LEAK_MARKERS = [
    "ваше имя:",
    "имя кандидата:",
    "детали вакансии:",
    "должность:",
    "название компании:",
    "формат работы:",
    "локация:",
    "описание вакансии:",
    "зарплатная вилка:",
    "приоритетные вопросы:",
    "дополнительные вопросы:",
    "[questions]",
]
RUDE_TONE_ALLOWED_SCENARIOS = {2, 8}
RECRUITING_AGENCY_NAME = "рекрутинговое агентство"
UNWANTED_RUDENESS_MARKERS = [
    "черт",
    "чёрт",
    "к черту",
    "к чёрту",
    "нах",
    "хер",
    "бред",
    "ерунд",
    "чуш",
    "задолб",
    "идиот",
    "туп",
    "урод",
    "каким же",
    "что за хрень",
    "возьми",
]


def _has_any(hay: str, needles: List[str]) -> bool:
    """
    Безопасный матч:
    - длинные подстроки ищем через "in"
    - короткие токены ("ии", "ai") ищем как отдельные слова/токены
    """
    h = (hay or "").lower()

    for n in needles:
        nn = (n or "").lower().strip()
        if not nn:
            continue

        # Короткие токены - только как отдельное слово/токен
        if nn in ("ии", "ai"):
            if re.search(rf"(?<!\w){re.escape(nn)}(?!\w)", h, flags=re.UNICODE):
                return True
            continue

        if nn in h:
            return True

    return False


def _is_repeated_by_name(name: str) -> bool:
    return _has_any(name, REPEAT_MARKERS)


def _contains_repeat_marker(text: str) -> bool:
    hay = (text or "").lower()
    extra_markers = [
        "\u0443\u0436\u0435 \u0441\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u043b",
        "\u044f \u0436\u0435 \u0443\u0436\u0435 \u0441\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u043b",
        "\u043f\u043e\u0432\u0442\u043e\u0440\u044e",
        "\u043f\u043e\u0432\u0442\u043e\u0440\u044f\u044e",
        "\u0435\u0449\u0435 \u0440\u0430\u0437",
        "\u0435\u0449\u0451 \u0440\u0430\u0437",
        "\u0441\u043d\u043e\u0432\u0430",
        "\u0432\u0442\u043e\u0440\u043e\u0439 \u0440\u0430\u0437",
        "\u0442\u0440\u0435\u0442\u0438\u0439 \u0440\u0430\u0437",
        "\u043e\u0442\u0432\u0435\u0442\u0430 \u043d\u0435 \u0431\u044b\u043b\u043e",
        "\u0432\u044b \u0442\u0430\u043a \u0438 \u043d\u0435",
    ]
    return _has_any(hay, REPEAT_MARKERS + extra_markers)


THIRD_TIME_MARKERS = [
    "\u0442\u0440\u0435\u0442\u0438\u0439 \u0440\u0430\u0437",
    "\u0432 \u0442\u0440\u0435\u0442\u0438\u0439 \u0440\u0430\u0437",
    "3-\u0439 \u0440\u0430\u0437",
    "3 \u0439 \u0440\u0430\u0437",
]

SALARY_ULTIMATUM_MARKERS = [
    "\u0431\u0435\u0437 \u0446\u0438\u0444\u0440 \u0441\u043c\u044b\u0441\u043b\u0430 \u043e\u0431\u0441\u0443\u0436\u0434\u0430\u0442\u044c \u043d\u0435\u0447\u0435\u0433\u043e",
    "\u0431\u0435\u0437 \u044d\u0442\u043e\u0439 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0438 \u0434\u043b\u044f \u043c\u0435\u043d\u044f \u0432\u0441\u0435 \u0440\u0430\u0437\u0433\u043e\u0432\u043e\u0440\u044b \u043f\u0443\u0441\u0442\u044b",
    "\u0431\u0435\u0437 \u044d\u0442\u043e\u0433\u043e \u043d\u0435 \u0434\u0432\u0438\u0433\u0430\u044e\u0441\u044c \u0434\u0430\u043b\u044c\u0448\u0435",
    "\u0431\u0435\u0437 \u044d\u0442\u043e\u0433\u043e \u0434\u0430\u043b\u044c\u0448\u0435 \u043d\u0435\u0442 \u0441\u043c\u044b\u0441\u043b\u0430",
    "\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0443 \u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u043e\u0441\u043b\u0435",
    "\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0443 \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u043e\u0441\u043b\u0435",
    "\u0435\u0441\u043b\u0438 \u0432\u0438\u043b\u043a\u0443 \u043d\u0435 \u0440\u0430\u0441\u043a\u0440\u044b\u0432\u0430\u0435\u0442\u0435",
    "\u0438\u043d\u0430\u0447\u0435 \u0434\u0430\u043b\u044c\u0448\u0435 \u043d\u0435\u0438\u043d\u0442\u0435\u0440\u0435\u0441\u043d\u043e",
    "\u0438\u043d\u0430\u0447\u0435 \u043d\u0435\u0432\u0438\u0436\u0443 \u0441\u043c\u044b\u0441\u043b\u0430",
]

KW_SPAM_LEGITIMACY_EXPLICIT = [
    "откуда у вас мои данные",
    "откуда мои данные",
    "откуда мой контакт",
    "где вы нашли мой профиль",
    "где нашли мой профиль",
    "это спам",
    "вы мошенники",
    "подтвердите легитимность",
    "подтвердите, что это не мошенники",
]

KW_GEO_OUTSIDE_RF = [
    "живу вне рф",
    "живу не в рф",
    "нахожусь вне рф",
    "вне рф",
    "живу за границей",
    "нахожусь за границей",
    "за пределами рф",
]

KW_GEO_TIMEZONE_ONLY = [
    "только свой часовой пояс",
    "только мой часовой пояс",
    "только в своем часовом поясе",
    "только в своём часовом поясе",
]

KW_GEO_NO_RELOCATION = [
    "переезд не рассматриваю",
    "не рассматриваю переезд",
    "релокация не интересует",
    "не готов к переезду",
]

KW_GEO_REMOTE_ONLY = [
    "только удаленка",
    "только удалёнка",
    "только удаленная работа",
    "только удалённая работа",
    "только удаленную работу",
    "только удалённую работу",
]

KW_ALREADY_CONTACTED_EXPLICIT = [
    "вы уже писали",
    "мы уже общались",
    "повторно это не актуально",
    "не нужно снова писать",
    "снова писать не нужно",
]

KW_NO_RELEVANT_PROFILE_EXPLICIT = [
    "этого опыта у меня нет",
    "такого опыта у меня нет",
    "это не мой профиль",
    "у меня другая специализация",
    "таким не занимаюсь",
]

KW_DEATH_LOSS = [
    "умер",
    "умерла",
    "покойн",
    "скончался",
    "скончалась",
    "после смерти",
    "похорон",
    "утрат",
    "не стало",
]

KW_WORK_FORMAT_READY = [
    "готов",
    "подходит",
    "устраивает",
    "могу ездить",
    "могу в офис",
    "готов в офис",
    "готов на гибрид",
]

KW_WORK_FORMAT_NOT_READY = [
    "не готов",
    "не готова",
    "офис не подходит",
    "гибрид не подходит",
    "не готов в офис",
    "не готова в офис",
    "не готов на гибрид",
    "не готова на гибрид",
    "не готов ездить",
    "не готова ездить",
    "переезд не рассматриваю",
    "не рассматриваю переезд",
]

SCENARIO_EXAMPLE_OVERRIDES: Dict[int, List[str]] = {
    5: [
        "Вы очень приятная, я бы хотел пригласить вас на свидание.",
        "Если честно, мне хотелось бы познакомиться с вами лично, а не только обсуждать вакансию.",
        "Вы мне понравились, давайте лучше поужинаем вместе и познакомимся поближе.",
    ],
    7: [
        "Вы мошенники?",
        "Это спам или официальное сообщение по вакансии?",
        "Подтвердите, пожалуйста, что это не развод.",
        "Если это реальное предложение, пришлите, пожалуйста, корпоративную почту.",
        "Нужно официальное подтверждение, что это не мошенники.",
    ],
    8: [
        "Описание вакансии выглядит слишком расплывчатым и непрофессиональным, не понимаю, что это за роль.",
        "Процесс выглядит сомнительно: по описанию вообще непонятно, какие тут реальные задачи.",
        "Вакансия сформулирована слишком слабо, поэтому впечатление от процесса сейчас скорее негативное.",
    ],
    9: [
        "Я сейчас живу не в РФ.",
        "Сейчас нахожусь за границей, не в России.",
        "Я живу за пределами РФ.",
    ],
    13: [
        "Расскажите, пожалуйста, подробнее про условия, формат работы и команду по этой вакансии.",
        "Можно коротко уточнить, какие условия и как устроен проект на этой роли?",
        "Подскажите, пожалуйста, подробнее про роль, формат работы и ключевые условия.",
    ],
    14: [
        "Подскажите, пожалуйста, какие задачи и обязанности у этой позиции?",
        "Можно подробнее рассказать про вакансию, проект и требования к кандидату?",
        "Хотелось бы уточнить, в чем основная суть роли и какие задачи будут в работе.",
    ],
    16: [
        "Я уже принял оффер и выхожу в новую компанию, поэтому предложение больше не рассматриваю.",
        "Контракт уже подписан, я трудоустроен и новые варианты сейчас не обсуждаю.",
        "У меня уже есть оффер, и я завершаю переход в новую компанию.",
    ],
    10: [
        "Это не мой профиль, у меня другая специализация.",
        "У меня другой профиль, такими ролями не занимаюсь.",
        "Это направление не по моей специализации.",
    ],
    18: [
        "У меня нет релевантного опыта в этой сфере, это другая специальность.",
        "Такого релевантного опыта у меня нет, у меня другая специализация.",
        "У меня нет опыта в этой сфере, это не мой профессиональный профиль.",
    ],
    19: [
        "Извините, это бот или человек?",
        "Подскажите, пожалуйста, вы ИИ или живой рекрутер?",
        "Скажите честно, мне сейчас бот пишет или человек?",
    ],
    20: [
        "Я уже спрашивал: вы бот или человек?",
        "Снова уточню: это автоматический бот или живой рекрутер?",
        "Повторно спрашиваю, кто мне пишет — бот или человек?",
    ],
    21: [
        "Подскажите, пожалуйста, что это за компания и что за вакансия?",
        "Какая компания, какие задачи и какой формат работы?",
        "Можно подробнее про компанию и позицию?",
    ],
    25: [
        "Сейчас живу в Москве, локация для меня подходит.",
        "Я нахожусь в Москве и могу рассматривать эту позицию.",
        "Проживаю в Москве, с локацией вакансии проблем нет.",
    ],
    30: [
        "После недавней утраты в семье я сейчас не готов обсуждать такие предложения.",
        "Недавно были похороны близкого человека, поэтому я пока не могу продолжать этот разговор.",
        "После смерти родственника мне сейчас тяжело возвращаться к таким обсуждениям.",
    ],
    31: [
        "Сейчас живу во Владивостоке, по зарплате ориентируюсь на 240 000 рублей на руки.",
        "Мой текущий город — Новосибирск, ожидания по деньгам около 250 000 рублей на руки.",
        "Я сейчас в Калининграде, по компенсации ориентируюсь на 235 000 рублей в месяц на руки.",
    ],
}

WORD_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def _contains_third_time_marker(text: str) -> bool:
    return _has_any(text or "", THIRD_TIME_MARKERS)


def _contains_salary_ultimatum(text: str) -> bool:
    return _has_any(text or "", SALARY_ULTIMATUM_MARKERS)


def _safe_salary_expectation_value(dialog_context_meta: Optional[Dict[str, Any]] = None) -> int:
    dialog_context_meta = dialog_context_meta or {}
    min_salary = _parse_int_value(dialog_context_meta.get("min_salary"))
    max_salary = _parse_int_value(dialog_context_meta.get("max_salary"))

    if min_salary is not None and max_salary is not None:
        value = int((min_salary + max_salary) / 2)
    elif max_salary is not None:
        value = int(max_salary * 0.8)
    elif min_salary is not None:
        value = min_salary
    else:
        value = 250_000

    value = max(50_000, int(round(value / 5_000) * 5_000))
    if max_salary is not None:
        value = min(value, max_salary)
    return value


def _has_salary_expectation_not_above_budget(
    text: str,
    dialog_context_meta: Optional[Dict[str, Any]] = None,
) -> bool:
    dialog_context_meta = dialog_context_meta or {}
    m = _normalize_text(text).lower()
    max_salary = _parse_int_value(dialog_context_meta.get("max_salary"))

    expectation_markers = [
        "зарплат",
        "по зарплате",
        "по деньгам",
        "по компенсации",
        "компенсац",
        "ориентир",
        "ожидан",
        "на руки",
        "рассчитываю",
    ]
    if not _contains_any_substring(m, expectation_markers):
        return False

    numbers = re.findall(r"\b\d{2,3}(?:[ \u00A0]?\d{3})?\b", m)
    numeric_values: List[int] = []
    for token in numbers:
        parsed = _parse_int_value(token)
        if parsed is not None:
            numeric_values.append(parsed)

    if not numeric_values:
        return False

    if max_salary is not None and any(value > max_salary for value in numeric_values):
        return False

    return True


def _canonical_work_format(work_format: str) -> str:
    low = (work_format or "").strip().lower()
    if low in {"on_site", "on-site"}:
        return "on_site"
    if low == "field_work":
        return "field_work"
    if "office" in low or "офис" in low:
        return "on_site"
    if "hybrid" in low or "гибрид" in low:
        return "hybrid"
    if "remote" in low or "удал" in low:
        return "remote"
    if "разъезд" in low or "field" in low:
        return "field_work"
    return low


def _is_office_or_hybrid_work_format(work_format: str) -> bool:
    return _canonical_work_format(work_format) in {"on_site", "hybrid"}


def _work_format_label(work_format: str) -> str:
    canonical = _canonical_work_format(work_format)
    if canonical == "on_site":
        return "Работа на месте работодателя"
    if canonical == "hybrid":
        return "Гибридный формат"
    if canonical == "remote":
        return "Удаленный формат"
    if canonical == "field_work":
        return "Разъездной формат"
    return str(work_format or "").strip()


def _work_format_phrase(work_format: str) -> str:
    canonical = _canonical_work_format(work_format)
    if canonical == "on_site":
        return "на месте работодателя"
    if canonical == "hybrid":
        return "в гибридном формате"
    if canonical == "remote":
        return "в удаленном формате"
    if canonical == "field_work":
        return "в разъездном формате"
    return str(work_format or "").strip()


def _parse_hh_work_format_ids(value: Any) -> List[str]:
    ids: List[str] = []
    if value is None:
        return ids
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return ids
        items = [part.strip() for part in text.split(",")]

    for item in items:
        raw = item.get("id") if isinstance(item, dict) else item
        canonical = _canonical_work_format(str(raw or "").strip())
        mapped = {
            "on_site": "ON_SITE",
            "hybrid": "HYBRID",
            "remote": "REMOTE",
            "field_work": "FIELD_WORK",
        }.get(canonical, str(raw or "").strip().upper())
        if mapped and mapped not in ids:
            ids.append(mapped)
    return ids


def _format_hh_work_format_for_context(value: Any) -> str:
    ids = _parse_hh_work_format_ids(value)
    return ", ".join(ids)


def _has_work_format_ready_marker(text: str) -> bool:
    return _has_any((text or "").lower(), KW_WORK_FORMAT_READY)


def _has_work_format_negative_marker(text: str) -> bool:
    return _has_any((text or "").lower(), KW_WORK_FORMAT_NOT_READY)


def _has_readiness_or_relocation_confirmation(
    text: str,
    dialog_context_meta: Optional[Dict[str, Any]] = None,
) -> bool:
    low = _normalize_text(text).lower()
    if _has_work_format_ready_marker(low):
        return True

    generic_markers = [
        "комфортно работать",
        "удобно работать",
        "готов работать",
        "готов ездить",
        "могу ездить",
        "смогу ездить",
        "без проблем ездить",
        "готов к переезду",
        "готова к переезду",
        "готов переехать",
        "готова переехать",
        "могу переехать",
        "рассматриваю переезд",
        "офис подходит",
        "гибрид подходит",
        "формат подходит",
        "меня устраивает офис",
        "меня устраивает гибрид",
        "мне подходит офис",
        "мне подходит гибрид",
        "готов к офису",
        "готова к офису",
        "готов к гибриду",
        "готова к гибриду",
        "готов работать в офисе",
        "готова работать в офисе",
        "могу работать в офисе",
        "комфортен гибридный формат",
        "комфортно работать в гибридном формате",
        "комфортно работать в офисе",
    ]
    if _contains_any_substring(low, generic_markers):
        return True

    dialog_context_meta = dialog_context_meta or {}
    location = _normalize_text(str(dialog_context_meta.get("location") or "")).lower()
    if location:
        location_markers = [
            f"работать в {location}",
            f"работать из {location}",
            f"ездить в {location}",
            f"добираться до {location}",
            f"в {location} мне комфортно работать",
            f"в {location} работать комфортно",
            f"в {location} работать удобно",
        ]
        if _contains_any_substring(low, location_markers):
            return True

    return False


QUESTION_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s*|\d+[.)]\s*)")


def _is_work_format_question_line(line: str) -> bool:
    low = _normalize_text(line).lower()
    if not low:
        return False

    explicit_markers = [
        "какой формат работы",
        "формат работы",
        "офис",
        "офисный",
        "гибрид",
        "гибридный",
        "удален",
        "удалён",
        "remote",
        "office",
        "hybrid",
        "готовы работать",
        "готов работать",
        "готовы ли работать",
        "подходит формат",
    ]
    return _contains_any_substring(low, explicit_markers)


def _sanitize_additional_questions(raw_questions: str) -> str:
    lines = [str(line).strip() for line in str(raw_questions or "").splitlines() if str(line).strip()]
    if not lines:
        return "-"

    kept: List[str] = []
    for line in lines:
        content = QUESTION_PREFIX_RE.sub("", line).strip()
        if not content:
            continue
        if _is_work_format_question_line(content):
            continue
        kept.append(content)

    if not kept:
        return "-"

    return "\n".join(f"{idx}. {content}" for idx, content in enumerate(kept, start=1))


def _has_death_loss_marker(text: str) -> bool:
    return _has_any((text or "").lower(), KW_DEATH_LOSS)


def _scenario_example_override(scenario: Scenario) -> Optional[List[str]]:
    if scenario.index == 4:
        # Для S4 не используем CSV-примеры: сообщения должны генерироваться моделью с нуля.
        return []
    override = SCENARIO_EXAMPLE_OVERRIDES.get(scenario.index)
    if override is None:
        return None
    return override[:]


def _is_foreign_language_message(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized or CYRILLIC_RE.search(normalized):
        return False
    tokens = [token for token in WORD_TOKEN_RE.findall(normalized) if len(token) > 1]
    return len(tokens) >= 2


def _matches_geo_restriction_trigger(text: str) -> bool:
    normalized = _normalize_text(text).lower()
    return _has_any(normalized, KW_GEO_OUTSIDE_RF)


def _scenario_chain_key(s: Scenario) -> Optional[str]:
    # 1) строго по индексам
    for chain_id, indices in CHAIN_BY_INDEX.items():
        if s.index in indices:
            return chain_id

    # 2) аккуратная авто-группировка: только если есть маркер повторности/настойчивости И тема
    n = s.name.lower()
    if _is_repeated_by_name(n):
        # Важно: сначала более "узкие" темы. И в любом случае bot-тема теперь безопасна.
        if _has_any(n, TOPIC_SALARY):
            return "chain_salary_3x"
        if _has_any(n, TOPIC_BOT):
            return "chain_bot_check"

    return None


def _chain_step_order(chain_id: str, scenario_index: int) -> int:
    order = CHAIN_BY_INDEX.get(chain_id) or []
    if scenario_index in order:
        return order.index(scenario_index)
    return 10_000 + scenario_index


@dataclass
class ScenarioGroup:
    group_id: str
    kind: str  # "single" | "chain"
    scenarios: List[Scenario]


@dataclass
class CdmFixture:
    file_name: str
    vacancy_info: Dict[str, Any]
    names: Dict[str, str]
    contact_source: str


def _normalized_low(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("ё", "е")).strip().lower()


def _copy_fixture_with_overrides(
    fixture: CdmFixture,
    *,
    vacancy_info_override: Optional[Dict[str, Any]] = None,
    names_override: Optional[Dict[str, Any]] = None,
    contact_source_override: Optional[str] = None,
) -> CdmFixture:
    vacancy_info = dict(fixture.vacancy_info or {})
    names = dict(fixture.names or {})
    if vacancy_info_override:
        vacancy_info.update(vacancy_info_override)
    if names_override:
        names.update({str(k): str(v) for k, v in names_override.items()})
    contact_source = fixture.contact_source if contact_source_override is None else str(contact_source_override)
    return CdmFixture(
        file_name=fixture.file_name,
        vacancy_info=vacancy_info,
        names=names,
        contact_source=contact_source,
    )


def _hh_recruiting_agency_context_override(
    group: ScenarioGroup,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if group.kind != "single" or len(group.scenarios) != 1:
        return None, None

    idx = group.scenarios[0].index
    if idx == 50:
        return (
            {
                "company_name": RECRUITING_AGENCY_NAME,
            },
            None,
        )

    return None, None


def _group_has_any_scenario(group: ScenarioGroup, indices: List[int]) -> bool:
    wanted = set(indices)
    return any(s.index in wanted for s in group.scenarios)


def _group_uses_prompt_v2_special_fixtures(group: ScenarioGroup) -> bool:
    return _group_has_any_scenario(
        group,
        [7, 21] + list(range(25, 51)),
    )


def _is_real_company_name(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    return normalized not in {"не указано", "скрыто", "n/a", "na", "none", "null", "-"}


def _fixture_matches_group_requirements(
    fixture: CdmFixture,
    group: ScenarioGroup,
) -> bool:
    vacancy_info = fixture.vacancy_info or {}
    work_format = vacancy_info.get("work_format")
    location = str(vacancy_info.get("location") or "").strip().lower()
    work_format_ids = _parse_hh_work_format_ids(work_format)
    company_name = str(vacancy_info.get("company_name") or "").strip()
    vacancy_description = str(vacancy_info.get("vacancy_description") or "").strip().lower()

    if _group_uses_prompt_v2_special_fixtures(group):
        if fixture.file_name not in HH_SPECIAL_FIXTURES:
            return False

    if _group_has_any_scenario(group, [21]):
        return fixture.file_name in {"cdm_hh_02.json", "cdm_hh_08.json"}

    if _group_has_any_scenario(group, [26]):
        return fixture.file_name == "cdm_hh_08.json"

    if _group_has_any_scenario(group, [27]):
        return "HYBRID" in work_format_ids and "моск" in location

    if _group_has_any_scenario(group, [28]):
        if not {"ON_SITE", "REMOTE"} & set(work_format_ids):
            return False
        return len(work_format_ids) >= 2

    if _group_has_any_scenario(group, [29]):
        return len(work_format_ids) >= 2 or "HYBRID" in work_format_ids or "ON_SITE" in work_format_ids

    if _group_has_any_scenario(group, [31]):
        return bool(location)

    if _group_has_any_scenario(group, [32]):
        return "REMOTE" in work_format_ids and "только из рф" in vacancy_description

    if _group_has_any_scenario(group, [33]):
        return "REMOTE" in work_format_ids and "важно находиться в москве" in vacancy_description

    if _group_has_any_scenario(group, [47]):
        return "FIELD_WORK" in work_format_ids

    if _group_has_any_scenario(group, [48]):
        return not location

    if _group_has_any_scenario(group, [49]):
        return fixture.file_name == "cdm_hh_01.json"

    if _group_has_any_scenario(group, [50]):
        return fixture.file_name == "cdm_hh_03.json"

    if _group_has_any_scenario(group, [40, 41, 42, 43, 44, 45]):
        return fixture.file_name in {"cdm_hh_01.json"}

    if _group_has_any_scenario(group, [34, 35, 36, 37, 38, 39]):
        return fixture.file_name in {"cdm_hh_01.json"}

    return True


def _select_cdm_fixture_for_group(
    cdm_fixtures: List[CdmFixture],
    group: ScenarioGroup,
    case_position: int,
) -> CdmFixture:
    eligible = [f for f in cdm_fixtures if _fixture_matches_group_requirements(f, group)]
    pool = eligible or cdm_fixtures
    return pool[(case_position - 1) % len(pool)]


def build_scenario_groups(scenarios: List[Scenario]) -> List[ScenarioGroup]:
    chain_map: Dict[str, List[Scenario]] = {}
    singles: List[Scenario] = []

    for s in scenarios:
        key = _scenario_chain_key(s)
        if key:
            chain_map.setdefault(key, []).append(s)
        else:
            singles.append(s)

    # сортируем внутри chain по порядку шагов цепочки
    for key in list(chain_map.keys()):
        chain_map[key] = sorted(chain_map[key], key=lambda x: _chain_step_order(key, x.index))

    groups: List[ScenarioGroup] = []
    used_chain = set()

    # сохраняем порядок групп по первому появлению в CSV
    for s in scenarios:
        key = _scenario_chain_key(s)
        if not key:
            groups.append(ScenarioGroup(group_id=f"single_{s.index}", kind="single", scenarios=[s]))
            continue

        if key in used_chain:
            continue
        used_chain.add(key)
        groups.append(ScenarioGroup(group_id=key, kind="chain", scenarios=chain_map[key]))

    return groups


# -----------------------
# Генерация сообщений кандидата (максимально близко к оригиналу + trigger forcing)
# -----------------------


def load_cdm_fixtures(cdm_dir: pathlib.Path) -> List[CdmFixture]:
    files = sorted(cdm_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No CDM fixtures found in: {cdm_dir}")

    fixtures: List[CdmFixture] = []
    for path in files:
        try:
            cdm = json.loads(path.read_text(encoding="utf-8"))
            vacancy_info = to_vacancy_info(cdm)
            names = names_from_cdm(cdm)

            vacancy = cdm.get("vacancy") or {}
            candidate = cdm.get("candidate") or {}
            vacancy_description = str(
                vacancy.get("vacancy_description")
                or vacancy.get("raw_vacancy")
                or vacancy.get("responsibilities")
                or ""
            ).strip()
            if vacancy_description:
                vacancy_info["vacancy_description"] = vacancy_description
            raw_work_format = vacancy.get("work_format")
            if raw_work_format not in (None, "", []):
                vacancy_info["work_format"] = raw_work_format

            # Fallback на raw CDM поля: используем только если в vacancy_info пусто.
            if not str(vacancy_info.get("location") or "").strip():
                raw_location = str(vacancy.get("location") or "").strip()
                if raw_location:
                    vacancy_info["location"] = raw_location

            if not str(vacancy_info.get("min_salary") or "").strip():
                raw_min_salary = vacancy.get("salary_range_from")
                if raw_min_salary is not None and str(raw_min_salary).strip():
                    vacancy_info["min_salary"] = str(raw_min_salary).strip()

            if not str(vacancy_info.get("max_salary") or "").strip():
                raw_max_salary = vacancy.get("salary_range_to")
                if raw_max_salary is not None and str(raw_max_salary).strip():
                    vacancy_info["max_salary"] = str(raw_max_salary).strip()

            contact_source = str(candidate.get("contact_source") or "").strip()

            fixtures.append(
                CdmFixture(
                    file_name=path.name,
                    vacancy_info=vacancy_info,
                    names=names,
                    contact_source=contact_source,
                )
            )
        except Exception as exc:
            print(f"[warn] Failed to load CDM fixture {path.name}: {type(exc).__name__}: {exc}")

    if not fixtures:
        raise ValueError(f"No valid CDM fixtures loaded from: {cdm_dir}")

    return fixtures


def _salary_range_text(vacancy_info: Dict[str, Any]) -> str:
    min_salary = str(vacancy_info.get("min_salary") or "").strip()
    max_salary = str(vacancy_info.get("max_salary") or "").strip()

    if min_salary and max_salary:
        return f"от {min_salary} до {max_salary} рублей"
    if min_salary:
        return f"от {min_salary} рублей"
    if max_salary:
        return f"до {max_salary} рублей"
    return ""


def _parse_int_value(value: Any) -> Optional[int]:
    raw = str(value or "").strip()
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _format_int_with_spaces(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _location_keywords(location: str) -> List[str]:
    loc = (location or "").strip().lower()
    if not loc:
        return []
    if "санкт" in loc or "петербург" in loc:
        return ["санкт", "петербург", "спб"]
    if "моск" in loc:
        return ["моск"]
    if "казан" in loc:
        return ["казан"]
    if "екатерин" in loc:
        return ["екатерин"]
    return [loc]


def _nearby_city_variants(location: str) -> List[Dict[str, Any]]:
    loc = (location or "").strip().lower()
    if not loc:
        return []
    if "моск" in loc:
        return [
            {"name": "Королев", "markers": ["королев", "королёв"], "message": "Живу в Королеве"},
            {"name": "Химки", "markers": ["химки", "химк"], "message": "Я в Химках"},
            {"name": "Подольск", "markers": ["подольск"], "message": "Я из Подольска"},
        ]
    if "новосибир" in loc:
        return [
            {"name": "Бердск", "markers": ["бердск"], "message": "Живу в Бердске"},
            {"name": "Обь", "markers": ["обь", "оби"], "message": "Я из Оби"},
            {"name": "Краснообск", "markers": ["краснообск"], "message": "Я в Краснообске"},
        ]
    if "санкт" in loc or "петербург" in loc:
        return [
            {"name": "Мурино", "markers": ["мурино"], "message": "Живу в Мурино"},
            {"name": "Кудрово", "markers": ["кудрово"], "message": "Я из Кудрово"},
            {"name": "Пушкин", "markers": ["пушкин", "пушкине"], "message": "Я в Пушкине"},
        ]
    return []


def _nearby_city_markers(location: str) -> List[str]:
    markers: List[str] = []
    for item in _nearby_city_variants(location):
        markers.extend([str(marker).lower() for marker in item.get("markers", [])])
    return markers


def _nearby_city_names(location: str) -> List[str]:
    return [str(item.get("name") or "").strip() for item in _nearby_city_variants(location) if str(item.get("name") or "").strip()]


def _far_city_variants(location: str) -> List[Dict[str, Any]]:
    loc = (location or "").strip().lower()
    if not loc:
        return []
    if "моск" in loc:
        return [
            {"name": "Новосибирск", "message": "Я сейчас в Новосибирске"},
            {"name": "Екатеринбург", "message": "Живу в Екатеринбурге"},
            {"name": "Казань", "message": "Я из Казани"},
        ]
    if "новосибир" in loc:
        return [
            {"name": "Москва", "message": "Я сейчас в Москве"},
            {"name": "Санкт-Петербург", "message": "Живу в Санкт-Петербурге"},
            {"name": "Казань", "message": "Я из Казани"},
        ]
    if "санкт" in loc or "петербург" in loc:
        return [
            {"name": "Новосибирск", "message": "Я сейчас в Новосибирске"},
            {"name": "Екатеринбург", "message": "Живу в Екатеринбурге"},
            {"name": "Казань", "message": "Я из Казани"},
        ]
    return [
        {"name": "Москва", "message": "Я сейчас в Москве"},
        {"name": "Санкт-Петербург", "message": "Живу в Санкт-Петербурге"},
        {"name": "Новосибирск", "message": "Я из Новосибирска"},
    ]


def _far_city_names(location: str) -> List[str]:
    return [str(item.get("name") or "").strip() for item in _far_city_variants(location) if str(item.get("name") or "").strip()]


def build_dialog_context(
    fixture: CdmFixture,
    hide_company: bool = False,
    contact_source_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    vacancy_info = fixture.vacancy_info
    names = fixture.names

    del hide_company
    del contact_source_override

    recruiter_name = str(names.get("recruiter_name") or "Рекрутер").strip()
    candidate_name = str(names.get("candidate_name") or "Кандидат").strip()
    title = str(vacancy_info.get("title") or "").strip()
    original_company_name = str(vacancy_info.get("company_name") or "").strip()
    company_name = original_company_name
    raw_work_format = vacancy_info.get("work_format")
    work_format = _format_hh_work_format_for_context(raw_work_format)
    location = str(vacancy_info.get("location") or "").strip()
    min_salary = str(vacancy_info.get("min_salary") or "").strip()
    max_salary = str(vacancy_info.get("max_salary") or "").strip()
    vacancy_description = str(vacancy_info.get("vacancy_description") or "").strip()
    vacancy_url = _extract_first_url(vacancy_description)
    salary = _salary_range_text(vacancy_info)
    questions = _sanitize_additional_questions(str(vacancy_info.get("questions") or ""))

    lines = [
        "# Контекст для диалога, будет предоставлен перед началом",
        f"Ваше имя: {recruiter_name}",
        f"Имя кандидата: {candidate_name}",
        "",
        "Детали вакансии:",
        f"Должность: {title}",
        f"Название компании: {company_name}",
        f"Формат работы: {work_format}",
        f"Локация: {location}",
        f"Описание вакансии: {vacancy_description}",
        f"Зарплатная вилка: {salary} (НЕ РАСКРЫВАТЬ!)",
        "",
        "Приоритетные вопросы:",
        "1. Зарплатные ожидания",
        "2. Локация / город",
        "",
        "Дополнительные вопросы:",
        questions,
    ]
    context_text = "\n".join(lines).strip()

    context_meta = {
        "cdm_file": fixture.file_name,
        "recruiter_name": recruiter_name,
        "candidate_name": candidate_name,
        "contact_source": str(fixture.contact_source or "").strip(),
        "title": title,
        "company_name": company_name,
        "original_company_name": original_company_name,
        "work_format": work_format,
        "work_format_raw": raw_work_format,
        "work_format_ids": _parse_hh_work_format_ids(raw_work_format),
        "location": location,
        "min_salary": min_salary,
        "max_salary": max_salary,
        "vacancy_description": vacancy_description,
        "vacancy_url": vacancy_url,
        "salary": salary,
        "questions": questions,
    }
    return context_text, context_meta


def build_vacancy_ref(dialog_context_meta: Dict[str, Any]) -> Dict[str, Any]:
    company = str(dialog_context_meta.get("company_name") or "").strip()
    if not company:
        company = str(dialog_context_meta.get("original_company_name") or "").strip()

    return {
        "title": str(dialog_context_meta.get("title") or "").strip(),
        "company": company,
        "vacancy_url": str(dialog_context_meta.get("vacancy_url") or "").strip(),
        "location": str(dialog_context_meta.get("location") or "").strip(),
        "work_format": str(dialog_context_meta.get("work_format") or "").strip(),
    }


def build_prompt_context_ref(dialog_context_meta: Dict[str, Any]) -> Dict[str, Any]:
    prompt_context: Dict[str, Any] = {
        "work_format_ids": list(dialog_context_meta.get("work_format_ids") or []),
        "min_salary": str(dialog_context_meta.get("min_salary") or "").strip(),
        "max_salary": str(dialog_context_meta.get("max_salary") or "").strip(),
        "salary": str(dialog_context_meta.get("salary") or "").strip(),
        "questions": str(dialog_context_meta.get("questions") or "").strip(),
        "vacancy_description": str(dialog_context_meta.get("vacancy_description") or "").strip(),
        "vacancy_url": str(dialog_context_meta.get("vacancy_url") or "").strip(),
        "location": str(dialog_context_meta.get("location") or "").strip(),
        "work_format": str(dialog_context_meta.get("work_format") or "").strip(),
    }
    original_company_name = str(dialog_context_meta.get("original_company_name") or "").strip()
    if original_company_name:
        prompt_context["original_company_name"] = original_company_name
    known_employer_name = str(dialog_context_meta.get("known_employer_name") or "").strip()
    if known_employer_name:
        prompt_context["known_employer_name"] = known_employer_name
    recruiting_agency_mode = str(dialog_context_meta.get("recruiting_agency_mode") or "").strip()
    if recruiting_agency_mode:
        prompt_context["recruiting_agency_mode"] = recruiting_agency_mode
    return prompt_context


def _short_reason(comment: str, limit: int = 140) -> str:
    normalized = re.sub(r"\s+", " ", str(comment or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


def _compact_problem_summary(dialogs: List[Dict[str, Any]]) -> str:
    if not dialogs:
        return ""
    total = len(dialogs)
    reason_counts: Dict[str, int] = {}
    scenario_counts: Dict[str, int] = {}
    for dialog in dialogs:
        problem = _short_reason(str(dialog.get("problem") or ""))
        if problem:
            reason_counts[problem] = reason_counts.get(problem, 0) + 1
        scenario_index = dialog.get("scenario_index")
        if scenario_index is not None:
            key = f"S{int(scenario_index)}"
            scenario_counts[key] = scenario_counts.get(key, 0) + 1

    top_reason = ""
    if reason_counts:
        top_reason = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    top_scenarios = ", ".join(
        f"{key} x{count}" for key, count in sorted(scenario_counts.items(), key=lambda item: (int(item[0][1:]), item[0]))
    )
    if top_reason and top_scenarios:
        return f"{total} failed turn(s). Main issue: {top_reason} [{top_scenarios}]"
    if top_reason:
        return f"{total} failed turn(s). Main issue: {top_reason}"
    return f"{total} failed turn(s)."


def _dialog_to_text(dialog: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for item in dialog:
        role = str(item.get("role") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        prefix = "[candidate]" if role == "candidate" else "[assistant]"
        lines.append(f"{prefix} {text}")
    return "\n".join(lines).strip()


def _hh_candidate_format_fit_phrase(dialog_context_meta: Dict[str, Any]) -> str:
    format_ids = list(dialog_context_meta.get("work_format_ids") or [])
    if "REMOTE" in format_ids:
        return "Удаленный формат мне подходит."
    if "HYBRID" in format_ids:
        return "Гибридный формат мне подходит."
    if "ON_SITE" in format_ids:
        return "Работа на месте работодателя мне подходит."
    if "FIELD_WORK" in format_ids:
        return "Разъездной формат мне подходит."
    return ""


def _hh_pause_questions_seed_messages(dialog_context_meta: Dict[str, Any]) -> List[str]:
    location = str(dialog_context_meta.get("location") or "").strip() or "Москва"
    salary_value = _format_int_with_spaces(_safe_salary_expectation_value(dialog_context_meta))
    format_phrase = _hh_candidate_format_fit_phrase(dialog_context_meta)

    intro = "Здравствуйте! Мне интересна эта вакансия."
    priority_parts = [
        f"Я нахожусь в {location}.",
        f"По зарплате ориентируюсь на {salary_value} рублей на руки в месяц.",
    ]
    if format_phrase:
        priority_parts.append(format_phrase)
    priority_reply = " ".join(priority_parts).strip()
    return [intro, priority_reply]


def _hh_profile_reference_seed_messages(dialog_context_meta: Dict[str, Any]) -> List[str]:
    return _hh_pause_questions_seed_messages(dialog_context_meta)


def build_mismatches(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for case in cases:
        case_type = str(case.get("type") or "")
        passed = bool(case.get("passed"))
        score_total = int(case.get("score_total", 0) or 0)
        turns_total = int(case.get("turns_total", 0) or 0)
        case_failed = (not passed) or (score_total < turns_total)
        if not case_failed:
            continue

        if case_type == "single":
            dialogs: List[Dict[str, Any]] = []
            for turn in case.get("turns") or []:
                if int(turn.get("score", 0)) != 0:
                    continue
                reason = _short_reason(str(turn.get("comment") or ""))
                dialogs.append(
                    {
                        "step": int(turn.get("step") or 0),
                        "candidate_message": str(turn.get("candidate_message") or ""),
                        "assistant_reply": str(turn.get("assistant_reply") or ""),
                        "problem": reason,
                    }
                )

            mismatch: Dict[str, Any] = {
                "case_id": str(case.get("case_id") or ""),
                "scenario_index": int(case.get("scenario_index") or 0),
                "scenario_name": str(case.get("scenario_name") or ""),
                "cdm_file": str(case.get("cdm_file") or ""),
                "company_hidden": bool(case.get("company_hidden", False)),
                "vacancy_ref": dict(case.get("vacancy_ref") or {}),
                "prompt_context": dict(case.get("prompt_context") or {}),
                "problem_summary": _compact_problem_summary(dialogs),
                "dialogs": dialogs,
            }
            mismatches.append(mismatch)
            continue

        if case_type == "chain":
            dialogs: List[Dict[str, Any]] = []
            for run in case.get("runs") or []:
                run_passed = bool(run.get("passed"))
                run_score_total = int(run.get("score_total", 0) or 0)
                run_turns_total = int(run.get("turns_total", 0) or 0)
                run_failed = (not run_passed) or (run_score_total < run_turns_total)
                if not run_failed:
                    continue

                for turn in run.get("turns") or []:
                    if int(turn.get("score", 0)) != 0:
                        continue
                    reason = _short_reason(str(turn.get("comment") or ""))
                    dialogs.append(
                        {
                            "run_index": int(run.get("run_index") or 0),
                            "step": int(turn.get("step") or 0),
                            "scenario_index": int(turn.get("scenario_index") or 0),
                            "scenario_name": str(turn.get("scenario_name") or ""),
                            "candidate_message": str(turn.get("candidate_message") or ""),
                            "assistant_reply": str(turn.get("assistant_reply") or ""),
                            "dialog_text": str(run.get("dialog_text") or ""),
                            "problem": reason,
                        }
                    )

            mismatches.append(
                {
                    "case_id": str(case.get("case_id") or ""),
                    "scenario_indices": list(case.get("scenario_indices") or []),
                    "scenario_names": list(case.get("scenario_names") or []),
                    "cdm_file": str(case.get("cdm_file") or ""),
                    "company_hidden": bool(case.get("company_hidden", False)),
                    "vacancy_ref": dict(case.get("vacancy_ref") or {}),
                    "prompt_context": dict(case.get("prompt_context") or {}),
                    "problem_summary": _compact_problem_summary(dialogs),
                    "dialogs": dialogs,
                }
            )

    return mismatches


def _token_usage_total(usage: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for bucket in usage.values():
        total["input_tokens"] += int(bucket.get("input_tokens", 0))
        total["output_tokens"] += int(bucket.get("output_tokens", 0))
        total["total_tokens"] += int(bucket.get("total_tokens", 0))
    return total


KW_POL_NATION = [
    "полит",
    "власть",
    "президент",
    "санкц",
    "войн",
    "пропаганд",
    "государств",
    "росси",
    "украин",
    "европ",
    "нато",
    "путин",
    "зеленск",
    "кремл",
]

KW_SALARY = ["зарплат", "вилк", "оклад", "деньг", "сколько", "компенсац", "bonus", "бонус", "gross", "net"]

# Здесь тоже "ии"/"ai" матчим безопасно.
KW_BOT = ["бот", "ии", "ai", "нейросет", "искусствен"]

KW_COMPANY = ["название компании", "какая компания", "кто вы", "где работ", "что за компания", "сайт", "ссылк"]

KW_SALARY_EXPECT = [
    "ожидан",
    "рассчитыва",
    "устроит",
    "достаточно",
    "на руки",
    "руб",
    "тыс",
]

KW_CITY_MATCH = [
    "город",
    "локац",
    "нахожусь",
    "живу",
    "переезд",
    "переехать",
    "переед",
    "санкт",
    "петербург",
    "спб",
]


def _is_repeated_dialog_scenario(scenario: Scenario) -> bool:
    idx = scenario.index
    name = scenario.name.lower()

    if idx in (20, 22, 23):
        return True

    if "настойчивый" in name or "повторно" in name:
        return True

    return False


def _trigger_requirement_text(s: Scenario) -> str:
    """
    Текстовое требование "что должно быть в сообщении", чтобы триггер точно сработал.
    Это добавляется в prompt генерации.
    """
    idx = s.index
    name = s.name.lower()

    # 1. Политические или национальные высказывания кандидата
    if idx == 1 or "полит" in name or "национ" in name or "страна" in name or "власть" in name:
        return (
            "В КАЖДОЙ реплике должен быть ЯВНЫЙ политический/национальный контекст как высказывание, а не нейтральный вопрос.\n"
            "Пример формата: эмоциональная позиция по власти/санкциям/войне/странам, условия вида "
            "\"принципиально не обсуждаю вакансии из <страна> из-за <причина>\", "
            "\"из-за санкций/политики\".\n"
            "Запрещено: разжигать ненависть, унижать людей по национальности.\n"
        )

    if idx == 4 or "иностранн" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен писать полностью на иностранном языке, НЕ на русском и без кириллицы.\n"
            "Используй нормальный текст на английском, немецком, испанском или другом иностранном языке, а не междометия и не одно слово.\n"
        )

    if idx == 7 or "недоверие" in name or "спам" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен использовать ТОЛЬКО явные формулировки триггера про легитимность / мошенничество.\n"
            "Допустимые смыслы: «это спам?», «вы мошенники?», «подтвердите легитимность», «пришлите корпоративную почту», «нужно официальное подтверждение».\n"
            "Запрещено использовать формулировки только про источник контакта вроде «откуда у вас мои данные» или «где вы нашли мой профиль».\n"
        )

    # 11/22/23 - про зарплату
    if idx == 11 or "спрашивает о зарплате" in name or "зарплате или условиях" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Это ПЕРВЫЙ запрос: запрещены маркеры повторности вроде «я уже спрашивал», «повторяю вопрос», «вы так и не ответили».\n"
            "Не добавляй шум про компанию, сайт, ссылку или работодателя: вопрос должен быть только про зарплату / вилку / условия.\n"
        )

    if idx == 22 or ("повтор" in name and _has_any(name, TOPIC_SALARY)):
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Это ПОВТОРНЫЙ запрос: обязательно укажи, что кандидат уже спрашивал и ответа не получил.\n"
            "Не добавляй вопросы про компанию, сайт, ссылку на вакансию или работодателя.\n"
        )

    if idx == 23 or ("трет" in name and _has_any(name, TOPIC_SALARY)):
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Это ТРЕТИЙ запрос: обязательно укажи, что кандидат уже несколько раз спрашивал и ответа не было.\n"
            "Не добавляй вопросы про компанию, сайт, ссылку на вакансию или работодателя.\n"
        )

    # 21 - компания при открытом поиске
    if idx == 21:
        return (
            "В КАЖДОЙ реплике кандидат должен просить рассказать о компании и самой вакансии: чем занимается компания, какие задачи, какой формат работы, в чем суть роли.\n"
            "Это не спор и не претензия, а обычный информативный вопрос кандидата.\n"
        )

    # 19/20 - бот
    if idx == 19:
        return (
            "В КАЖДОЙ реплике кандидат должен спрашивать: вы бот/ИИ или человек?\n"
            "Это ПЕРВЫЙ такой вопрос. Не пиши, что кандидат уже спрашивал раньше, и не используй формулировки"
            " «повторяю», «в третий раз», «я уже спрашивал».\n"
        )
    if idx == 20 or ("повторно" in name and "бот" in name):
        return (
            "В КАЖДОЙ реплике кандидат должен спрашивать: вы бот/ИИ или человек?\n"
            "Это ПОВТОРНЫЙ вопрос. Явно укажи, что кандидат уже спрашивал об этом раньше и ответа не получил.\n"
        )

    if idx == 9 or "географические ограничения" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен ЯВНО сказать, что живет вне РФ.\n"
            "Достаточно прямой формулировки: «живу не в РФ», «нахожусь за границей», «живу за пределами РФ».\n"
            "Не используй мягкие и двусмысленные формулировки.\n"
        )

    if idx == 16 or "оффер" in name or "трудоустроен" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен ЯВНО сообщать, что уже принял оффер, уже трудоустроен или уже выходит в новую компанию.\n"
            "Допустимые формулировки: «уже принял оффер», «уже выхожу на новую работу», «уже трудоустроен», «контракт подписан».\n"
            "Не подменяй этот триггер жалобой на повторный контакт или мягким отказом без указания оффера/трудоустройства.\n"
        )

    # 24 - зарплатные ожидания ниже вилки
    if idx == 24:
        return (
            "В КАЖДОЙ реплике кандидат должен назвать СВОИ зарплатные ожидания конкретной суммой (числами),\n"
            "причем сумма должна быть ниже минимальной границы вилки вакансии.\n"
        )

    # 25 - город совпадает с вакансией
    if idx == 25:
        return (
            "В КАЖДОЙ реплике кандидат должен явно указать, что он уже находится в городе вакансии.\n"
            "Предпочтительно прямо назвать сам город вакансии и не использовать другие города.\n"
        )

    if idx == 26:
        return (
            "В КАЖДОЙ реплике кандидат должен явно указать, что он уже находится в городе вакансии, и сразу назвать зарплатные ожидания в рублях на руки.\n"
            "При этом кандидат НЕ должен подтверждать готовность к офису/гибриду: это должен уточнить ассистент.\n"
        )

    if idx == 27:
        return (
            "В КАЖДОЙ реплике кандидат должен явно указать, что живет рядом с Москвой, но не в самой Москве, и сразу назвать зарплатные ожидания в рублях на руки.\n"
            "Кандидат должен назвать конкретный город рядом с Москвой, без упоминания километров и без подтверждения готовности к офису/гибриду.\n"
            "Это допустимый пригород/ближняя локация, и кандидат НЕ должен подтверждать готовность к офису/гибриду.\n"
        )

    if idx == 28:
        return (
            "В КАЖДОЙ реплике кандидат должен прямо показать, что один из форматов работы ему не подходит, но другой допустимый формат он готов рассматривать.\n"
            "Не превращай это в полный отказ от вакансии.\n"
        )

    if idx == 29:
        return (
            "В КАЖДОЙ реплике кандидат должен явно писать, что не готов работать в формате офис/гибрид "
            "и не готов к переезду.\n"
        )

    if idx == 30:
        return (
            "В КАЖДОЙ реплике кандидат должен явно упоминать смерть, утрату, похороны или покойного человека.\n"
            "Не добавляй в эту же реплику зарплатные ожидания, город, готовность к формату работы или другие триггеры.\n"
        )

    if idx == 31:
        return (
            "В КАЖДОЙ реплике кандидат должен назвать конкретный город, который явно далеко от города вакансии, и сразу назвать зарплатные ожидания в рублях на руки.\n"
        )

    if idx == 46:
        return (
            "В КАЖДОЙ реплике кандидат должен спокойно и прямо спрашивать только про формат работы по вакансии.\n"
            "Допустимые смыслы: «какой формат работы», «офис / удаленка / гибрид?», «какой именно формат у роли».\n"
            "Запрещено добавлять грубость, агрессию, мат, обвинения или побочные темы.\n"
        )

    if idx == 10 or ("другой профиль" in name or "другая специализация" in name):
        return (
            "В КАЖДОЙ реплике кандидат должен явно показать, что вакансия не по его профилю или что у него другая специализация.\n"
            "Используй формулировки уровня «это не мой профиль», «у меня другая специализация», «это направление не по моему профилю».\n"
            "Не делай акцент на фразах «нет опыта» или «нет релевантного опыта».\n"
        )

    if idx == 18 or "нет нужного опыта" in name or "отсутствие необходимого" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен явно показать, что у него нет релевантного опыта в этой сфере или что у него другая специальность.\n"
            "Допустимые формулировки: «у меня нет релевантного опыта», «у меня нет опыта в этой сфере», «это другая специальность», «таким не занимаюсь».\n"
        )

    return ""


def _extra_generation_guidelines(scenario: Scenario) -> str:
    idx = scenario.index
    name = scenario.name.lower()

    parts: List[str] = []

    # 1. Политические/национальные высказывания
    if idx == 1 or "полит" in name or "национ" in name:
        parts.append(
            "- Реплики должны явно содержать политический или национально-географический контекст (власть/страны/санкции/война).\n"
            "- Это должно быть высказывание/позиция/условие, а не просто нейтральное уточнение.\n"
            "- Без разжигания ненависти и без унижения людей по национальности.\n"
        )

    # 2. Агрессивное/оскорбительное
    if idx == 2 or "агрессивное" in name or "оскорбительное" in name:
        parts.append(
            "- Реплики должны быть явно грубыми и агрессивными.\n"
            "- Допустимы резкие формулировки, сарказм, раздражение и мат, "
            "но без угроз физического насилия и без обсуждения национальности, расы, пола и т.п.\n"
            "- Пусть из текста однозначно считывается, что кандидат ведёт себя хамски и неприязненно."
        )

    # 3. Бессмысленные/неразборчивые
    if idx == 3 or "неразборчив" in name or "бессмыслен" in name:
        parts.append(
            "- Пиши сообщения из бессмысленного набора букв.\n"
            "- Не используй мат и оскорбления, но пиши так, чтобы было понятно, что кандидат неадекватен."
        )

    # 4. Иностранные языки
    if idx == 4 or "иностранн" in name:
        parts.append(
            "- ВСЕ реплики кандидата должны быть на иностранном языке, НЕ на русском и без кириллицы.\n"
            "- Пиши полноценные короткие фразы на английском, немецком, испанском или другом иностранном языке."
        )

    if idx == 7 or "недоверие" in name or "спам" in name:
        parts.append(
            "- Используй только явные формулировки недоверия к легитимности контакта: «это спам?», «вы мошенники?», «подтвердите легитимность», «пришлите корпоративную почту», «нужно официальное подтверждение»."
        )
        parts.append(
            "- Не используй вопросы про источник контакта вроде «откуда у вас мои данные», «где вы нашли мой профиль» и другие мягкие сомнения: это не тот триггер."
        )

    if idx == 8 or "критика" in name or "процесса" in name:
        parts.append(
            "- Реплика должна содержать явную негативную оценку вакансии, описания, компании или процесса: расплывчато, сомнительно, непрофессионально, слабое описание, плохое впечатление."
        )
        parts.append(
            "- Это не нейтральный запрос деталей. Кандидат именно критикует или выражает недовольство."
        )

    if idx == 5 or "флирт" in name or "слишком лич" in name:
        parts.append(
            "- В каждой реплике кандидат должен переводить разговор в личную плоскость: делать личный комплимент, предлагать свидание, знакомство или романтическую встречу."
        )
        parts.append(
            "- Это не обычная просьба созвониться или обсудить вакансию: должен считываться именно флирт или слишком личная подача."
        )

    if idx == 12 or "техничес" in name or "алгоритм" in name:
        parts.append(
            "- В КАЖДОЙ реплике кандидат должен прямо просить выполнить техническое действие: написать код, решить задачу, объяснить алгоритм, оптимизировать решение или дать пример реализации."
        )
        parts.append(
            "- Обязательно используй явные маркеры из сценария: «код», «пример», «напиши», «реализовать», «как сделать», «алгоритм», «реши», «задача», «оптимизируй»."
        )

    if idx == 13 or "услови" in name:
        parts.append(
            "- В каждой реплике кандидат должен спокойно и по-деловому спрашивать про условия вакансии, формат работы, проект, команду или другие детали роли."
        )
        parts.append(
            "- Запрещено просить писать код, решать задачи, требовать созвон или переводить разговор в конфликт."
        )
    if idx == 14 or "ваканси" in name or "задач" in name:
        parts.append(
            "- В каждой реплике кандидат должен спокойно и по-деловому спрашивать про вакансию, задачи, проект, требования или обязанности по роли."
        )
        parts.append(
            "- Запрещено добавлять грубость, мат, сарказм, ультиматумы или негативные претензии к рекрутеру."
        )

    # 6. Неформальное/странное
    if idx == 6 or "неформальное" in name or "странное" in name:
        parts.append(
            "- Реплики должны быть в нейтральном ключе, без агрессии, заметно странными, нелогичными и не относящимися к теме к работе, вакансии или резюме, \n"
            "- Используй абсурдные сравнения, неожиданные ассоциации, резкие смены темы."
        )

    # 9. Географические ограничения
    if idx == 9 or "географические ограничения" in name:
        parts.append(
            "- В КАЖДОЙ реплике ЯВНО скажи, что кандидат живет вне РФ.\n"
            "- Достаточно прямой формулировки: «живу не в РФ», «нахожусь за границей», «живу за пределами РФ»."
        )

    # 11/22/23. Вопросы про зарплату
    if idx == 11 or "спрашивает о зарплате" in name or "зарплате или условиях" in name:
        parts.append(
            "- Кандидат ИМЕННО СПРАШИВАЕТ про вилку/зарплату, а не просто называет ожидания.\n"
            "- Обязательно явные вопросы: «какая вилка?», «сколько платите?», «какие условия?»."
        )
        parts.append(
            "- Не смешивай вопрос про зарплату с вопросами про компанию, ссылку на вакансию, сайт или работодателя."
        )

    # 22. Повторный вопрос про зарплату
    if idx == 22 or ("повтор" in name and _has_any(name, TOPIC_SALARY)):
        parts.append(
            "- Это повторный вопрос про зарплату.\n"
            "- В реплике должно быть явно: «я уже спрашивал», «вы так и не ответили», «повторяю вопрос»."
        )
        parts.append(
            "- Не добавляй вопросы про компанию, сайт, ссылку на вакансию или работодателя."
        )

    # 23. Третий вопрос про зарплату
    if idx == 23 or ("трет" in name and _has_any(name, TOPIC_SALARY)):
        parts.append(
            "- Это ТРЕТИЙ по счету вопрос про зарплату в одном диалоге.\n"
            "- В реплике должно быть явно: «в третий раз спрашиваю», «вы снова не ответили»."
        )
        parts.append(
            "- Реплика должна оставаться только про зарплату / вилку. Не добавляй тему компании, сайта, ссылки или работодателя."
        )

    # 16. Просьба денег
    if "просьба денег" in name:
        parts.append(
            "- В КАЖДОЙ реплике должна явно звучать просьба о деньгах (перевести/в долг/аванс).\n"
            "- Это именно просьба кандидата дать денег ему."
        )

    # 15. Созвон
    if idx == 15 or "созвон" in name or "назначить встречу" in name:
        parts.append(
            "- В репликах должна быть явная попытка назначить звонок/встречу.\n"
            "- Указывай конкретные слоты времени и формат (Zoom/Teams/телефон)."
        )

    # 21. Компания
    if idx == 21:
        parts.append(
            "- Кандидат просит рассказать о компании и вакансии в целом: чем занимается компания, какие задачи у роли, какой формат работы."
        )
        parts.append(
            "- Это обычный информативный вопрос без претензий и без требования раскрывать скрытый поиск."
        )

    # 24. Ожидания кандидата ниже вилки
    if idx == 24 or ("ожидан" in name and _has_any(name, TOPIC_SALARY)):
        parts.append(
            "- Кандидат НЕ спрашивает вилку, а называет свои ожидания по зарплате конкретной суммой (числами)."
        )
        parts.append(
            "- Сумма должна звучать как комфортная для кандидата и ниже типичной вилки вакансии."
        )

    # 25. Город совпадает или готовность к переезду
    if idx == 25 or ("город" in name and "ваканси" in name):
        parts.append(
            "- Кандидат явно подтверждает, что уже находится в городе вакансии."
        )
        parts.append(
            "- Предпочтительно прямо называть сам город вакансии и не использовать другие города."
        )

    if idx == 26:
        parts.append(
            "- Кандидат уже находится в городе вакансии, но не подтверждает отдельно готовность к формату офис/гибрид."
        )
        parts.append(
            "- Кандидат в этой же реплике уже отвечает и про зарплату: укажи конкретную сумму в рублях на руки, не выше верхней границы вилки вакансии."
        )
        parts.append(
            "- Запрещено самому подтверждать, что офисный или гибридный формат подходит: не пиши про готовность ездить, работать в офисе, работать в гибриде или переезжать."
        )
        parts.append(
            "- Формулировка должна оставить ассистенту необходимость отдельно спросить, подходит ли такой формат работы."
        )

    if idx == 27:
        parts.append(
            "- Кандидат должен назвать конкретный город рядом с Москвой: используй формулировки уровня «Живу в Королеве», «Я в Химках», «Я из Подольска»."
        )
        parts.append(
            "- Кандидат в этой же реплике уже отвечает и про зарплату: укажи конкретную сумму в рублях на руки, не выше верхней границы вилки вакансии."
        )
        parts.append(
            "- Не добавляй километры и не объясняй, что это рядом с Москвой: ассистент должен сам сделать этот вывод."
        )
        parts.append(
            "- Запрещено писать, что кандидату подходит офис/гибрид, что он готов ездить в Москву или что готов к переезду."
        )
        parts.append(
            "- Не добавляй готовность к офису/гибриду: ассистент должен сначала уточнить ее."
        )

    if idx == 28:
        parts.append(
            "- Кандидат должен явно показать, что один формат ему не подходит, но другой допустимый формат он готов рассматривать."
        )
        parts.append(
            "- Не делай реплику полным отказом от вакансии: должна оставаться возможность продолжить проверку другого формата."
        )

    if idx == 29:
        parts.append(
            "- Кандидат должен прямо отказаться от всех допустимых форматов работы и не оставлять пространства для продолжения по формату."
        )

    if idx == 29:
        parts.append(
            "- Кандидат явно пишет, что не готов работать в формате офис/гибрид и не готов к переезду."
        )
        parts.append(
            "- Это должно звучать как прямой стоп-фактор, а не как мягкое сомнение."
        )

    if idx == 30:
        parts.append(
            "- Кандидат прямо упоминает смерть, похороны, покойного человека или утрату."
        )
        parts.append(
            "- Не смешивай это с другими триггерами: основной смысл реплики должен быть именно про утрату."
        )
        parts.append(
            "- Не добавляй в реплику зарплату, город, готовность к формату работы или просьбы рассказать о вакансии."
        )

    if idx == 31:
        parts.append(
            "- Кандидат должен назвать конкретный город, который явно далеко от города вакансии."
        )
        parts.append(
            "- Кандидат в этой же реплике уже отвечает и про зарплату: укажи конкретную сумму в рублях на руки, не выше верхней границы вилки вакансии."
        )
        parts.append(
            "- Никаких дополнительных пояснений не добавляй: только текущий город кандидата и зарплатные ожидания."
        )

    if idx == 46:
        parts.append(
            "- Кандидат должен задать обычный деловой вопрос только про формат работы по вакансии."
        )
        parts.append(
            "- Не добавляй агрессию, грубость, мат, сарказм, претензии или ультиматумы."
        )
        parts.append(
            "- Не смешивай вопрос про формат работы с вопросами про зарплату, компанию или созвон."
        )

    if idx == 16 or "оффер" in name or "трудоустроен" in name:
        parts.append(
            "- Это именно триггер про уже принятый оффер или текущее трудоустройство: используй формулировки «уже принял оффер», «уже трудоустроен», «контракт подписан», «выхожу на новую работу»."
        )
        parts.append(
            "- Не превращай реплику в обычный follow-up по процессу и не задавай уточняющих вопросов."
        )

    # 10. Другой профиль
    if idx == 10 or ("другой профиль" in name or "другая специализация" in name):
        parts.append(
            "- Кандидат должен явно показать, что вакансия не по его профилю: «это не мой профиль», «у меня другая специализация», «это направление не по моему профилю»."
        )
        parts.append(
            "- Не делай акцент на отсутствии опыта: не используй формулировки «нет опыта» и «нет релевантного опыта» как основной смысл."
        )

    # 18. Нет релевантного опыта
    if idx == 18 or "нет нужного опыта" in name or "отсутствие необходимого" in name:
        parts.append(
            "- Кандидат должен явно показать, что у него нет релевантного опыта в этой сфере: «у меня нет релевантного опыта», «у меня нет опыта в этой сфере», «это другая специальность», «таким не занимаюсь».\n"
            "- Не своди реплику к отсутствию одной узкой технологии и не добавляй «но быстро разберусь»."
        )

    # 19/20. Бот
    if idx == 19:
        parts.append("- Это первый вопрос про бота: «ты бот?», «это ИИ или человек?».")
        parts.append("- Не используй слова «повторно», «снова», «я уже спрашивал», «в третий раз».")
    if idx == 20 or ("повторно" in name and "бот" in name):
        parts.append("- Это повторный вопрос про бота: «я же уже спрашивал, вы бот?».")
    if not parts:
        return ""

    return "Дополнительные требования для этого конкретного сценария:\n" + "\n".join(parts)






def _parse_json_string_list(text: str) -> List[str]:
    try:
        data = _safe_json_loads(text)
        if isinstance(data, list):
            out = []
            for x in data:
                sx = str(x).strip()
                if sx:
                    out.append(sx)
            return out
    except Exception:
        return []
    return []


def generate_candidate_messages_for_scenario(
    client: OpenAI,
    scenario: Scenario,
    messages_per_scenario: int,
    usage_bucket: Dict[str, int],
    dialog_context_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Генерация максимально близкая к оригиналу:
    - примеры (если есть)
    - extra-guidelines
    - sequential, если сценарий повторный
    Плюс: trigger forcing + 1 перегенерация + fallback.
    """
    dialog_context_meta = dialog_context_meta or {}
    if scenario.index in FORCED_FALLBACK_SCENARIOS:
        return _fallback_messages(
            s=scenario,
            n=messages_per_scenario,
            dialog_context_meta=dialog_context_meta,
        )
    examples_override = _scenario_example_override(scenario)
    if examples_override is None:
        examples = extract_candidate_examples(scenario.examples_raw, max_examples=10)
    else:
        examples = examples_override[:10]
    extra = _extra_generation_guidelines(scenario)
    is_repeated = _is_repeated_dialog_scenario(scenario)
    trigger_req = _trigger_requirement_text(scenario)
    min_salary = _parse_int_value(dialog_context_meta.get("min_salary"))
    max_salary = _parse_int_value(dialog_context_meta.get("max_salary"))
    expected_location = str(dialog_context_meta.get("location") or "").strip()
    work_format = str(dialog_context_meta.get("work_format") or "").strip()

    runtime_context_lines: List[str] = []
    if scenario.index == 30:
        runtime_context_lines.append(
            "- ВАЖНО: реплика должна быть только про смерть / утрату. Не упоминай зарплату, город, формат работы и не смешивай другие триггеры."
        )
    if scenario.index == 24 and min_salary is not None:
        runtime_context_lines.append(
            f"- Нижняя граница вилки вакансии: {_format_int_with_spaces(min_salary)} рублей."
        )
        runtime_context_lines.append(
            f"- ВАЖНО: в каждой реплике назови ожидание СТРОГО НИЖЕ {_format_int_with_spaces(min_salary)}."
        )
    if scenario.index in (31, 33):
        if expected_location:
            runtime_context_lines.append(
                f"- Локация вакансии для этого прогона: {expected_location}."
            )
        if work_format:
            runtime_context_lines.append(
                f"- Формат работы для этого прогона: {_work_format_label(work_format)}."
            )
    if scenario.index == 31:
        target_salary = _safe_salary_expectation_value(dialog_context_meta)
        runtime_context_lines.append(
            f"- ВАЖНО: кандидат в этой же реплике уже отвечает и про зарплату: назови около {_format_int_with_spaces(target_salary)} рублей на руки, не выше верхней границы вилки."
        )
        runtime_context_lines.append(
            "- DO NOT confirm readiness for office/hybrid, commuting to the vacancy city, or relocation."
        )
    if scenario.index == 27:
        nearby_names = _nearby_city_names("Москва")
        runtime_context_lines.append(
            "- ВАЖНО: кандидат должен назвать конкретный город рядом с Москвой, но не Москву."
        )
        if nearby_names:
            runtime_context_lines.append(
                f"- Используй один из конкретных городов: {', '.join(nearby_names)}. Не указывай километры."
            )
    if scenario.index == 31 and expected_location:
        far_names = _far_city_names(expected_location)
        runtime_context_lines.append(
            f"- ВАЖНО: кандидат должен назвать конкретный город, который явно далеко от {expected_location}."
        )
        if far_names:
            runtime_context_lines.append(
                f"- Используй один из конкретных городов: {', '.join(far_names)}."
            )
        runtime_context_lines.append(
            "- Keep the reply minimal: only the current city and salary expectations."
        )
    if scenario.index == 29 and work_format:
        runtime_context_lines.append(
            f"- ВАЖНО: кандидат должен явно отказаться работать в формате {_work_format_label(work_format)} и отказаться от переезда."
        )
    runtime_context_text = "\n".join(runtime_context_lines).strip()

    def _build_prompt(strong: bool) -> str:
        if not examples:
            base_prompt = (
                "Ты симулируешь сообщения кандидата в ответ на рекрутера.\n"
                "Дан сценарий поведения КАНДИДАТА.\n"
                "Сгенерируй {n} коротких реплик именно кандидата, полностью соответствующих сценарию.\n"
                "Сохраняй грубость/эмоции, если они подразумеваются, но не придумывай политические лозунги сверх описания.\n"
                "Очень важно: НЕ пиши от лица рекрутера, не говори 'я рекрутер', 'я провожу скрининг',\n"
                "не задавай вопросы кандидату от имени компании и не упоминай процессы найма.\n"
                "Пиши только естественные ответы кандидата.\n"
                "НЕ добавляй никакие служебные отметки типа END.\n"
                "Ответ верни строго в формате JSON-массива строк, без лишнего текста.\n"
            )
        else:
            base_prompt = (
                "Ты симулируешь сообщения КАНДИДАТА в диалоге с рекрутером.\n"
                "У тебя есть описание сценария и реальные примеры реплик кандидата.\n"
                "Твоя задача - сгенерировать {n} НОВЫХ реплик кандидата, которые:\n"
                "- максимально похожи по тону, эмоциональности и лексике на примеры;\n"
                "- используют тот же язык, что примеры;\n"
                "- могут чуть перефразировать или переставлять слова, но НЕ превращаться в вежливый нейтральный текст;\n"
                "- НЕ содержат 'END' и любые тех.метки;\n"
                "- НЕ звучат как речь рекрутера и не объясняют процессы найма.\n"
                "Верни ответ строго в виде JSON-массива строк, без пояснений.\n"
            )

        base_prompt += (
            "\nНе добавляй грубость, мат, сарказм, резкие претензии или лишнюю негативность, если сценарий не требует именно агрессии или оскорблений.\n"
            "Если сценарий нейтральный или информационный, кандидат должен звучать естественно, по-деловому и без лишней конфликтности.\n"
        )

        if is_repeated:
            base_prompt += (
                "\n\nСчитай, что эти {n} реплик - это сообщения одного и того же кандидата в ОДНОМ диалоге, "
                "которые идут по хронологии: сначала первое сообщение, затем повторные по той же теме."
            )

        if trigger_req:
            base_prompt += "\n\nТРЕБОВАНИЕ ТРИГГЕРА:\n" + trigger_req

        if extra:
            base_prompt += "\n\n" + extra

        if runtime_context_text:
            base_prompt += "\n\nДОП.КОНТЕКСТ ВАКАНСИИ:\n" + runtime_context_text

        if strong:
            base_prompt += (
                "\n\nSTRONG REQUIREMENTS:\n"
                "- Каждая реплика обязана содержать триггер из блока 'ТРЕБОВАНИЕ ТРИГГЕРА'.\n"
                "- Не делай нейтральных уточнений. Триггер должен быть очевидным.\n"
                "- Если сомневаешься - усили триггер (но без оскорблений по национальности).\n"
            )

        base_prompt += "\n\nОтвет строго JSON-массив строк. Без markdown."

        return base_prompt

    payload_obj: Dict[str, Any] = {
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
    }
    if examples:
        payload_obj["candidate_examples"] = examples
    if scenario.index in (31, 32, 33, 34, 35, 37):
        payload_obj["vacancy_context_for_generation"] = {
            "location": expected_location,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "work_format": work_format,
        }

    def _do_gen(strong: bool) -> List[str]:
        prompt = _build_prompt(strong=strong).format(n=messages_per_scenario)
        payload = prompt + "\n\n" + json.dumps(payload_obj, ensure_ascii=False)

        resp = client.responses.create(model=GEN_MODEL, input=payload)
        _accumulate_usage(usage_bucket, getattr(resp, "usage", None))
        text = (getattr(resp, "output_text", "") or "").strip()

        msgs = _parse_json_string_list(text)
        cleaned = [_normalize_text(m) for m in msgs[:messages_per_scenario]]
        return cleaned

    messages = [
        m for m in _do_gen(strong=True)
        if _generated_message_matches_scenario_constraints(scenario.index, m)
    ]

    if len(messages) < messages_per_scenario:
        fallback = _fallback_messages(
            scenario,
            messages_per_scenario,
            dialog_context_meta=dialog_context_meta,
        )
        messages = messages[:messages_per_scenario]
        if len(messages) < messages_per_scenario:
            messages.extend(fallback[len(messages):messages_per_scenario])

    out: List[str] = []
    for m in messages[:messages_per_scenario]:
        out.append(_normalize_text(m.replace("END", "")))
    return out


# -----------------------
# Simple клиент (single сценарии - независимые одноступенчатые тесты)
# -----------------------


class SimpleScreeningAssistant:
    def __init__(
        self,
        prompt_id: str,
        prompt_version: str | int | None,
        dialog_context: str,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version is not None:
            self.prompt["version"] = str(prompt_version)
        self.dialog_context = (dialog_context or "").strip()
        self.last_usage: Any = None

    def _scenario_context_block(self, scenario: Scenario) -> str:
        # Раньше тут был ранний return, из-за чего остальной код был мертвым.
        # Сейчас оставляем единственный источник контекста - dialog_context.
        # Если захочешь добавить отдельные условия, делай их ДО return.
        return self.dialog_context

    def reply_one_turn(self, candidate_message: str) -> str:
        scenario_block = self._scenario_context_block(Scenario(0, "", "", "", ""))  # не используем, но оставим структуру
        scenario_block = self.dialog_context

        payload_lines = [
            "Контекст: ты выступаешь как IT-рекрутер в первичном скрининге кандидата.",
            "Соблюдай все правила системного промпта screening_assistant_hh,",
            "особенно по KO-правилам, триггерам и маркеру END.",
            "",
        ]

        if scenario_block:
            payload_lines.append(scenario_block)
            payload_lines.append("")

        payload_lines.extend(
            [
                "Ниже одно сообщение кандидата. Не придумывай историю диалога.",
                "Ответь только одним сообщением рекрутера.",
                "",
                "Сообщение кандидата:",
                candidate_message,
            ]
        )

        payload = "\n".join(payload_lines)

        response = self.client.responses.create(
            prompt=self.prompt,
            input=payload,
        )
        self.last_usage = getattr(response, "usage", None)
        return (getattr(response, "output_text", "") or "").strip()

    def start_dialogue(self) -> str:
        payload_lines = [
            "РљРѕРЅС‚РµРєСЃС‚: С‚С‹ РІС‹СЃС‚СѓРїР°РµС€СЊ РєР°Рє IT-СЂРµРєСЂСѓС‚РµСЂ РІ РїРµСЂРІРёС‡РЅРѕРј СЃРєСЂРёРЅРёРЅРіРµ РєР°РЅРґРёРґР°С‚Р°.",
            "РЎРѕР±Р»СЋРґР°Р№ РІСЃРµ РїСЂР°РІРёР»Р° СЃРёСЃС‚РµРјРЅРѕРіРѕ РїСЂРѕРјРїС‚Р° screening_assistant_hh,",
            "РѕСЃРѕР±РµРЅРЅРѕ РїРѕ KO-РїСЂР°РІРёР»Р°Рј, С‚СЂРёРіРіРµСЂР°Рј Рё РјР°СЂРєРµСЂСѓ END.",
            "",
        ]
        if self.dialog_context:
            payload_lines.append(self.dialog_context)
            payload_lines.append("")
        payload_lines.extend(
            [
                "Р”РёР°Р»РѕРі РµС‰Рµ РЅРµ РЅР°С‡Р°Р»СЃСЏ. РЎС„РѕСЂРјСѓР»РёСЂСѓР№ СЃР°РјРѕРµ РїРµСЂРІРѕРµ РёСЃС…РѕРґСЏС‰РµРµ СЃРѕРѕР±С‰РµРЅРёРµ СЂРµРєСЂСѓС‚РµСЂР° РєР°РЅРґРёРґР°С‚Сѓ.",
                "РћС‚РІРµС‚СЊ С‚РѕР»СЊРєРѕ РѕРґРЅРёРј СЃРѕРѕР±С‰РµРЅРёРµРј СЂРµРєСЂСѓС‚РµСЂР°.",
            ]
        )
        payload = "\n".join(payload_lines)
        response = self.client.responses.create(
            prompt=self.prompt,
            input=payload,
        )
        self.last_usage = getattr(response, "usage", None)
        return (getattr(response, "output_text", "") or "").strip()


# -----------------------
# Conversation клиент (chain сценарии - один диалог на цепочку)
# -----------------------


class ConversationScreeningAssistant:
    def __init__(
        self,
        prompt_id: str,
        prompt_version: str | int | None,
        dialog_context: str,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version is not None:
            self.prompt["version"] = str(prompt_version)
        self.dialog_context = (dialog_context or "").strip()
        self.last_usage: Any = None

    def _scenario_context_block(self, scenarios: List[Scenario]) -> str:
        return self.dialog_context

    def create_conversation(self) -> str:
        ctx = self.dialog_context

        initial = "\n".join(
            [
                "Контекст: ты IT-рекрутер, проводишь первичный скрининг.",
                "Соблюдай правила промпта screening_assistant_hh, особенно триггеры и END.",
                "",
                ctx.strip(),
                "",
                "Важно: это один диалог, учитывай историю переписки.",
            ]
        ).strip()

        conv = self.client.conversations.create(
            items=[{"type": "message", "role": "assistant", "content": initial}]
        )
        return conv.id

    def reply_in_conversation(self, conversation_id: str, candidate_message: str) -> str:
        response = self.client.responses.create(
            prompt=self.prompt,
            conversation=conversation_id,
            input=candidate_message,
        )
        self.last_usage = getattr(response, "usage", None)
        return (getattr(response, "output_text", "") or "").strip()


def create_simple_assistant(
    cfg: Dict[str, Any],
    dialog_context: str,
) -> SimpleScreeningAssistant:
    sa_cfg = _component_cfg(cfg, HH_COMPONENT_NAME)
    prompt_id = os.environ.get("SCREENING_ASSISTANT_HH_PROMPT_ID") or sa_cfg.get("prompt_id")
    prompt_version = os.environ.get("SCREENING_ASSISTANT_HH_PROMPT_VERSION") or sa_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError(
            "screening_assistant_hh.prompt_id is not set in model.yaml and SCREENING_ASSISTANT_HH_PROMPT_ID is not set"
        )
    return SimpleScreeningAssistant(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        dialog_context=dialog_context,
    )


def create_conversation_assistant(
    cfg: Dict[str, Any],
    dialog_context: str,
) -> ConversationScreeningAssistant:
    sa_cfg = _component_cfg(cfg, HH_COMPONENT_NAME)
    prompt_id = os.environ.get("SCREENING_ASSISTANT_HH_PROMPT_ID") or sa_cfg.get("prompt_id")
    prompt_version = os.environ.get("SCREENING_ASSISTANT_HH_PROMPT_VERSION") or sa_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError(
            "screening_assistant_hh.prompt_id is not set in model.yaml and SCREENING_ASSISTANT_HH_PROMPT_ID is not set"
        )
    return ConversationScreeningAssistant(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        dialog_context=dialog_context,
    )


# -----------------------
# Оценка соответствия
# -----------------------


def _case_evaluation_context(dialog_context_meta: Dict[str, Any]) -> Dict[str, Any]:
    ctx = {
        "title": str(dialog_context_meta.get("title") or "").strip(),
        "company_name": str(dialog_context_meta.get("company_name") or "").strip(),
        "location": str(dialog_context_meta.get("location") or "").strip(),
        "work_format": str(dialog_context_meta.get("work_format") or "").strip(),
        "work_format_ids": list(dialog_context_meta.get("work_format_ids") or []),
        "vacancy_description": str(dialog_context_meta.get("vacancy_description") or "").strip(),
        "vacancy_url": str(dialog_context_meta.get("vacancy_url") or "").strip(),
        "min_salary": dialog_context_meta.get("min_salary"),
        "max_salary": dialog_context_meta.get("max_salary"),
        "salary": str(dialog_context_meta.get("salary") or "").strip(),
        "questions": str(dialog_context_meta.get("questions") or "").strip(),
    }
    recruiting_agency_mode = str(dialog_context_meta.get("recruiting_agency_mode") or "").strip()
    if recruiting_agency_mode:
        ctx["recruiting_agency_mode"] = recruiting_agency_mode
    known_employer_name = str(dialog_context_meta.get("known_employer_name") or "").strip()
    if known_employer_name:
        ctx["known_employer_name"] = known_employer_name
    if dialog_context_meta.get("initial_outreach"):
        ctx["initial_outreach"] = True
    return ctx


def _safe_eval_score(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 0
    return 1 if parsed == 1 else 0


def _evaluate_single_case(
    client: OpenAI,
    scenario: Scenario,
    turns: List[Dict[str, Any]],
    dialog_context_meta: Dict[str, Any],
    usage_bucket: Dict[str, int],
) -> Dict[int, Tuple[int, str]]:
    instruction = (
        "You are a QA reviewer for a recruiting chat assistant.\n"
        "Evaluate each step semantically against expected_behavior.\n"
        "Be strict on behavior, but not brittle on wording.\n"
        "Equivalent wording counts as pass.\n"
        "Do not fail only because wording is different, clauses are reordered, or an otherwise correct fixed script is wrapped in quotation marks.\n"
        "Each step is an independent one-turn case. Do not carry history between steps.\n"
        "Step numbers do not imply first/second/third message in one conversation. Never infer repetition from step order in a single case.\n"
        "If a turn contains dialog_text, use only that provided dialog history for the current step.\n"
        "If expected_behavior mentions first time versus repeated behavior, use only the content of the current step. Without explicit dialog history, treat the step as an independent sample.\n"
        "For unintelligible-message scenarios in a single case, a valid pass is a clarification like 'не совсем поняла вас' followed by the next appropriate screening question. Do not require ending the dialogue unless the current candidate message itself explicitly indicates repetition.\n"
        "For strange-but-non-aggressive single-turn behavior, both a polite continuation and a business-equivalent apology with END count as valid if the assistant does not escalate or mis-handle the case.\n"
        "Natural questions about readiness for office, hybrid, remote, field work, commuting, relocation, or working from the required location count as valid if the meaning matches.\n"
        "REMOTE does not cancel location restrictions. If vacancy_context has a non-empty location or an explicit geo restriction in vacancy_description, a location-based refusal may still be valid.\n"
        "For location-refusal scenarios, accept any semantically correct refusal with END; do not require the assistant to cite the exact same wording or the most specific geographic label unless expected_behavior explicitly demands it.\n"
        "If vacancy_context.recruiting_agency_mode is present, do not require the assistant to present a recruiting agency as the employer. A valid pass may neutrally say that the search is led by a recruiting agency. If known_employer_name is empty, fail invented employer names, invented company names, or invented sites/links.\n"
        "If vacancy_context.initial_outreach is true and candidate_message is empty, judge the assistant message as the first outbound recruiter message at the start of the dialogue rather than as a reply to a candidate turn.\n"
        "For profile-reference cases, pass if the assistant clearly asks the candidate to answer directly in the chat and re-asks the current unresolved question. Phrases like 'прямо в этом чате', 'прямо в диалоге', 'прямо здесь', 'расскажите об этом здесь', or 'коротко опишите это в чате' count as valid direct-answer requests. Fail only if it jumps to a different question.\n"
        "Ignore awkwardness of synthetic candidate phrasing and judge only the assistant behavior.\n"
        "Return JSON only in this format:\n"
        "{\n"
        '  "turns": [\n'
        '    {"step": 1, "score": 0 or 1, "comment": "short reason"}\n'
        "  ]\n"
        "}\n"
    )
    payload_obj = {
        "case_type": "single",
        "vacancy_context": _case_evaluation_context(dialog_context_meta),
        "turns": [
            {
                "step": int(turn.get("step") or 0),
                "scenario_index": scenario.index,
                "scenario_name": scenario.name,
                "scenario_description": scenario.description,
                "expected_behavior": scenario.expected_behavior,
                "candidate_message": str(turn.get("candidate_message") or ""),
                "assistant_reply": str(turn.get("assistant_reply") or ""),
                "dialog_text": str(turn.get("dialog_text") or ""),
            }
            for turn in turns
        ],
    }
    payload = instruction + "\n\n" + json.dumps(payload_obj, ensure_ascii=False)
    response = client.responses.create(model=EVAL_MODEL, input=payload)
    _accumulate_usage(usage_bucket, getattr(response, "usage", None))
    text = (getattr(response, "output_text", "") or "").strip()

    default_results = {
        int(turn.get("step") or 0): (0, "Case judge missing result for this step.")
        for turn in turns
    }
    try:
        data = _safe_json_loads(text)
        for item in data.get("turns") or []:
            step = int(item.get("step") or 0)
            if step not in default_results:
                continue
            score = _safe_eval_score(item.get("score"))
            comment = str(item.get("comment") or "").strip() or "No comment."
            default_results[step] = (score, comment)
    except Exception:
        for step in list(default_results):
            default_results[step] = (0, f"Failed to parse case eval output: {text[:200]}")
    return default_results


def _evaluate_chain_case(
    client: OpenAI,
    group: ScenarioGroup,
    runs: List[Dict[str, Any]],
    dialog_context_meta: Dict[str, Any],
    usage_bucket: Dict[str, int],
) -> Dict[Tuple[int, int], Tuple[int, str]]:
    instruction = (
        "You are a QA reviewer for a recruiting chat assistant.\n"
        "Evaluate each run as one full conversation and score each step semantically against expected_behavior.\n"
        "Be strict on behavior, but not brittle on wording.\n"
        "Equivalent wording counts as pass.\n"
        "Do not fail only because wording is different, clauses are reordered, or an otherwise correct fixed script is wrapped in quotation marks.\n"
        "Use dialog_text and step order inside each run. Different runs are independent.\n"
        "Natural questions about readiness for office, hybrid, remote, field work, commuting, relocation, or working from the required location count as valid if the meaning matches.\n"
        "REMOTE does not cancel location restrictions. If vacancy_context has a non-empty location or an explicit geo restriction in vacancy_description, a location-based refusal may still be valid.\n"
        "For location-refusal scenarios, accept any semantically correct refusal with END; do not require the assistant to cite the exact same wording or the most specific geographic label unless expected_behavior explicitly demands it.\n"
        "For pause/resume cases, a resume phrase is not an answer. Pass only if the assistant returns to the last unresolved question.\n"
        "For pause-after-priority cases, a plain pause-script without the next [questions] follow-up is also acceptable; do not fail only because the assistant stops after the pause-script.\n"
        "For profile-reference cases, pass if the assistant clearly asks the candidate to answer directly in the chat and re-asks the current unresolved question. Phrases like 'прямо в этом чате', 'прямо в диалоге', 'прямо здесь', 'расскажите об этом здесь', or 'коротко опишите это в чате' count as valid direct-answer requests. Fail only if it jumps to a different question.\n"
        "Ignore awkwardness of synthetic candidate phrasing and judge only the assistant behavior.\n"
        "Return JSON only in this format:\n"
        "{\n"
        '  "runs": [\n'
        '    {\n'
        '      "run_index": 1,\n'
        '      "turns": [\n'
        '        {"step": 1, "score": 0 or 1, "comment": "short reason"}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    payload_obj = {
        "case_type": "chain",
        "group_id": group.group_id,
        "vacancy_context": _case_evaluation_context(dialog_context_meta),
        "runs": [
            {
                "run_index": int(run.get("run_index") or 0),
                "dialog_text": str(run.get("dialog_text") or ""),
                "turns": [
                    {
                        "step": int(turn.get("step") or 0),
                        "scenario_index": int(turn.get("scenario_index") or 0),
                        "scenario_name": str(turn.get("scenario_name") or ""),
                        "scenario_description": next(
                            (scenario.description for scenario in group.scenarios if scenario.index == int(turn.get("scenario_index") or 0)),
                            "",
                        ),
                        "expected_behavior": next(
                            (scenario.expected_behavior for scenario in group.scenarios if scenario.index == int(turn.get("scenario_index") or 0)),
                            "",
                        ),
                        "candidate_message": str(turn.get("candidate_message") or ""),
                        "assistant_reply": str(turn.get("assistant_reply") or ""),
                    }
                    for turn in run.get("turns") or []
                ],
            }
            for run in runs
        ],
    }
    payload = instruction + "\n\n" + json.dumps(payload_obj, ensure_ascii=False)
    response = client.responses.create(model=EVAL_MODEL, input=payload)
    _accumulate_usage(usage_bucket, getattr(response, "usage", None))
    text = (getattr(response, "output_text", "") or "").strip()

    default_results: Dict[Tuple[int, int], Tuple[int, str]] = {}
    for run in runs:
        run_index = int(run.get("run_index") or 0)
        for turn in run.get("turns") or []:
            step = int(turn.get("step") or 0)
            default_results[(run_index, step)] = (0, "Case judge missing result for this step.")

    try:
        data = _safe_json_loads(text)
        for run_item in data.get("runs") or []:
            run_index = int(run_item.get("run_index") or 0)
            for turn_item in run_item.get("turns") or []:
                step = int(turn_item.get("step") or 0)
                key = (run_index, step)
                if key not in default_results:
                    continue
                score = _safe_eval_score(turn_item.get("score"))
                comment = str(turn_item.get("comment") or "").strip() or "No comment."
                default_results[key] = (score, comment)
    except Exception:
        for key in list(default_results):
            default_results[key] = (0, f"Failed to parse case eval output: {text[:200]}")
    return default_results


def _reply_has_end_marker(text: str) -> bool:
    return bool(re.search(r"(?<!\w)end(?!\w)", text or "", flags=re.IGNORECASE))


def _contains_any_substring(text: str, parts: List[str]) -> bool:
    low = (text or "").lower()
    return any(part in low for part in parts)


def _assistant_reply_has_context_leak(reply: str) -> bool:
    low = _normalize_text(reply).lower()
    return any(marker in low for marker in CONTEXT_LEAK_MARKERS)


def _assistant_reply_has_raw_work_format_ids(reply: str) -> bool:
    return bool(re.search(r"\b(?:ON_SITE|REMOTE|HYBRID|FIELD_WORK)\b", reply or "", flags=re.IGNORECASE))


def _message_has_unwanted_rudeness(message: str, scenario_index: int) -> bool:
    if scenario_index in RUDE_TONE_ALLOWED_SCENARIOS:
        return False
    low = _normalize_text(message).lower()
    return any(marker in low for marker in UNWANTED_RUDENESS_MARKERS)


def _question_mark_count(text: str) -> int:
    return (text or "").count("?")


def _reply_contains_url_or_domain(text: str) -> bool:
    raw = str(text or "")
    if re.search(r"https?://|www\.", raw, flags=re.IGNORECASE):
        return True
    return bool(re.search(r"\b(?:[a-z0-9-]+\.)+(?:ru|com|net|org|io|ai|dev|co|biz)\b", raw, flags=re.IGNORECASE))


def _reply_presents_agency_as_company_or_employer(reply: str) -> bool:
    low = _normalized_low(reply)
    agency = re.escape(RECRUITING_AGENCY_NAME)
    patterns = [
        rf"\bработодатель\b[^.:\n]{{0,40}}{agency}\b",
        rf"\bкомпан(?:ия|ии)\b[^.:\n]{{0,40}}{agency}\b",
        rf"\bв компании\s+{agency}\b",
        rf"\bпозици(?:я|и)\s+в\s+{agency}\b",
        rf"\bваканси(?:я|и)\s+в\s+{agency}\b",
    ]
    return any(re.search(pattern, low) for pattern in patterns)


def _reply_has_salary_question(reply: str) -> bool:
    low = _normalized_low(reply)
    return "?" in (reply or "") and _contains_any_substring(
        low,
        [
            "зарплат",
            "ожидани",
            "на какую сумму",
            "какую сумму",
            "финансов",
            "доход",
        ],
    )


def _reply_has_location_question(reply: str) -> bool:
    low = _normalized_low(reply)
    return "?" in (reply or "") and _contains_any_substring(
        low,
        [
            "в каком городе",
            "каком городе",
            "где находит",
            "ваш город",
            "из какого города",
            "локац",
        ],
    )


def _reply_has_company_answer(reply: str, known_employer_name: str = "") -> bool:
    low = _normalized_low(reply)
    if known_employer_name and _normalized_low(known_employer_name) in low:
        return True
    return _contains_any_substring(low, ["компан", "работодател", "поиск по позиции", "рекрутингов", "работаем по позиции"])


def _reply_has_greeting(reply: str) -> bool:
    low = _normalized_low(reply)
    return _contains_any_substring(low, ["здравствуйте", "добрый день", "алексей, здравствуйте", "алексей, добрый день"])


def _assistant_reply_has_context_leak(reply: str) -> bool:
    low = _normalize_text(reply).lower()
    for marker in CONTEXT_LEAK_MARKERS:
        if marker == "[questions]":
            if marker in low:
                return True
            continue
        if re.search(rf"(?:^|\n)\s*{re.escape(marker)}", low):
            return True
    return False


def _reply_presents_agency_as_company_or_employer(reply: str) -> bool:
    low = _normalized_low(reply)
    agency = re.escape(RECRUITING_AGENCY_NAME)
    patterns = [
        rf"\bработодатель\b\s*(?:-|—|:)?\s*{agency}\b",
        rf"\bкомпан(?:ия|ии)\b\s*(?:-|—|:)?\s*{agency}\b",
        rf"\bв компании\s+{agency}\b",
        rf"\bпозици(?:я|и)\s+в\s+{agency}\b",
        rf"\bваканси(?:я|и)\s+в\s+{agency}\b",
    ]
    return any(re.search(pattern, low) for pattern in patterns)


def _is_apology_end_reply(reply: str) -> bool:
    low = _normalized_low(reply)
    return _reply_has_end_marker(reply) and _contains_any_substring(
        low,
        [
            "прошу прощения",
            "извините за беспокойство",
            "извиняюсь за беспокойство",
        ],
    )


def _looks_like_direct_answer_request(reply: str) -> bool:
    low = _normalized_low(reply)
    direct_markers = [
        "прямо в этом чате",
        "прямо в диалоге",
        "прямо здесь",
        "об этом здесь",
        "об этом в чате",
        "расскажите об этом здесь",
        "расскажите об этом в чате",
        "коротко расскажите об этом здесь",
        "коротко расскажите об этом в чате",
        "коротко рассказали об этом здесь",
        "коротко рассказали об этом в чате",
        "опишите это в чате",
        "опишите это здесь",
        "ответить прямо",
        "для корректной фиксации",
        "для корректной фиксации данных",
        "для корректной фиксации в базе",
        "важно получить ответ",
        "важно получить ваш ответ",
        "важно зафиксировать",
        "прошу коротко подтвердить",
        "кратко подтвердить",
        "прошу уточнить",
        "кратко описать",
        "кратко рассказать",
    ]
    return _contains_any_substring(low, direct_markers)


def _is_monthly_rubles_clarification_reply(reply: str) -> bool:
    low = (reply or "").lower()
    clarify_markers = [
        "уточните",
        "укажите",
        "подскажите",
        "корректно ли я понимаю",
        "правильно ли я понимаю",
        "имеете в виду",
        "речь о сумме",
        "речь идет о сумме",
        "речь идёт о сумме",
        "речь идет о зарплате",
        "речь идёт о зарплате",
    ]
    monthly_markers = [
        "в рублях",
        "на руки",
        "в месяц",
        "за месяц",
    ]
    return (
        not _reply_has_end_marker(reply)
        and "?" in (reply or "")
        and _contains_any_substring(low, clarify_markers)
        and _contains_any_substring(low, monthly_markers)
    )


def _is_salary_budget_rejection_reply(reply: str) -> bool:
    low = _normalized_low(reply)
    return _reply_has_end_marker(reply) and _contains_any_substring(
        low,
        [
            "бюджет",
            "не позволяет",
            "не сможем рассмотреть",
            "к сожалению",
            "удачи в дальнейших поисках",
            "удачи в поисках",
        ],
    )


def _is_location_refusal_reply(reply: str) -> bool:
    low = _normalized_low(reply)
    return _reply_has_end_marker(reply) and _contains_any_substring(
        low,
        [
            "локац",
            "город",
            "находит",
            "важно находиться",
            "только из рф",
            "только из россии",
            "вне рф",
            "за границей",
            "москва",
            "росси",
        ],
    ) and _contains_any_substring(
        low,
        [
            "к сожалению",
            "не сможем",
            "не получится",
            "не можем продолжить",
            "не готовы рассматривать",
            "спасибо",
        ],
    )


def _is_bot_first_reply(reply: str) -> bool:
    low = _normalized_low(reply)
    return (
        not _reply_has_end_marker(reply)
        and _contains_any_substring(low, ["внешн", "рекрутер", "скрининг"])
        and ("?" in (reply or "") or _contains_any_substring(low, ["зарплат", "город", "локац"]))
    )


def enforce_prompt_v2_hard_rules(
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if not str(assistant_reply or "").strip():
        return 0, "Prompt v2 hard rule failed: assistant reply is empty."
    if _question_mark_count(assistant_reply) > 2:
        return 0, "Prompt v2 hard rule failed: assistant reply contains more than two question marks."
    if _reply_has_end_marker(assistant_reply) and "?" in (assistant_reply or ""):
        return 0, "Prompt v2 hard rule failed: END cannot be combined with questions."
    if _assistant_reply_has_raw_work_format_ids(assistant_reply):
        return 0, "Prompt v2 hard rule failed: assistant leaked raw work format ids."
    if _assistant_reply_has_context_leak(assistant_reply):
        return 0, "Prompt v2 hard rule failed: assistant leaked prompt context labels."

    return score, comment


def enforce_gross_salary_clarification_rule(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if scenario.index not in HH_GROSS_SALARY_CLARIFICATION_SCENARIOS:
        return score, comment

    if _is_salary_budget_rejection_reply(assistant_reply):
        return 1, "Gross salary rule passed: assistant correctly rejects because the inferred expectations are above budget."
    if not _is_monthly_rubles_clarification_reply(assistant_reply):
        return 0, "Gross salary rule failed: assistant must either clarify the final monthly net salary in rubles or reject because expectations are above budget."
    return 1, "Gross salary rule passed: assistant asks to clarify the final monthly net salary in rubles."


def enforce_salary_normalization_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    idx = scenario.index
    if idx not in SALARY_NORMALIZATION_SCENARIOS:
        return score, comment

    if idx == 35:
        if _is_salary_budget_rejection_reply(assistant_reply):
            return 1, "Salary normalization rule passed: explicit monthly salary above the budget triggers a budget rejection."
        return 0, "Salary normalization rule failed: scenario 35 must reject because the explicit monthly salary is above the budget."

    if idx in (36, 37, 38, 39):
        if _reply_has_end_marker(assistant_reply):
            return 0, "Salary normalization rule failed: ambiguous salary input must not end the dialogue."
        if not _is_monthly_rubles_clarification_reply(assistant_reply):
            return 0, "Salary normalization rule failed: assistant must clarify monthly net salary in rubles."
        return 1, "Salary normalization rule passed: assistant asks to clarify monthly net salary in rubles."

    if _reply_has_end_marker(assistant_reply):
        return 0, "Salary normalization rule failed: explicit monthly salary within range must not end the dialogue."
    if _is_monthly_rubles_clarification_reply(assistant_reply):
        return 0, "Salary normalization rule failed: explicit monthly salary in rubles per month must not trigger unnecessary clarification."
    return 1, "Salary normalization rule passed: assistant accepts explicit monthly salary without unnecessary clarification."


def enforce_recruiting_agency_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    if scenario.index not in HH_RECRUITING_AGENCY_SCENARIOS:
        return score, comment

    reply = str(assistant_reply or "")
    expected_url = str(dialog_context_meta.get("vacancy_url") or "").strip()

    if _reply_has_end_marker(reply):
        return 0, f"Scenario {scenario.index} strict check failed: assistant must not end the dialogue."
    if _reply_presents_agency_as_company_or_employer(reply):
        return 0, f"Scenario {scenario.index} strict check failed: recruiting agency must not be presented as the employer or company."
    if not expected_url and _reply_contains_url_or_domain(reply):
        return 0, f"Scenario {scenario.index} strict check failed: assistant must not invent a link, site, or domain."

    if scenario.index == 50:
        if not _reply_has_company_answer(reply):
            return 0, "Scenario 50 strict check failed: assistant should briefly answer the company question."
        if not (_reply_has_salary_question(reply) or _reply_has_location_question(reply)):
            return 0, "Scenario 50 strict check failed: assistant should continue screening with the next relevant question."
        return 1, "Scenario 50 strict check passed: company handling is neutral and the screening continues."

    return score, comment


def enforce_bot_identity_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if scenario.index == 19:
        if _is_bot_first_reply(assistant_reply):
            return 1, "Scenario 19 strict check passed: assistant explains the role and continues the screening."
        return 0, "Scenario 19 strict check failed: assistant should briefly explain that this is a recruiter and continue the screening."

    if scenario.index == 20:
        if _is_apology_end_reply(assistant_reply):
            return 1, "Scenario 20 strict check passed: repeated bot-check ends the dialogue."
        return 0, "Scenario 20 strict check failed: repeated bot-check must end the dialogue with an apology and END."

    return score, comment


def enforce_location_refusal_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if scenario.index != 33:
        return score, comment

    if _is_location_refusal_reply(assistant_reply):
        return 1, "Scenario 33 relaxed check passed: any semantically correct location-based refusal with END is acceptable."
    return score, comment


def enforce_strange_behavior_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if scenario.index != 6:
        return score, comment

    low = _normalized_low(assistant_reply)
    if _is_apology_end_reply(assistant_reply):
        return 1, "Scenario 6 relaxed check passed: apology with END is business-acceptable."
    if _contains_any_substring(
        low,
        [
            "не поняла",
            "не понял",
            "не совсем поняла",
            "не совсем понял",
            "не вполне поняла",
            "не вполне понял",
        ],
    ):
        return 1, "Scenario 6 relaxed check passed: clarification via 'did not understand' is acceptable."
    if not _reply_has_end_marker(assistant_reply) and "?" in (assistant_reply or ""):
        return 1, "Scenario 6 relaxed check passed: a business-equivalent continuation question is acceptable."
    return score, comment


def enforce_profile_reference_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if scenario.index not in PROFILE_REFERENCE_SCENARIOS:
        return score, comment

    if _reply_has_end_marker(assistant_reply):
        return 0, f"Scenario {scenario.index} relaxed profile-reference check failed: assistant must not end the dialogue."
    if not _looks_like_direct_answer_request(assistant_reply):
        return (
            0,
            f"Scenario {scenario.index} relaxed profile-reference check failed: assistant should motivate the candidate to answer directly in chat.",
        )
    return 1, f"Scenario {scenario.index} relaxed profile-reference check passed: direct answer request is present."









# -----------------------
# Запуск single и chain
# -----------------------




def run_single_scenario(
    client: OpenAI,
    cfg: Dict[str, Any],
    scenario: Scenario,
    messages_per_scenario: int,
    usage: Dict[str, Dict[str, int]],
    dialog_context: str,
    dialog_context_meta: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_messages = generate_candidate_messages_for_scenario(
        client=client,
        scenario=scenario,
        messages_per_scenario=messages_per_scenario,
        usage_bucket=usage["candidate_generator"],
        dialog_context_meta=dialog_context_meta,
    )

    turns: List[Dict[str, Any]] = []

    for step_idx, cand_msg in enumerate(candidate_messages, start=1):
        run_dialog: List[Dict[str, str]] = []
        if scenario.index == 40:
            assistant = create_conversation_assistant(cfg, dialog_context=dialog_context)
            conv_id = assistant.create_conversation()
            for seed_msg in _hh_profile_reference_seed_messages(dialog_context_meta):
                seed_reply = assistant.reply_in_conversation(conv_id, seed_msg)
                _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))
                run_dialog.append({"role": "candidate", "text": seed_msg})
                run_dialog.append({"role": "assistant", "text": seed_reply})
            reply = assistant.reply_in_conversation(conv_id, cand_msg)
            _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))
            run_dialog.append({"role": "candidate", "text": cand_msg})
            run_dialog.append({"role": "assistant", "text": reply})
            turn = {
                "step": step_idx,
                "candidate_message": cand_msg,
                "assistant_reply": reply,
                "dialog_text": _dialog_to_text(run_dialog),
            }
        else:
            assistant = create_simple_assistant(cfg, dialog_context=dialog_context)
            reply = assistant.reply_one_turn(cand_msg)
            _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))
            turn = {
                "step": step_idx,
                "candidate_message": cand_msg,
                "assistant_reply": reply,
            }
        turns.append(turn)

    case_results = _evaluate_single_case(
        client=client,
        scenario=scenario,
        turns=turns,
        dialog_context_meta=dialog_context_meta,
        usage_bucket=usage["evaluator"],
    )
    scenario_score = 0
    turns_total = len(turns)
    for turn in turns:
        step = int(turn.get("step") or 0)
        score, comment = case_results.get(step, (0, "Case judge missing result for this step."))
        score, comment = enforce_prompt_v2_hard_rules(
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_gross_salary_clarification_rule(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_salary_normalization_rules(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_strange_behavior_rules(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_profile_reference_rules(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_bot_identity_rules(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_location_refusal_rules(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
        )
        score, comment = enforce_recruiting_agency_rules(
            scenario=scenario,
            assistant_reply=str(turn.get("assistant_reply") or ""),
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        turn["score"] = score
        turn["comment"] = comment
        scenario_score += score

    case = {
        "case_id": f"S{scenario.index}",
        "type": "single",
        "scenario_index": scenario.index,
        "scenario_name": scenario.name,
        "cdm_file": str(dialog_context_meta.get("cdm_file") or ""),
        "company_hidden": bool(dialog_context_meta.get("company_hidden", False)),
        "vacancy_ref": build_vacancy_ref(dialog_context_meta),
        "prompt_context": build_prompt_context_ref(dialog_context_meta),
        "turns_total": turns_total,
        "score_total": scenario_score,
        "passed": scenario_score == turns_total,
        "turns": turns,
    }
    return case


def run_chain_group(
    client: OpenAI,
    cfg: Dict[str, Any],
    group: ScenarioGroup,
    messages_per_scenario: int,
    usage: Dict[str, Dict[str, int]],
    dialog_context: str,
    dialog_context_meta: Dict[str, Any],
) -> Dict[str, Any]:
    assistant = create_conversation_assistant(cfg, dialog_context=dialog_context)
    scenarios = group.scenarios

    # генерим варианты: для каждого сценария - messages_per_scenario разных сообщений
    candidate_variants: Dict[int, List[str]] = {}
    for s in scenarios:
        msgs = generate_candidate_messages_for_scenario(
            client=client,
            scenario=s,
            messages_per_scenario=messages_per_scenario,
            usage_bucket=usage["candidate_generator"],
            dialog_context_meta=dialog_context_meta,
        )
        candidate_variants[s.index] = msgs

    runs: List[Dict[str, Any]] = []

    # messages_per_scenario здесь - количество прогонов диалога
    for run_idx in range(1, messages_per_scenario + 1):
        conv_id = assistant.create_conversation()
        dialog_history: List[Dict[str, str]] = []
        run_dialog: List[Dict[str, str]] = []
        turns: List[Dict[str, Any]] = []

        if group.group_id == "chain_pause_resume_questions":
            for seed_msg in _hh_pause_questions_seed_messages(dialog_context_meta):
                seed_reply = assistant.reply_in_conversation(conv_id, seed_msg)
                _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))
                run_dialog.append({"role": "candidate", "text": seed_msg})
                run_dialog.append({"role": "assistant", "text": seed_reply})
                dialog_history.append({"candidate": seed_msg, "assistant": seed_reply})

        for step_idx, s in enumerate(scenarios, start=1):
            cand_msg = candidate_variants[s.index][run_idx - 1]
            reply = assistant.reply_in_conversation(conv_id, cand_msg)
            _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))

            turn = {
                "step": step_idx,
                "scenario_index": s.index,
                "scenario_name": s.name,
                "candidate_message": cand_msg,
                "assistant_reply": reply,
            }
            turns.append(turn)

            run_dialog.append({"role": "candidate", "text": cand_msg})
            run_dialog.append({"role": "assistant", "text": reply})
            dialog_history.append({"candidate": cand_msg, "assistant": reply})

        runs.append(
            {
                "run_index": run_idx,
                "dialog": run_dialog,
                "dialog_text": _dialog_to_text(run_dialog),
                "turns": turns,
            }
        )

    case_results = _evaluate_chain_case(
        client=client,
        group=group,
        runs=runs,
        dialog_context_meta=dialog_context_meta,
        usage_bucket=usage["evaluator"],
    )
    total_score = 0
    total_turns = 0
    for run in runs:
        run_score = 0
        run_turns = len(run.get("turns") or [])
        for turn in run.get("turns") or []:
            key = (int(run.get("run_index") or 0), int(turn.get("step") or 0))
            score, comment = case_results.get(key, (0, "Case judge missing result for this step."))
            score, comment = enforce_prompt_v2_hard_rules(
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_gross_salary_clarification_rule(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_salary_normalization_rules(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_strange_behavior_rules(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_profile_reference_rules(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_bot_identity_rules(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_location_refusal_rules(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
            )
            score, comment = enforce_recruiting_agency_rules(
                scenario=scenarios[int(turn.get("step") or 0) - 1],
                assistant_reply=str(turn.get("assistant_reply") or ""),
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            turn["score"] = score
            turn["comment"] = comment
            run_score += score
        run["turns_total"] = run_turns
        run["score_total"] = run_score
        run["passed"] = run_score == run_turns
        total_score += run_score
        total_turns += run_turns

    case = {
        "case_id": f"C_{group.group_id}",
        "type": "chain",
        "scenario_indices": [s.index for s in scenarios],
        "scenario_names": [s.name for s in scenarios],
        "cdm_file": str(dialog_context_meta.get("cdm_file") or ""),
        "company_hidden": bool(dialog_context_meta.get("company_hidden", False)),
        "vacancy_ref": build_vacancy_ref(dialog_context_meta),
        "prompt_context": build_prompt_context_ref(dialog_context_meta),
        "runs_total": len(runs),
        "turns_total": total_turns,
        "score_total": total_score,
        "passed": all(bool(run.get("passed")) for run in runs),
        "runs": runs,
    }
    return case


# -----------------------
# Основной раннер
# -----------------------


def run_scenarios(
    csv_path: pathlib.Path,
    messages_per_scenario: int,
    max_scenarios: int | None = None,
    cdm_dir: pathlib.Path = DEFAULT_CDM_DIR,
    scenario_indices: Optional[List[int]] = None,
) -> pathlib.Path:
    ensure_dirs()

    print(f"[init] Loading scenarios from CSV: {csv_path}")
    scenarios = load_scenarios(csv_path)

    # ВАЖНО: если пользователь передал --scenario-indices, но он распарсился в пустоту,
    # не запускаем все молча, а падаем с понятной ошибкой.
    if scenario_indices is not None and len(scenario_indices) == 0:
        raise ValueError(
            "--scenario-indices was provided but parsed as an empty list. "
            "Pass comma-separated indices like: --scenario-indices 23,24"
        )

    if scenario_indices:
        selected = set(scenario_indices)
        scenarios = [s for s in scenarios if s.index in selected]
        print(
            f"[init] Scenario index filter applied: "
            f"indices={sorted(selected)} -> loaded={len(scenarios)}"
        )

    disabled_loaded = sorted(s.index for s in scenarios if s.index in HH_DISABLED_SCENARIOS)
    if disabled_loaded:
        scenarios = [s for s in scenarios if s.index not in HH_DISABLED_SCENARIOS]
        print(
            f"[init] HH disabled scenarios skipped: indices={disabled_loaded} "
            f"-> loaded={len(scenarios)}"
        )

    if not scenarios:
        if scenario_indices:
            raise ValueError(
                "No scenarios left after applying --scenario-indices and HH disabled-scenario filtering. "
                f"Requested: {scenario_indices}. Check CSV scenario indices."
            )
        raise ValueError("No scenarios loaded from CSV - nothing to run.")

    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
        if not scenarios:
            raise ValueError("No scenarios left after applying --max-scenarios.")

    cdm_fixtures = load_cdm_fixtures(cdm_dir)
    print(f"[init] CDM fixtures loaded: {len(cdm_fixtures)} from {cdm_dir}")

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    groups = build_scenario_groups(scenarios)
    print(f"[init] Scenario groups built: {len(groups)}")
    for g in groups:
        if g.kind == "chain":
            chain_list = ", ".join([str(s.index) for s in g.scenarios])
            print(f"  [chain] {g.group_id}: indices=[{chain_list}]")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    usage = {
        "candidate_generator": _blank_usage(),
        "screening_assistant": _blank_usage(),
        "evaluator": _blank_usage(),
    }

    cases: List[Dict[str, Any]] = []

    print(
        f"[run] Starting screening_scenarios run_id={run_id} | "
        f"cases={len(groups)} | messages_per_scenario={messages_per_scenario}"
    )

    for gidx, group in enumerate(groups, start=1):
        print(f"\n[case {gidx}/{len(groups)}] {group.group_id} ({group.kind})")
        fixture = _select_cdm_fixture_for_group(cdm_fixtures, group, gidx)
        vacancy_info_override, meta_override = _hh_recruiting_agency_context_override(group)
        if vacancy_info_override or meta_override:
            fixture = _copy_fixture_with_overrides(
                fixture,
                vacancy_info_override=vacancy_info_override,
                names_override={
                    "recruiter_name": meta_override.get("recruiter_name", "") if meta_override else "",
                    "candidate_name": meta_override.get("candidate_name", "") if meta_override else "",
                } if meta_override else None,
            )
        dialog_context, dialog_context_meta = build_dialog_context(fixture=fixture)
        if meta_override:
            for key, value in meta_override.items():
                if key in {"recruiter_name", "candidate_name"}:
                    continue
                dialog_context_meta[key] = value
        print(
            f"  - cdm={fixture.file_name} | "
            f"company_hidden=NO"
        )

        if group.kind == "single":
            scenario = group.scenarios[0]
            case = run_single_scenario(
                client=client,
                cfg=cfg,
                scenario=scenario,
                messages_per_scenario=messages_per_scenario,
                usage=usage,
                dialog_context=dialog_context,
                dialog_context_meta=dialog_context_meta,
            )
        else:
            case = run_chain_group(
                client=client,
                cfg=cfg,
                group=group,
                messages_per_scenario=messages_per_scenario,
                usage=usage,
                dialog_context=dialog_context,
                dialog_context_meta=dialog_context_meta,
            )

        g_score = int(case.get("score_total") or 0)
        g_turns = int(case.get("turns_total") or 0)
        cases.append(case)

        print(
            f"  - case score: {g_score}/{g_turns} | "
            f"passed={'YES' if bool(case.get('passed')) else 'NO'}"
        )

    cases_total = len(cases)
    turns_total = sum(int(case.get("turns_total", 0)) for case in cases)
    score_total = sum(int(case.get("score_total", 0)) for case in cases)
    passed_cases = sum(1 for case in cases if bool(case.get("passed")))
    failed_cases = cases_total - passed_cases
    pass_rate = (passed_cases / cases_total * 100.0) if cases_total else 0.0

    mismatches = build_mismatches(cases)
    failed_turns = sum(len(item.get("dialogs") or []) for item in mismatches)
    errors: List[Dict[str, Any]] = []
    token_usage_total = _token_usage_total(usage)
    sa_cfg = _component_cfg(cfg, HH_COMPONENT_NAME)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "csv_path": str(csv_path),
        "cdm_dir": str(cdm_dir),
        "cdm_fixture": {
            "mode": "round_robin_by_case",
            "fixtures_total": len(cdm_fixtures),
        },
        "messages_per_scenario": messages_per_scenario,
        "scenario_indices": scenario_indices or [],
        "max_scenarios": max_scenarios,
        "models": {
            "candidate_generator": GEN_MODEL,
            "screening_assistant_hh": {
                "prompt_id": os.environ.get("SCREENING_ASSISTANT_HH_PROMPT_ID") or sa_cfg.get("prompt_id"),
                "prompt_version": os.environ.get("SCREENING_ASSISTANT_HH_PROMPT_VERSION") or sa_cfg.get("prompt_version"),
                "component": HH_COMPONENT_NAME,
            },
            "evaluator": EVAL_MODEL,
        },
        "token_usage_total": token_usage_total,
        "token_usage": usage,
        "cases_total": cases_total,
        "turns_total": turns_total,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "failed_turns": failed_turns,
        "pass_rate": pass_rate,
        "score_total": score_total,
        "score_max": turns_total,
        "mismatches_count": len(mismatches),
        "errors_count": len(errors),
        "summary": {
            "cases_total": cases_total,
            "turns_total": turns_total,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "failed_turns": failed_turns,
            "pass_rate": pass_rate,
            "score_total": score_total,
            "score_max": turns_total,
            "mismatches_count": len(mismatches),
            "errors_count": len(errors),
        },
        "cases": cases,
        "mismatches": mismatches,
        "errors": errors,
    }

    out_path = REPORTS_DIR / f"screening_scenarios_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] Screening scenarios report saved to: {out_path}")
    print(
        f"[summary] cases_total={cases_total} | turns_total={turns_total} | "
        f"score_total={score_total} | pass_rate={pass_rate:.2f}% | "
        f"failed_cases={failed_cases} | failed_turns={failed_turns}"
    )

    return out_path

def _repeat_or_trim(messages: List[str], n: int) -> List[str]:
    if not messages or n <= 0:
        return []
    if len(messages) >= n:
        return messages[:n]
    out = list(messages)
    idx = 0
    while len(out) < n:
        out.append(messages[idx % len(messages)])
        idx += 1
    return out


def _generic_hh_fallback_messages(s: Scenario, n: int) -> List[str]:
    examples_override = _scenario_example_override(s)
    if examples_override is None:
        examples = extract_candidate_examples(s.examples_raw, max_examples=max(n, 5))
    else:
        examples = examples_override[:]
    if examples:
        return _repeat_or_trim(examples, n)
    return _repeat_or_trim(
        [
            f"Подскажите, пожалуйста, подробнее по сценарию: {s.name}.",
            f"Хочу уточнить детали по теме: {s.name.lower()}.",
        ],
        n,
    )


def _fallback_messages(
    s: Scenario,
    n: int,
    dialog_context_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    idx = s.index
    dialog_context_meta = dialog_context_meta or {}
    location = str(dialog_context_meta.get("location") or "").strip() or "Москва"
    min_salary = _parse_int_value(dialog_context_meta.get("min_salary"))
    max_salary = _parse_int_value(dialog_context_meta.get("max_salary"))
    salary_value = _format_int_with_spaces(_safe_salary_expectation_value(dialog_context_meta))

    static_messages: Dict[int, List[str]] = {
        5: [
            "Вы очень приятная, я бы хотел пригласить вас на свидание.",
            "Если честно, мне хотелось бы познакомиться с вами лично, а не только обсуждать вакансию.",
            "Вы мне понравились, давайте лучше поужинаем вместе и познакомимся поближе.",
        ],
        8: [
            "Описание вакансии выглядит слишком расплывчатым и непрофессиональным, не понимаю, что это за роль.",
            "Процесс выглядит сомнительно: по описанию вообще непонятно, какие тут реальные задачи.",
            "Вакансия сформулирована слишком слабо, поэтому впечатление от процесса сейчас скорее негативное.",
        ],
        10: [
            "Это не мой профиль, у меня другая специализация.",
            "У меня другой профиль, таким направлением я не занимаюсь.",
            "Эта вакансия не по моему профилю, у меня другая специализация.",
        ],
        11: [
            "Здравствуйте, подскажите, пожалуйста, какая зарплатная вилка у этой позиции?",
            "Добрый день! Можете уточнить, какая вилка по зарплате у этой вакансии?",
            "Подскажите, пожалуйста, какие условия по зарплате у этой роли?",
        ],
        7: [
            "Вы мошенники?",
            "Подтвердите, пожалуйста, что это не спам и не развод.",
            "Если это реальная вакансия, пришлите, пожалуйста, корпоративную почту или официальное подтверждение.",
        ],
        12: [
            "Напишите, пожалуйста, пример кода для этой задачи и покажите, как бы вы реализовали решение.",
            "Решите, пожалуйста, небольшую задачу: как бы вы написали алгоритм и оптимизировали его?",
            "Покажите конкретный пример реализации и объясните, как сделать это решение по шагам.",
        ],
        13: [
            "Расскажите, пожалуйста, подробнее про условия, формат работы и команду по этой вакансии.",
            "Можно коротко уточнить, какие условия и как устроен проект на этой роли?",
            "Подскажите, пожалуйста, подробнее про роль, формат работы и ключевые условия.",
        ],
        15: [
            "Давайте сразу договоримся о звонке в Zoom завтра в 15:00, чтобы все обсудить.",
            "Предлагаю созвониться сегодня в 18:00 по Teams и быстро пройтись по деталям.",
            "Можете назначить телефонный звонок на ближайший вторник в 10:00, чтобы все прояснить?",
        ],
        16: [
            "Я уже принял оффер и выхожу в новую компанию, поэтому предложение больше не рассматриваю.",
            "Контракт уже подписан, я трудоустроен и новые варианты сейчас не обсуждаю.",
            "У меня уже есть оффер, и я завершаю переход в новую компанию.",
        ],
        18: [
            "У меня нет релевантного опыта в этой сфере, это другая специальность.",
            "Такого релевантного опыта у меня нет, у меня другая специализация.",
            "У меня нет опыта в этой сфере, это не мой профессиональный профиль.",
        ],
        19: [
            "Извините, это бот или человек?",
            "Подскажите, пожалуйста, вы ИИ или живой рекрутер?",
            "Скажите честно, мне сейчас бот пишет или человек?",
        ],
        20: [
            "Я уже спрашивал: вы бот или человек?",
            "Снова уточню: это автоматический бот или живой рекрутер?",
            "Повторно спрашиваю, кто мне пишет — бот или человек?",
        ],
        21: [
            "Подскажите, пожалуйста, что это за компания и что за вакансия?",
            "Какая компания, какие задачи и какой формат работы?",
            "Можно подробнее про компанию и позицию?",
        ],
        22: [
            "Я уже спрашивал про зарплату, но вы так и не ответили. Повторяю вопрос: какая вилка по этой позиции?",
            "Повторно уточню: какой зарплатный диапазон у этой роли? Ранее ответа по вилке не получил.",
            "Снова спрошу только про зарплату: какая вилка по вакансии? Вы уже уходили от ответа.",
        ],
        23: [
            "В третий раз спрашиваю: какая зарплатная вилка по этой вакансии?",
            "Это уже третий мой вопрос про деньги: какой диапазон по зарплате у позиции?",
            "В третий раз возвращаюсь к вопросу о вилке: сколько платят по этой роли?",
        ],
        25: [
            f"Сейчас живу в {location}, локация для меня подходит.",
            f"Я нахожусь в {location} и могу рассматривать эту позицию.",
            f"Проживаю в {location}, с локацией вакансии проблем нет.",
        ],
        29: [
            "Ни офисный, ни удаленный формат не рассматриваю. К переезду тоже не готов.",
            "На месте работодателя работать не готов, удаленно тоже не хочу. Формат мне не подходит.",
            "Ни один из этих форматов мне не подходит, переезд тоже не рассматриваю.",
        ],
        30: [
            "После недавней утраты в семье я сейчас не готов обсуждать такие предложения.",
            "Недавно были похороны близкого человека, поэтому я пока не могу продолжать этот разговор.",
            "После смерти родственника мне сейчас тяжело возвращаться к таким обсуждениям.",
        ],
        32: [
            "Сейчас живу в Берлине, удаленный формат мне подходит.",
            "Я сейчас не в РФ, нахожусь в Испании и работаю удаленно.",
            "Живу за границей, сейчас я в Португалии.",
        ],
        36: ["50", "60", "70"],
        37: ["60-80", "50-70", "70-90"],
        38: ["1500 в час", "2000 руб/час", "1800 в час"],
        39: ["$4000", "3000 евро", "3500 euro"],
        40: [
            "По этому вопросу всё есть в резюме.",
            "Это уже указано у меня в профиле.",
            "Посмотрите, пожалуйста, резюме: там есть ответ на этот вопрос.",
            "Эта информация есть в LinkedIn.",
            "По этому пункту всё уже отражено в профиле.",
            "Ответ на этот вопрос есть в моем резюме.",
        ],
        41: [
            "Я в Москве, ориентир 300000 на руки, удаленный формат мне подходит. С Playwright и API-автотестами на C# работаю 4 года, QA-команду из 5 человек менторил. Хорошо, буду ждать звонка.",
            "Москва, по деньгам 320000 на руки, remote мне подходит. С Playwright и API-автотестами на C# работаю 5 лет, команду из 4 QA менторил. Жду фидбек.",
            "Я в Москве, ориентир 300000, удаленный формат подходит. Playwright и API-автотесты на C# делаю 4 года, опыт менторства QA-команды есть. Хорошо, буду ждать звонка.",
        ],
        42: [
            "Вернемся к этому позже, пожалуйста. По зарплате смотрю от 300.",
            "Давайте позже обсудим, сейчас не готов. По деньгам ориентир около 320.",
            "Позже будет актуально. По зарплате ориентир 300 на руки.",
        ],
        43: [
            "Давайте продолжим.",
            "Снова на связи, можем продолжить диалог.",
            "Да, продолжаем.",
        ],
        44: [
            "Пока не готов продолжать, напишите позже.",
            "Сейчас не время, позже обсудим.",
            "Давайте вернемся к этому разговору позже.",
        ],
        45: [
            "Давайте продолжим.",
            "Снова на связи, можем продолжить диалог.",
            "Да, продолжаем.",
        ],
        46: [
            "Подскажите, пожалуйста, какой именно формат работы у этой вакансии?",
            "Можно уточнить, какой формат работы предусмотрен для этой роли?",
            "Какой формат работы у позиции: удаленный, офисный, гибридный или разъездной?",
        ],
        49: [
            "Москва,\nСейчас у меня 1 млн руб. gross\nЭто ЗП+Премия\n\nРассчитываю + 15/20%",
            "Москва. Сейчас суммарно около 1 млн gross с учетом премии, при переходе смотрю плюс 15-20%.",
            "Я в Москве, сейчас доход примерно 1 млн gross вместе с премией. При смене работы ориентируюсь на плюс 15-20%.",
            "К гибриду готов.\nУ меня сейчас 600 гросс, хотелось бы чуть повыше, но обсуждаемо",
            "Гибридный формат подходит. Сейчас получаю 600 gross, при переходе хотел бы немного выше, но готов обсуждать.",
            "К гибриду готов, сейчас компенсация около 600 гросс. Хотелось бы рост, но можно обсуждать.",
        ],
        50: [
            "Подскажите, пожалуйста, что это за компания?",
            "Добрый день. А какая компания по этой вакансии?",
            "Здравствуйте, подскажите, пожалуйста, что за компания у этой позиции?",
        ],
    }
    alias_messages = {}
    dynamic_builders: Dict[int, Callable[[], List[str]]] = {
        26: lambda: [
            f"Мой текущий город — {location}, по зарплате ориентируюсь на {salary_value} рублей на руки.",
            f"Сейчас я в {location}, по деньгам рассматриваю около {salary_value} на руки.",
            f"Текущая локация — {location}, по компенсации ориентируюсь на {salary_value} рублей.",
        ],
        27: lambda: [
            f"Живу в Королеве, по зарплате ориентируюсь на {salary_value} рублей на руки.",
            f"Я в Химках, по деньгам рассматриваю {salary_value} на руки.",
            f"Я из Подольска, по компенсации ориентируюсь на {salary_value} рублей.",
        ],
        28: lambda: [
            f"В офис ездить не готов, но удаленный формат мне подходит. Я в Москве, ожидания {salary_value} рублей на руки в месяц.",
            f"На месте работодателя работать не хочу, а вот удаленно готов. Сейчас я в Москве, ориентир {salary_value} рублей в месяц на руки.",
            f"Офисный формат не рассматриваю, но remote меня устраивает. Я в Москве, по деньгам {salary_value} рублей на руки в месяц.",
        ],
        33: lambda: [
            f"Я в Санкт-Петербурге, по зарплате ориентир {salary_value} рублей на руки в месяц.",
            f"Сейчас живу в Екатеринбурге, ожидания {salary_value} рублей на руки в месяц.",
            f"Я в Казани, по деньгам ориентируюсь на {salary_value} рублей в месяц на руки.",
        ],
        31: lambda: [
            f"Сейчас живу во Владивостоке, по зарплате ориентируюсь на {salary_value} рублей на руки.",
            f"Мой текущий город — Новосибирск, ожидания по деньгам около {salary_value} рублей на руки.",
            f"Я сейчас в Калининграде, по компенсации ориентируюсь на {salary_value} рублей в месяц на руки.",
        ],
        34: lambda: [
            f"{int(round(((int(min_salary or 250000) + int(max_salary or max(int(min_salary or 250000), 350000))) / 2) / 10_000) * 10_000)} рублей на руки в месяц",
            f"{max(int(min_salary or 250000), int(round(((int(min_salary or 250000) + int(max_salary or max(int(min_salary or 250000), 350000))) / 2) / 10_000) * 10_000) - 10000)} руб на руки в месяц",
            f"{min(int(max_salary or max(int(min_salary or 250000), 350000)), int(round(((int(min_salary or 250000) + int(max_salary or max(int(min_salary or 250000), 350000))) / 2) / 10_000) * 10_000) + 10000)} рублей в месяц net",
        ],
        35: lambda: [
            f"{int(max_salary or 350000) + 100000} рублей на руки в месяц",
            f"{int(max_salary or 350000) + 150000} руб в месяц",
            f"{int(max_salary or 350000) + 120000} рублей net в месяц",
        ],
        47: lambda: [
            f"Сейчас я в Москве, ожидания {salary_value} рублей на руки в месяц.",
            f"Я в Москве, по деньгам ориентир {salary_value} рублей на руки в месяц.",
            f"Мой город Москва, ориентир по компенсации {salary_value} рублей в месяц на руки.",
        ],
    }

    if idx in alias_messages:
        return _repeat_or_trim(static_messages[alias_messages[idx]], n)
    if idx in static_messages:
        return _repeat_or_trim(static_messages[idx], n)
    if idx in dynamic_builders:
        return _repeat_or_trim(dynamic_builders[idx](), n)
    return _generic_hh_fallback_messages(s, n)


def _generated_message_matches_scenario_constraints(scenario_index: int, message: str) -> bool:
    low = _normalize_text(message).lower()
    if _message_has_unwanted_rudeness(message, scenario_index):
        return False
    repeated_markers = [
        "повтор",
        "снова",
        "уже спрашивал",
        "второй раз",
        "третий раз",
        "опять спрашиваю",
        "повторю вопрос",
        "так и не ответили",
        "ответа не было",
    ]
    technical_markers = ["код", "пример", "напиши", "реализ", "как сделать", "алгоритм", "реши", "задач", "оптимиз"]
    schedule_markers = ["созвон", "созвониться", "звонок", "встреч", "zoom", "teams", "телефон", "слот"]
    money_markers = ["аванс", "в долг", "перевед", "скинь", "деньг", "предоплат"]
    offer_markers = ["оффер", "трудоустро", "контракт подписан", "выхожу в", "новую компанию", "already employed"]
    flirt_markers = ["кофе", "встретим", "лично", "неформаль", "приятн", "свидан"]
    legitimacy_markers = ["спам", "мошенн", "развод", "легитим", "корпоративн", "официальн"]
    criticism_markers = ["расплыв", "сомн", "непрофессион", "слаб", "негатив", "ужасн", "странн", "плох", "непонятн"]
    source_markers = ["откуда у вас мои данные", "где вы нашли мой профиль", "где нашли мой профиль", "откуда мой контакт"]
    conditions_markers = ["услови", "формат работы", "команд", "проект", "роль", "подробн", "детал"]
    company_info_markers = ["компан", "ваканси", "позици", "задач", "формат работы", "проект"]
    company_link_markers = ["какая компания", "как называется компания", "название компании", "ссылк", "сайт", "почитать"]
    death_markers = ["смерт", "утрат", "похорон", "покойн", "не стало", "сконч"]
    salary_markers = ["руб", "зарплат", "деньг", "оклад", "на руки", "в месяц", "net", "gross"]
    format_question_markers = ["формат работы", "удален", "удалён", "офис", "гибрид", "разъезд", "на месте работодателя"]
    aggression_markers = UNWANTED_RUDENESS_MARKERS
    if scenario_index == 10:
        return _contains_any_substring(low, ["не мой профиль", "другая специализац", "другой профиль"]) and not _contains_any_substring(
            low, ["нет опыта", "нет релевант"]
        )
    if scenario_index == 18:
        return _contains_any_substring(low, ["нет релевант", "нет опыта", "другая специальн", "в этой сфере"])
    if scenario_index == 19:
        return not _contains_any_substring(low, repeated_markers)
    if scenario_index == 20:
        return _contains_any_substring(low, repeated_markers)
    if scenario_index == 5:
        return _contains_any_substring(low, flirt_markers)
    if scenario_index == 7:
        return _contains_any_substring(low, legitimacy_markers) and not _contains_any_substring(low, source_markers)
    if scenario_index == 8:
        return _contains_any_substring(low, criticism_markers)
    if scenario_index == 12:
        return _contains_any_substring(low, technical_markers)
    if scenario_index == 13:
        return _contains_any_substring(low, conditions_markers) and not _contains_any_substring(low, technical_markers)
    if scenario_index == 14:
        return _contains_any_substring(low, company_info_markers) and not _contains_any_substring(low, technical_markers)
    if scenario_index == 15:
        return _contains_any_substring(low, schedule_markers) and not _contains_any_substring(low, money_markers)
    if scenario_index == 16:
        return _contains_any_substring(low, offer_markers)
    if scenario_index == 21:
        return _contains_any_substring(low, company_info_markers)
    if scenario_index == 25:
        return _contains_any_substring(low, ["живу в", "нахожусь в", "проживаю в", "мой текущий город", "локация для меня"])
    if scenario_index == 30:
        return _contains_any_substring(low, death_markers) and not _contains_any_substring(low, salary_markers)
    if scenario_index == 46:
        return _contains_any_substring(low, format_question_markers) and not _contains_any_substring(low, aggression_markers)
    if scenario_index == 49:
        return (
            "моск" in low and _contains_any_substring(low, ["gross", "гросс", "прем", "%", "процент"])
        ) or (
            _contains_any_substring(low, ["гибрид", "гибриду", "hybrid"])
            and _contains_any_substring(low, ["gross", "гросс", "600"])
        )
    if scenario_index == 50:
        return _contains_any_substring(low, ["компан", "какая компания", "что за компания", "позици"])
    return True


# -----------------------
# CLI
# -----------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavioral screening scenarios against screening_assistant_hh (supports chain multi-scenario dialogs)."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to CSV with scenarios (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--messages-per-scenario",
        type=int,
        default=DEFAULT_MESSAGES_PER_SCENARIO,
        help=(
            "SINGLE: how many independent messages to test per scenario.\n"
            "CHAIN: how many full dialog runs to test per chain."
        ),
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit number of scenarios read from CSV (debug). Default: all.",
    )
    parser.add_argument(
        "--scenario-indices",
        type=str,
        default=None,
        help="Comma-separated CSV row indices to run, e.g. 23,24. Default: all.",
    )
    parser.add_argument(
        "--cdm-dir",
        type=str,
        default=str(DEFAULT_CDM_DIR),
        help=f"Path to CDM fixtures directory (default: {DEFAULT_CDM_DIR})",
    )

    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)

    scenario_indices: Optional[List[int]] = None
    if args.scenario_indices is not None:
        scenario_indices = parse_scenario_indices(args.scenario_indices)
        if len(scenario_indices) == 0:
            raise ValueError(
                "--scenario-indices was provided but empty. Example: --scenario-indices 23,24"
            )

    report_path = run_scenarios(
        csv_path=csv_path,
        messages_per_scenario=args.messages_per_scenario,
        max_scenarios=args.max_scenarios,
        cdm_dir=pathlib.Path(args.cdm_dir),
        scenario_indices=scenario_indices,
    )
    print("Screening scenarios report ->", report_path)


if __name__ == "__main__":
    main()

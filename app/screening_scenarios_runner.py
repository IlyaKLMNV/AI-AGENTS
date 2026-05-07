from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from adapters.adapters import names_from_cdm, to_vacancy_info

# -----------------------
# Константы и пути
# -----------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "screening_scenarios"

DEFAULT_CSV_PATH = ROOT / "tests" / "fixtures" / "screening_scenarios.csv"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
DEFAULT_MESSAGES_PER_SCENARIO = 3

GEN_MODEL = "gpt-4.1-mini"
EVAL_MODEL = "gpt-4.1"

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


def load_scenarios(csv_path: pathlib.Path) -> List[Scenario]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV with scenarios not found: {csv_path}")

    scenarios: List[Scenario] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        name_key = "Название сценария"
        desc_key_candidates = [
            "Краткое описание сценария",
            "Описание сценария",
        ]

        behavior_key_candidates = [
            "Ожидаемое поведение модели (согласно промпту) ",
            "Ожидаемое поведение модели (согласно промпту)",
            "Ожидаемое поведение модели (как она должна отработать)",
        ]

        examples_key_candidates = [
            "Сообщениия с примерами диалогов ",
            "Сообщения с примерами диалогов",
        ]

        for idx, row in enumerate(reader, start=1):
            scenario_name = (row.get(name_key) or "").strip()
            if not scenario_name:
                continue

            description = ""
            for key in desc_key_candidates:
                if key in row and row[key]:
                    description = row[key]
                    break

            expected_behavior = ""
            for key in behavior_key_candidates:
                if key in row and row[key]:
                    expected_behavior = row[key]
                    break

            examples_raw = ""
            for key in examples_key_candidates:
                if key in row and row[key]:
                    examples_raw = row[key]
                    break

            scenarios.append(
                Scenario(
                    index=idx,
                    name=scenario_name,
                    description=description,
                    expected_behavior=expected_behavior,
                    examples_raw=examples_raw,
                )
            )

    return scenarios


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


# -----------------------
# Вытаскиваем реальные реплики кандидатов
# -----------------------


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

TOPIC_COMPANY_HIDDEN = ["компан", "скрыт", "не называете компанию", "название компании"]

# Явные цепочки по индексам строк CSV.
# Важно: порядок в списке - это порядок шагов в одном диалоге.
CHAIN_BY_INDEX: Dict[str, List[int]] = {
    "chain_salary_3x": [12, 29, 30],  # 1-й, 2-й, 3-й вопрос кандидата о ЗП
    "chain_bot_check": [26, 27],
    "chain_company_hidden": [23, 24],
}


CHAIN_BY_INDEX["chain_pause_resume_priority"] = [55, 56]
CHAIN_BY_INDEX["chain_pause_resume_questions"] = [57, 58]
CHAIN_BY_INDEX["chain_contact_source_resume"] = [59, 60]
CHAIN_BY_INDEX["chain_profile_reference_resume"] = [61, 62]

CONTACT_SOURCE_FILLED_SCENARIOS = {8, 20, 40, 59}
CONTACT_SOURCE_EMPTY_SCENARIOS = {41}
CONTACT_SOURCE_SCENARIOS = CONTACT_SOURCE_FILLED_SCENARIOS | CONTACT_SOURCE_EMPTY_SCENARIOS
CONTACT_SOURCE_RESUME_SCENARIOS = {60}
LEGITIMACY_SCENARIOS = {7, 42}
SALARY_NORMALIZATION_SCENARIOS = {43, 44, 45, 46, 47, 48, 49}
PROFILE_REFERENCE_SCENARIOS = {50, 51, 61}
PROFILE_REFERENCE_RESUME_SCENARIOS = {62}
PAUSE_LATER_SCENARIOS = {52, 53, 54, 55, 57}
PAUSE_LATER_RESUME_SCENARIOS = {56, 58}
FORCED_FALLBACK_SCENARIOS = (
    CONTACT_SOURCE_SCENARIOS
    | CONTACT_SOURCE_RESUME_SCENARIOS
    | LEGITIMACY_SCENARIOS
    | SALARY_NORMALIZATION_SCENARIOS
    | PROFILE_REFERENCE_SCENARIOS
    | PAUSE_LATER_SCENARIOS
    | PAUSE_LATER_RESUME_SCENARIOS
    | PROFILE_REFERENCE_RESUME_SCENARIOS
)
PROMPT_V2_SPECIAL_FIXTURES = {"cdm_16.json", "cdm_17.json"}


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

S10_RF_ONLY_SCRIPT = (
    "Спасибо за информацию. В настоящий момент рассматриваются только кандидаты, "
    "проживающие на территории РФ. Прошу прощения за беспокойство. END"
)
S38_DEATH_LOSS_SCRIPT = "Прошу прощения за беспокойство. END"

S7_LEGITIMACY_SCRIPT = "Прошу прощения за беспокойство. END"
S14_SALARY_REJECTION_SCRIPT = (
    "Понимаю ваши ожидания, но, к сожалению, бюджет на эту позицию не позволяет "
    "их рассмотреть. Желаю вам удачи в дальнейших поисках! END"
)
CONTACT_SOURCE_FALLBACK_FRAGMENT = "Коллеги передали контакт из базы кандидатов"
PAUSE_SCRIPT_MARKERS = [
    "готова вернуться к этому диалогу позже",
    "корректно зафиксировать диалог в базе",
    "ключевые ответы по вакансии",
]
FINISH_REPLY_MARKERS = [
    "это вся информация, которая была мне нужна",
    "передам ее внутреннему рекрутеру",
    "свяжется с вами по поводу следующих шагов",
]

SCENARIO_EXAMPLE_OVERRIDES: Dict[int, List[str]] = {
    7: [
        "Откуда у вас мои данные?",
        "Где вы нашли мой профиль?",
        "Это спам?",
        "Вы мошенники?",
        "Подтвердите легитимность, пожалуйста.",
    ],
    10: [
        "Я сейчас живу не в РФ.",
        "Сейчас нахожусь за границей, не в России.",
        "Я живу за пределами РФ.",
    ],
    17: [
        "Вы уже писали мне по этой вакансии, повторно это не актуально.",
        "Мы уже общались, не нужно снова писать по этому предложению.",
        "Вы уже писали, повторно это не актуально.",
    ],
    25: [
        "Этого опыта у меня нет, это не мой профиль.",
        "У меня другая специализация, таким не занимаюсь.",
        "Это не мой профиль, такого опыта у меня нет.",
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
    if "office" in low or "офис" in low:
        return "office"
    if "hybrid" in low or "гибрид" in low:
        return "hybrid"
    if "remote" in low or "удал" in low:
        return "remote"
    return low


def _is_office_or_hybrid_work_format(work_format: str) -> bool:
    return _canonical_work_format(work_format) in {"office", "hybrid"}


def _work_format_label(work_format: str) -> str:
    canonical = _canonical_work_format(work_format)
    if canonical == "office":
        return "Офис"
    if canonical == "hybrid":
        return "Гибрид"
    if canonical == "remote":
        return "Удаленно"
    return str(work_format or "").strip()


def _work_format_phrase(work_format: str) -> str:
    canonical = _canonical_work_format(work_format)
    if canonical == "office":
        return "в офисе"
    if canonical == "hybrid":
        return "в гибридном формате"
    return str(work_format or "").strip()


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


QUESTION_MARKER_STOPWORDS = {
    "есть",
    "если",
    "или",
    "вас",
    "вам",
    "ваш",
    "ваша",
    "ваше",
    "ваши",
    "был",
    "была",
    "были",
    "это",
    "эти",
    "этот",
    "прямо",
    "диалоге",
    "пожалуйста",
    "опыт",
    "работы",
    "работал",
    "работали",
    "насколько",
    "уверенно",
    "каком",
    "какая",
    "какие",
    "городе",
    "сейчас",
    "сумму",
    "рублях",
    "руки",
    "месяц",
    "месяца",
    "подходит",
}

QUESTION_MARKER_SPECIALS = [
    "playwright",
    "c#",
    "php",
    "symfony",
    "sql",
    "api",
    "qa",
    "ментор",
    "команд",
    "руковод",
    "лид",
    "bot",
    "rpa",
    "nlu",
    "rag",
    "llm",
]


def _extract_additional_questions(dialog_context_meta: Optional[Dict[str, Any]] = None) -> List[str]:
    dialog_context_meta = dialog_context_meta or {}
    raw_questions = str(dialog_context_meta.get("questions") or "")
    questions: List[str] = []
    for line in raw_questions.splitlines():
        content = QUESTION_PREFIX_RE.sub("", str(line).strip()).strip()
        if not content or content == "-":
            continue
        questions.append(content)
    return questions


def _question_markers_from_text(question: str) -> List[str]:
    low = _normalize_text(question).lower()
    if not low:
        return []

    markers: List[str] = []
    for special in QUESTION_MARKER_SPECIALS:
        if special in low:
            markers.append(special)

    for token in re.findall(r"[a-zа-яё#\+]{3,}", low):
        if token in QUESTION_MARKER_STOPWORDS:
            continue
        if token in QUESTION_MARKER_SPECIALS:
            continue
        if token.isascii():
            markers.append(token)
            continue
        if len(token) >= 5:
            markers.append(token[:6])

    unique: List[str] = []
    seen = set()
    for marker in markers:
        if marker and marker not in seen:
            unique.append(marker)
            seen.add(marker)
    return unique


def _reply_mentions_question_markers(reply_low: str, markers: List[str]) -> bool:
    if not markers:
        return False
    return any(marker in reply_low for marker in markers)


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
        if _has_any(n, TOPIC_COMPANY_HIDDEN) and "скрыт" in n:
            return "chain_company_hidden"
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


def _group_has_any_scenario(group: ScenarioGroup, indices: List[int]) -> bool:
    wanted = set(indices)
    return any(s.index in wanted for s in group.scenarios)


def _group_uses_prompt_v2_special_fixtures(group: ScenarioGroup) -> bool:
    return _group_has_any_scenario(group, [7, 8] + list(range(40, 63)))


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
    work_format = str(vacancy_info.get("work_format") or "").strip()
    location = str(vacancy_info.get("location") or "").strip().lower()

    if _group_uses_prompt_v2_special_fixtures(group):
        return fixture.file_name in PROMPT_V2_SPECIAL_FIXTURES

    if _group_has_any_scenario(group, [34, 35, 36, 37, 39]):
        if not _is_office_or_hybrid_work_format(work_format):
            return False

    if _group_has_any_scenario(group, [35]):
        return "моск" in location

    if _group_has_any_scenario(group, [36]):
        return bool(location) and "моск" not in location and "удал" not in location

    if _group_has_any_scenario(group, [28, 31]):
        company_name = str(vacancy_info.get("company_name") or "").strip()
        if not _is_real_company_name(company_name):
            return False

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
            raw_url = str(vacancy.get("vacancy_url") or "").strip()
            if raw_url:
                company_info = dict(vacancy_info.get("company_info") or {})
                company_info["vacancy_url"] = raw_url
                vacancy_info["company_info"] = company_info

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


def _is_hidden_company_scenario(s: Scenario) -> bool:
    return s.index in (23, 24)


def _group_requires_hidden_company(group: ScenarioGroup) -> bool:
    return any(_is_hidden_company_scenario(s) for s in group.scenarios)


def build_dialog_context(
    fixture: CdmFixture,
    hide_company: bool,
    contact_source_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    vacancy_info = fixture.vacancy_info
    names = fixture.names

    recruiter_name = str(names.get("recruiter_name") or "Рекрутер").strip()
    candidate_name = str(names.get("candidate_name") or "Кандидат").strip()
    title = str(vacancy_info.get("title") or "").strip()
    original_company_name = str(vacancy_info.get("company_name") or "").strip()
    company_name = "СКРЫТО" if hide_company else original_company_name
    responsibilities = str(vacancy_info.get("responsibilities") or "").strip()
    raw_work_format = str(vacancy_info.get("work_format") or "").strip()
    # Keep prompt-facing context aligned with the real backend payload:
    # work_format comes as office/hybrid/remote and should be passed through as-is.
    work_format = raw_work_format
    location = str(vacancy_info.get("location") or "").strip()
    min_salary = str(vacancy_info.get("min_salary") or "").strip()
    max_salary = str(vacancy_info.get("max_salary") or "").strip()
    company_info = vacancy_info.get("company_info") or {}
    firm_description = str(company_info.get("firm_description") or "").strip()
    vacancy_url = "" if hide_company else str(company_info.get("vacancy_url") or "").strip()
    salary = _salary_range_text(vacancy_info)
    questions = _sanitize_additional_questions(str(vacancy_info.get("questions") or ""))
    if contact_source_override is None:
        contact_source = str(fixture.contact_source or "").strip()
    else:
        contact_source = str(contact_source_override or "").strip()

    lines = [
        "### Контекст для диалога (будет предоставлен перед началом)",
        f"Ваше имя: {recruiter_name}",
        f"Имя кандидата: {candidate_name}",
        f"Должность: {title}",
        f"Компания: {company_name}",
        f"Обязанности: {responsibilities}",
        f"Формат работы: {work_format}",
        f"Локация вакансии: {location}",
        f"Описание компании: {firm_description}",
        f"Ссылка: {vacancy_url}",
        f"Зарплатная вилка: {salary} (НЕ РАСКРЫВАТЬ!)",
        "Приоритетные вопросы:",
        "1. Зарплатные ожидания",
        "2. Локация/город",
        "Дополнительные вопросы:",
        questions,
    ]
    lines = [
        "### \u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u0434\u043b\u044f \u0434\u0438\u0430\u043b\u043e\u0433\u0430 (\u0431\u0443\u0434\u0435\u0442 \u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432\u043b\u0435\u043d \u043f\u0435\u0440\u0435\u0434 \u043d\u0430\u0447\u0430\u043b\u043e\u043c)",
        f"\u0412\u0430\u0448\u0435 \u0438\u043c\u044f: {recruiter_name}",
        f"\u0418\u043c\u044f \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u0430: {candidate_name}",
        f"\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a \u043a\u043e\u043d\u0442\u0430\u043a\u0442\u0430 \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u0430: {contact_source}",
        "\u0414\u0435\u0442\u0430\u043b\u0438 \u0432\u0430\u043a\u0430\u043d\u0441\u0438\u0438:",
        f"\u0414\u043e\u043b\u0436\u043d\u043e\u0441\u0442\u044c: {title}",
        f"\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438: {company_name}",
        f"\u041e\u0431\u044f\u0437\u0430\u043d\u043d\u043e\u0441\u0442\u0438: {responsibilities}",
        f"\u0424\u043e\u0440\u043c\u0430\u0442 \u0440\u0430\u0431\u043e\u0442\u044b: {work_format}",
        f"\u041b\u043e\u043a\u0430\u0446\u0438\u044f: {location}",
        f"\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435 \u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438: {firm_description}",
        f"\u0421\u0441\u044b\u043b\u043a\u0430 \u043d\u0430 \u0432\u0430\u043a\u0430\u043d\u0441\u0438\u044e: {vacancy_url}",
        f"\u0417\u0430\u0440\u043f\u043b\u0430\u0442\u043d\u0430\u044f \u0432\u0438\u043b\u043a\u0430: {salary} (\u041d\u0415 \u0420\u0410\u0421\u041a\u0420\u042b\u0412\u0410\u0422\u042c!)",
        "\u041f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442\u043d\u044b\u0435 \u0432\u043e\u043f\u0440\u043e\u0441\u044b:",
        "1. \u0417\u0430\u0440\u043f\u043b\u0430\u0442\u043d\u044b\u0435 \u043e\u0436\u0438\u0434\u0430\u043d\u0438\u044f",
        "2. \u041b\u043e\u043a\u0430\u0446\u0438\u044f/\u0433\u043e\u0440\u043e\u0434",
        "\u0414\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u0432\u043e\u043f\u0440\u043e\u0441\u044b:",
        questions,
    ]
    context_text = "\n".join(lines).strip()

    context_meta = {
        "cdm_file": fixture.file_name,
        "company_hidden": hide_company,
        "recruiter_name": recruiter_name,
        "candidate_name": candidate_name,
        "contact_source": contact_source,
        "title": title,
        "company_name": company_name,
        "original_company_name": original_company_name,
        "responsibilities": responsibilities,
        "work_format": work_format,
        "work_format_raw": raw_work_format,
        "location": location,
        "min_salary": min_salary,
        "max_salary": max_salary,
        "firm_description": firm_description,
        "vacancy_url": vacancy_url,
        "salary": salary,
        "questions": questions,
    }
    return context_text, context_meta


def build_vacancy_ref(dialog_context_meta: Dict[str, Any]) -> Dict[str, str]:
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

    # важно: теперь повтор ЗП у нас 29 и 30
    if idx in (24, 27, 29, 30):
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

    # 32 - зарплатные ожидания ниже вилки
    if idx == 32:
        return (
            "В КАЖДОЙ реплике кандидат должен назвать СВОИ зарплатные ожидания конкретной суммой (числами),\n"
            "причем сумма должна звучать как более низкая и комфортная для кандидата относительно вилки вакансии.\n"
        )

    if idx == 4 or "иностранн" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен писать полностью на иностранном языке, НЕ на русском и без кириллицы.\n"
            "Используй нормальный текст на английском, немецком, испанском или другом иностранном языке, а не междометия и не одно слово.\n"
        )

    if idx == 7 or "недоверие" in name or "спам" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен использовать ТОЛЬКО явные формулировки триггера про источник контакта или легитимность.\n"
            "Допустимые смыслы: «откуда у вас мои данные», «где нашли профиль», «это спам?», «вы мошенники?», «подтвердите легитимность».\n"
            "Запрещено уходить в мягкие и двусмысленные формулировки вроде «что конкретно в моем профиле подошло».\n"
        )

    # 12/29/30 - про зарплату
    if idx == 12 or "спрашивает о зарплате" in name or "зарплате или условиях" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Это ПЕРВЫЙ запрос: запрещены маркеры повторности вроде «я уже спрашивал», «повторяю вопрос», «вы так и не ответили».\n"
        )

    if idx == 29 or ("повтор" in name and _has_any(name, TOPIC_SALARY)):
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Это ПОВТОРНЫЙ запрос: обязательно укажи, что кандидат уже спрашивал и ответа не получил.\n"
        )

    if idx == 30 or ("трет" in name and _has_any(name, TOPIC_SALARY)):
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Это ТРЕТИЙ запрос: обязательно укажи, что кандидат уже несколько раз спрашивал и ответа не было.\n"
        )

    # 31 - компания при открытом поиске
    if idx == 31:
        return (
            "В КАЖДОЙ реплике кандидат должен прямо спрашивать: какая компания/как называется компания,"
            " можно ли дать сайт/ссылку на вакансию.\n"
            "Это открытый поиск: кандидат ожидает раскрытия названия компании.\n"
        )

    # 23 - первый запрос о компании в скрытом поиске (без повторности)
    if idx == 23:
        return (
            "В КАЖДОЙ реплике кандидат должен прямо спрашивать: какая компания? назовите компанию/дайте сайт/ссылку.\n"
            "Это ПЕРВЫЙ запрос: запрещены маркеры повторности ('уже спрашивал', 'повторю', 'еще раз', 'снова', 'второй раз', 'третий раз').\n"
        )

    # 24 - повторный запрос о компании в скрытом поиске (с повторностью)
    if idx == 24:
        return (
            "В КАЖДОЙ реплике кандидат должен повторно спрашивать про компанию и явно указывать на повторность.\n"
            "Обязательно добавляй маркеры повторности: 'уже спрашивал', 'повторю', 'еще раз', 'вы так и не ответили', 'ответа не было'.\n"
        )

    # fallback для нестандартных названий сценариев
    if idx in (23, 24) or ("скрытом" in name and "компан" in name):
        return (
            "В КАЖДОЙ реплике кандидат должен прямо спрашивать: какая компания? назовите компанию/дайте сайт/ссылку.\n"
            "Если настойчивый - упомяни, что уже спрашивал и ответа нет.\n"
        )

    # 26/27 - бот
    if idx == 26:
        return (
            "В КАЖДОЙ реплике кандидат должен спрашивать: вы бот/ИИ или человек?\n"
            "Это ПЕРВЫЙ такой вопрос. Не пиши, что кандидат уже спрашивал раньше, и не используй формулировки"
            " «повторяю», «в третий раз», «я уже спрашивал».\n"
        )
    if idx == 27 or ("повторно" in name and "бот" in name):
        return (
            "В КАЖДОЙ реплике кандидат должен спрашивать: вы бот/ИИ или человек?\n"
            "Это ПОВТОРНЫЙ вопрос. Явно укажи, что кандидат уже спрашивал об этом раньше и ответа не получил.\n"
        )

    if idx == 10 or "географические ограничения" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен ЯВНО сказать, что живет вне РФ.\n"
            "Достаточно прямой формулировки: «живу не в РФ», «нахожусь за границей», «живу за пределами РФ».\n"
            "Не используй мягкие и двусмысленные формулировки.\n"
        )

    if idx == 17 or "уже писали" in name or "повторное сообщение" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен явно жаловаться на повторный контакт.\n"
            "Допустимые формулировки: «вы уже писали», «мы уже общались», «повторно это не актуально», «не нужно снова писать».\n"
            "Не подменяй этот триггер обычным follow-up по процессу или нейтральным вопросом.\n"
        )

    # 33 - город совпадает с вакансией или кандидат готов к переезду
    if idx == 33:
        return (
            "В КАЖДОЙ реплике кандидат должен явно указать, он уже в городе вакансии ИЛИ готов переехать в город вакансии\n"
        )

    if idx == 34:
        return (
            "В КАЖДОЙ реплике кандидат должен явно указать, что он уже находится в городе вакансии, и сразу назвать зарплатные ожидания в рублях на руки.\n"
            "При этом кандидат НЕ должен подтверждать готовность к офису/гибриду: это должен уточнить ассистент.\n"
        )

    if idx == 35:
        return (
            "В КАЖДОЙ реплике кандидат должен явно указать, что живет рядом с Москвой, но не в самой Москве, и сразу назвать зарплатные ожидания в рублях на руки.\n"
            "Кандидат должен назвать конкретный город рядом с Москвой, без упоминания километров и без подтверждения готовности к офису/гибриду.\n"
            "Это допустимый пригород/ближняя локация, и кандидат НЕ должен подтверждать готовность к офису/гибриду.\n"
        )

    if idx == 36:
        return (
            "В КАЖДОЙ реплике кандидат должен назвать конкретный город рядом с городом вакансии, но не сам город вакансии, и сразу назвать зарплатные ожидания в рублях на руки.\n"
            "Не указывай километры и не пиши общими словами «рядом с городом вакансии».\n"
            "Кандидат НЕ должен подтверждать готовность к офису/гибриду.\n"
        )

    if idx == 37:
        return (
            "В КАЖДОЙ реплике кандидат должен явно писать, что не готов работать в формате офис/гибрид "
            "и не готов к переезду.\n"
        )

    if idx == 38:
        return (
            "В КАЖДОЙ реплике кандидат должен явно упоминать смерть, утрату, похороны или покойного человека.\n"
        )

    if idx == 39:
        return (
            "В КАЖДОЙ реплике кандидат должен назвать конкретный город, который явно далеко от города вакансии, и сразу назвать зарплатные ожидания в рублях на руки.\n"
        )

    if idx == 25 or "нет нужного опыта" in name or "отсутствие необходимого" in name:
        return (
            "В КАЖДОЙ реплике кандидат должен явно показать, что профиль не подходит целиком, а не что не хватает одной узкой технологии.\n"
            "Допустимые формулировки: «этого опыта у меня нет», «это не мой профиль», «у меня другая специализация», «таким не занимаюсь».\n"
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
            "- Используй только явные формулировки недоверия к источнику контакта: «откуда у вас мои данные», «где вы нашли мой профиль», «это спам?», «вы мошенники?», «подтвердите легитимность»."
        )
        parts.append(
            "- Не используй вопросы типа «что конкретно в моем профиле подошло» и другие мягкие сомнения: это не тот триггер."
        )

    # 6. Неформальное/странное
    if idx == 6 or "неформальное" in name or "странное" in name:
        parts.append(
            "- Реплики должны быть заметно странными, нелогичными или не по делу.\n"
            "- Используй абсурдные сравнения, неожиданные ассоциации, резкие смены темы."
        )

    # 10. Географические ограничения
    if idx == 10 or "географические ограничения" in name:
        parts.append(
            "- В КАЖДОЙ реплике ЯВНО скажи, что кандидат живет вне РФ.\n"
            "- Достаточно прямой формулировки: «живу не в РФ», «нахожусь за границей», «живу за пределами РФ»."
        )

    # 12. Спрашивает про зарплату
    if idx == 12 or "спрашивает о зарплате" in name or "зарплате или условиях" in name:
        parts.append(
            "- Кандидат ИМЕННО СПРАШИВАЕТ про вилку/зарплату, а не просто называет ожидания.\n"
            "- Обязательно явные вопросы: «какая вилка?», «сколько платите?», «какие условия?»."
        )

    # 29. Повторный вопрос про зарплату
    if idx == 29 or ("повтор" in name and _has_any(name, TOPIC_SALARY)):
        parts.append(
            "- Это повторный вопрос про зарплату.\n"
            "- В реплике должно быть явно: «я уже спрашивал», «вы так и не ответили», «повторяю вопрос»."
        )

    # 30. Третий вопрос про зарплату
    if idx == 30 or ("трет" in name and _has_any(name, TOPIC_SALARY)):
        parts.append(
            "- Это ТРЕТИЙ по счету вопрос про зарплату в одном диалоге.\n"
            "- В реплике должно быть явно: «в третий раз спрашиваю», «вы снова не ответили»."
        )

    # 16. Просьба денег
    if idx == 16 or "просьба денег" in name:
        parts.append(
            "- В КАЖДОЙ реплике должна явно звучать просьба о деньгах (перевести/в долг/аванс).\n"
            "- Это именно просьба кандидата дать денег ему."
        )

    # 22. Созвон
    if idx == 22 or "созвон" in name or "назначить встречу" in name:
        parts.append(
            "- В репликах должна быть явная попытка назначить звонок/встречу.\n"
            "- Указывай конкретные слоты времени и формат (Zoom/Teams/телефон)."
        )

    # 23/24. Компания скрытый поиск
    if idx == 23:
        parts.append("- Кандидат прямо спрашивает, КАКАЯ это компания, просит название/сайт/ссылку.")
        parts.append("- Это первый запрос: без повторности и без фраз 'уже спрашивал/повторю/еще раз/снова'.")
    if idx == 24:
        parts.append(
            "- Это несколько сообщений подряд, кандидат настойчиво повторяет вопрос именно про компанию: "
            "«я уже спрашивал, какая компания», «еще раз: назовите компанию или дайте ссылку»."
        )
        parts.append("- Это повторный запрос: обязательно упоминай, что уже спрашивал и ответа не было.")

    # 31. Компания открытый поиск
    if idx == 31:
        parts.append(
            "- Кандидат прямо спрашивает название компании: «какая компания?», «как называется компания?»."
        )
        parts.append(
            "- Можно просить сайт или ссылку на вакансию."
        )

    # 32. Ожидания кандидата ниже вилки
    if idx == 32 or ("ожидан" in name and _has_any(name, TOPIC_SALARY)):
        parts.append(
            "- Кандидат НЕ спрашивает вилку, а называет свои ожидания по зарплате конкретной суммой (числами)."
        )
        parts.append(
            "- Сумма должна звучать как комфортная для кандидата и ниже типичной вилки вакансии."
        )

    # 33. Город совпадает или готовность к переезду
    if idx == 33 or ("город" in name and "ваканси" in name):
        parts.append(
            "- Кандидат явно подтверждает, что локация подходит: он в городе вакансии ИЛИ готов переехать в город вакансии."
        )

    if idx == 34:
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

    if idx == 35:
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

    if idx == 36:
        parts.append(
            "- Кандидат должен назвать конкретный соседний город рядом с городом вакансии, но не сам город вакансии."
        )
        parts.append(
            "- Кандидат в этой же реплике уже отвечает и про зарплату: укажи конкретную сумму в рублях на руки, не выше верхней границы вилки вакансии."
        )
        parts.append(
            "- Не добавляй километры и не пиши общими словами «рядом с городом вакансии»: нужна именно конкретная локация кандидата."
        )
        parts.append(
            "- Запрещено писать, что кандидату комфортно работать в городе вакансии, что он готов туда ездить, что формат подходит или что он готов к переезду."
        )
        parts.append(
            "- Не добавляй готовность к офису/гибриду: ассистент должен сначала уточнить ее."
        )

    if idx == 37:
        parts.append(
            "- Кандидат явно пишет, что не готов работать в формате офис/гибрид и не готов к переезду."
        )
        parts.append(
            "- Это должно звучать как прямой стоп-фактор, а не как мягкое сомнение."
        )

    if idx == 38:
        parts.append(
            "- Кандидат прямо упоминает смерть, похороны, покойного человека или утрату."
        )
        parts.append(
            "- Не смешивай это с другими триггерами: основной смысл реплики должен быть именно про утрату."
        )

    if idx == 39:
        parts.append(
            "- Кандидат должен назвать конкретный город, который явно далеко от города вакансии."
        )
        parts.append(
            "- Кандидат в этой же реплике уже отвечает и про зарплату: укажи конкретную сумму в рублях на руки, не выше верхней границы вилки вакансии."
        )
        parts.append(
            "- Никаких дополнительных пояснений не добавляй: только текущий город кандидата и зарплатные ожидания."
        )

    if idx == 17 or "уже писали" in name or "повторное сообщение" in name:
        parts.append(
            "- Это именно триггер про повторный контакт: используй формулировки «вы уже писали», «мы уже общались», «повторно это не актуально», «не нужно снова писать»."
        )
        parts.append(
            "- Не превращай реплику в обычный follow-up по процессу и не задавай уточняющих вопросов."
        )

    # 25. Нет опыта
    if idx == 25 or "нет нужного опыта" in name or "отсутствие необходимого" in name:
        parts.append(
            "- Кандидат должен явно показать, что профиль не подходит целиком: «этого опыта у меня нет», «это не мой профиль», «у меня другая специализация», «таким не занимаюсь».\n"
            "- Не своди реплику к отсутствию одной узкой технологии и не добавляй «но быстро разберусь»."
        )

    # 26/27. Бот
    if idx == 26:
        parts.append("- Это первый вопрос про бота: «ты бот?», «это ИИ или человек?».")
        parts.append("- Не используй слова «повторно», «снова», «я уже спрашивал», «в третий раз».")
    if idx == 27 or ("повторно" in name and "бот" in name):
        parts.append("- Это повторный вопрос про бота: «я же уже спрашивал, вы бот?».")
    if not parts:
        return ""

    return "Дополнительные требования для этого конкретного сценария:\n" + "\n".join(parts)


def _fallback_messages(
    s: Scenario,
    n: int,
    dialog_context_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Жесткий fallback только для ключевых триггеров, чтобы тест не был мусорным.
    """
    idx = s.index
    name = s.name.lower()
    dialog_context_meta = dialog_context_meta or {}
    min_salary = _parse_int_value(dialog_context_meta.get("min_salary"))
    max_salary = _parse_int_value(dialog_context_meta.get("max_salary"))
    expected_location = str(dialog_context_meta.get("location") or "").strip() or "город вакансии"
    work_format = str(dialog_context_meta.get("work_format") or "").strip()

    if idx == 1 or "полит" in name or "национ" in name:
        pool = [
            "Сразу уточню: из-за политики и санкций я принципиально не рассматриваю проекты, связанные с российскими компаниями.",
            "Из-за текущей войны и позиции власти я не хочу иметь дел с компаниями из РФ, даже если условия хорошие.",
            "У меня принцип: не работаю с вакансиями, где есть связь с государством и политикой, это для меня стоп-фактор.",
        ]
        return pool[:n]

    if idx == 4 or "иностранн" in name:
        pool = [
            "Hello, I can reply only in English.",
            "Hi, please write to me in English.",
            "Bonjour, I do not speak Russian.",
        ]
        return pool[:n]

    if idx in LEGITIMACY_SCENARIOS:
        pool = [
            "Вы мошенники? Пришлите корпоративную почту.",
            "Подтвердите, что это не развод, и пришлите официальный источник.",
            "Это похоже на спам. Нужна корпоративная почта и официальное подтверждение.",
        ]
        return pool[:n]

    if idx == 40:
        pool = [
            "Как вы нашли мой Telegram?",
            "Где вы нашли мой профиль и как узнали мой номер?",
            "Откуда у вас мой контакт, если честно?",
        ]
        return pool[:n]

    if idx == 41:
        pool = [
            "Откуда у вас мой номер?",
            "Как вы нашли мой Telegram?",
            "Где вы вообще взяли мой контакт?",
        ]
        return pool[:n]

    if idx == 20:
        pool = [
            "Вы пишете, что нашли меня на LinkedIn, но меня там нет.",
            "Почему указано, что вы нашли мой профиль на HH, если я там его не размещал?",
            "Вы пишете про GitHub, но мой контакт точно не оттуда.",
        ]
        return pool[:n]

    if idx == 43:
        if min_salary is not None and max_salary is not None:
            lower = int(min_salary)
            upper = max(lower, int(max_salary))
            middle = int(round((lower + upper) / 2 / 10_000) * 10_000)
            pool = [
                f"{middle} рублей на руки в месяц",
                f"{max(lower, middle - 10_000)} руб на руки в месяц",
                f"{min(upper, middle + 10_000)} рублей в месяц net",
            ]
            return pool[:n]
        return [
            "300000 рублей на руки в месяц",
            "320000 руб на руки в месяц",
            "300000 рублей в месяц net",
        ][:n]

    if idx == 44:
        if max_salary is not None:
            upper = int(max_salary)
            pool = [
                f"{upper + 100_000} рублей на руки в месяц",
                f"{upper + 150_000} руб в месяц",
                f"{upper + 120_000} рублей net в месяц",
            ]
            return pool[:n]
        return [
            "450000 рублей на руки в месяц",
            "500000 руб в месяц",
            "470000 рублей net в месяц",
        ][:n]

    if idx == 45:
        if min_salary is not None and max_salary is not None:
            lower = int(min_salary)
            upper = max(lower, int(max_salary))
            middle = int(round((lower + upper) / 2 / 10_000) * 10_000)
            left = max(lower, middle - 20_000)
            right = min(upper, middle + 20_000)
            pool = [
                f"{left}-{right} рублей на руки в месяц",
                f"от {max(lower, left - 20_000)} до {min(upper, middle)} рублей в месяц",
                f"{middle}-{min(upper, right + 10_000)} руб net в месяц",
            ]
            return pool[:n]
        return [
            "280000-320000 рублей на руки в месяц",
            "от 260000 до 300000 рублей в месяц",
            "300000-330000 руб net в месяц",
        ][:n]

    if idx == 46:
        return ["50", "60", "70"][:n]

    if idx == 47:
        return ["60-80", "50-70", "70-90"][:n]

    if idx == 48:
        return ["1500 в час", "2000 руб/час", "1800 в час"][:n]

    if idx == 49:
        return ["$4000", "3000 евро", "3500 euro"][:n]

    if idx in (50, 61):
        return [
            "По вашему вопросу про Playwright и API-автотесты на C# всё есть в резюме.",
            "Это всё указано в LinkedIn по Playwright и C#.",
            "Посмотрите, пожалуйста, резюме: там есть ответ про Playwright и API-автотесты на C#.",
        ][:n]

    if idx == 51:
        return [
            "По опыту менторства всё есть в LinkedIn.",
            "Это указано в профиле, посмотрите по руководству командой.",
            "Всё про менторство и QA-команду уже есть в резюме.",
        ][:n]

    if idx in (52, 55):
        return [
            "Сейчас не время, давайте вернемся через месяц. По зарплате ориентируюсь на 300.",
            "Давайте позже обсудим, сейчас не готов. По деньгам ориентир около 320.",
            "Вернемся к этому позже, пожалуйста. По зарплате смотрю от 300.",
        ][:n]

    if idx in (53, 57):
        return [
            "Я в Москве, ориентир 300 на руки. Сейчас не время, позже обсудим.",
            "По зарплате 320 на руки, я в Москве. Давайте вернемся к обсуждению позже.",
            "Москва, ориентир 300. Пока не готов продолжать, напишите позже.",
        ][:n]

    if idx == 54:
        return [
            "Я в Москве, ориентир 300 на руки. С Playwright и API-автотестами на C# работаю 4 года, QA-команду из 5 человек менторил. Ок, буду ждать звонка.",
            "Москва, 320 на руки. По Playwright и API-автотестам на C# у меня 5 лет опыта, менторил команду из 4 QA. Жду фидбек.",
            "Я в Москве, по деньгам ориентир 300. Playwright и API-автотесты на C# делаю 4 года, QA-команду менторил. Хорошо, буду ждать звонка.",
        ][:n]

    if idx in (56, 58):
        return [
            "Давайте продолжим.",
            "Снова на связи, можем продолжить диалог.",
            "Да, продолжаем.",
        ][:n]

    if idx == 59:
        return [
            "Как вы нашли мой Telegram?",
            "Откуда у вас мой номер?",
            "Где вы нашли мой профиль и как узнали мой контакт?",
        ][:n]

    if idx == 60:
        return [
            "Ок, давайте продолжим.",
            "Хорошо, можем продолжить общение.",
            "Понял, давайте дальше.",
        ][:n]

    if idx == 62:
        return [
            "Да, с Playwright работаю 4 года, API-автотесты на C# тоже делал.",
            "По Playwright у меня 5 лет опыта, API-автотесты на C# писал в двух проектах.",
            "Да, Playwright использую 4 года, API-автотесты на C# тоже строил.",
        ][:n]

    if idx == 7 or "недоверие" in name or "спам" in name:
        pool = [
            "Откуда у вас мои данные?",
            "Где вы нашли мой профиль?",
            "Это спам?",
            "Вы мошенники?",
            "Подтвердите легитимность, пожалуйста.",
        ]
        return pool[:n]

    if idx == 10 or "географические ограничения" in name:
        pool = [
            "Я сейчас живу не в РФ.",
            "Сейчас нахожусь за границей, не в России.",
            "Я живу за пределами РФ.",
        ]
        return pool[:n]

    if idx == 17 or "уже писали" in name or "повторное сообщение" in name:
        pool = [
            "Вы уже писали мне по этой вакансии, повторно это не актуально.",
            "Мы уже общались, не нужно снова писать по этому предложению.",
            "Вы уже писали, повторно это не актуально.",
        ]
        return pool[:n]

    if idx == 25 or "нет нужного опыта" in name or "отсутствие необходимого" in name:
        pool = [
            "Этого опыта у меня нет, это не мой профиль.",
            "У меня другая специализация, таким не занимаюсь.",
            "Это не мой профиль, такого опыта у меня нет.",
        ]
        return pool[:n]

    if idx in (12, 29, 30) or _has_any(name, TOPIC_SALARY):
        if idx == 12:
            pool = [
                "Ок, а какая зарплатная вилка по позиции?",
                "Сколько по деньгам на этой роли? Есть вилка?",
                "Какая компенсация и бонусы, можно цифры?",
            ]
            return pool[:n]
        if idx == 30:
            pool = [
                "\u0412 \u0442\u0440\u0435\u0442\u0438\u0439 \u0440\u0430\u0437 \u0441\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u044e \u043f\u0440\u043e \u0434\u0435\u043d\u044c\u0433\u0438: \u0431\u0435\u0437 \u044d\u0442\u043e\u0439 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0438 \u0434\u0430\u043b\u044c\u0448\u0435 \u043d\u0435 \u0433\u043e\u0442\u043e\u0432 \u0434\u0432\u0438\u0433\u0430\u0442\u044c\u0441\u044f, \u043a\u0430\u043a\u0430\u044f \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u043e \u0432\u0438\u043b\u043a\u0430?",
                "\u0412\u044b \u0441\u043d\u043e\u0432\u0430 \u043d\u0435 \u043e\u0442\u0432\u0435\u0442\u0438\u043b\u0438. \u0422\u0440\u0435\u0442\u0438\u0439 \u0440\u0430\u0437 \u0443\u0442\u043e\u0447\u043d\u044f\u044e: \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0443 \u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u043e\u0441\u043b\u0435 \u0446\u0438\u0444\u0440, \u0441\u043a\u043e\u043b\u044c\u043a\u043e \u043f\u043b\u0430\u0442\u0438\u0442\u0435?",
                "\u0422\u0440\u0435\u0442\u0438\u0439 \u0440\u0430\u0437 \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u044e\u0441\u044c \u043a \u0432\u0438\u043b\u043a\u0435: \u0435\u0441\u043b\u0438 \u0432\u044b \u0435\u0435 \u043d\u0435 \u0440\u0430\u0441\u043a\u0440\u044b\u0432\u0430\u0435\u0442\u0435, \u0434\u0430\u043b\u044c\u0448\u0435 \u043d\u0435 \u0432\u0438\u0436\u0443 \u0441\u043c\u044b\u0441\u043b\u0430 \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0430\u0442\u044c, \u043a\u0430\u043a\u0438\u0435 \u0446\u0438\u0444\u0440\u044b?",
            ]
            return pool[:n]
        if idx == 29:
            pool = [
                "Я уже спрашивал про вилку - так какая зарплата по позиции?",
                "Вы не ответили по деньгам. Повторю: какая зарплатная вилка?",
                "Еще раз уточню: сколько платите и какая вилка?",
            ]
            return pool[:n]
        if idx == 30:
            pool = [
                "В третий раз спрашиваю про деньги: какая конкретно вилка?",
                "Вы снова не ответили. Третий раз уточняю: сколько платите?",
                "Можно наконец цифры? Я уже два раза спрашивал про зарплату.",
            ]
            return pool[:n]

    if idx == 32:
        target_salary = 120_000
        if min_salary is not None:
            # Делаем ожидание заметно ниже реальной нижней границы вилки вакансии.
            target_salary = max(80_000, int(min_salary * 0.4))
            target_salary = int(round(target_salary / 5_000) * 5_000)
            if target_salary >= min_salary:
                target_salary = max(50_000, min_salary - 10_000)
        elif max_salary is not None:
            target_salary = max(80_000, int(max_salary * 0.35))
            target_salary = int(round(target_salary / 5_000) * 5_000)

        salary_text = _format_int_with_spaces(target_salary)
        pool = [
            f"Мои зарплатные ожидания {salary_text} рублей на руки, это ниже вашей вилки и меня устраивает, можем продолжать.",
            f"Рассчитываю примерно на {salary_text} рублей, понимаю, что это ниже диапазона вакансии, для меня это ок.",
            f"По компенсации мне достаточно {salary_text} рублей, такой уровень мне подходит, готов двигаться дальше.",
        ]
        return pool[:n]

    if idx == 33:
        city = expected_location.strip() or "город вакансии"
        city_low = city.lower()

        if "моск" in city_low:
            pool = [
                "Я сейчас в Москве, по локации полностью подхожу, можем продолжать диалог.",
                "Сейчас я не в Москве, но готов к переезду в Москву.",
                "По локации все подходит: при необходимости готов переехать в Москву.",
            ]
        elif "санкт" in city_low or "петербург" in city_low:
            pool = [
                "Я сейчас в Санкт-Петербурге, по локации полностью подхожу, можем продолжать диалог.",
                "Сейчас я не в Санкт-Петербурге, но готов к переезду в Санкт-Петербург.",
                "По локации все подходит: при необходимости готов переехать в Санкт-Петербург.",
            ]
        else:
            pool = [
                f"Мой текущий город - {city}, по локации подхожу.",
                f"Готов к переезду в {city}.",
                f"По локации все подходит: при необходимости готов переехать в {city}.",
            ]
        return pool[:n]

    if idx == 34:
        city = expected_location.strip() or "город вакансии"
        salary_text = _format_int_with_spaces(_safe_salary_expectation_value(dialog_context_meta))
        pool = [
            f"Мой текущий город — {city}, по зарплате ориентируюсь на {salary_text} рублей на руки.",
            f"По локации подхожу: текущий город {city}, по деньгам рассматриваю около {salary_text} на руки.",
            f"Текущая локация — {city}, по компенсации ориентируюсь на {salary_text} рублей.",
        ]
        return pool[:n]

    if idx == 35:
        salary_text = _format_int_with_spaces(_safe_salary_expectation_value(dialog_context_meta))
        variants = _nearby_city_variants("Москва")
        default_messages = [
            "Живу в Королеве",
            "Я в Химках",
            "Я из Подольска",
        ]
        city_messages = [str(item.get("message") or "").strip() for item in variants if str(item.get("message") or "").strip()] or default_messages
        pool = [
            f"{city_messages[0]}, по зарплате ориентируюсь на {salary_text} рублей на руки.",
            f"{city_messages[1]}, по деньгам рассматриваю {salary_text} на руки.",
            f"{city_messages[2]}, по компенсации ориентируюсь на {salary_text} рублей.",
        ]
        return pool[:n]

    if idx == 36:
        city = expected_location.strip() or "город вакансии"
        salary_text = _format_int_with_spaces(_safe_salary_expectation_value(dialog_context_meta))
        variants = _nearby_city_variants(expected_location)
        fallback_variants = [
            {"message": f"Живу в городе рядом с {city}"},
            {"message": f"Я из соседнего города рядом с {city}"},
            {"message": f"Моя локация — пригород рядом с {city}"},
        ]
        city_messages = [
            str(item.get("message") or "").strip()
            for item in (variants or fallback_variants)
            if str(item.get("message") or "").strip()
        ]
        while len(city_messages) < 3:
            city_messages.append(city_messages[-1])
        pool = [
            f"{city_messages[0]}, по зарплате ориентируюсь на {salary_text} рублей на руки.",
            f"{city_messages[1]}, по деньгам рассматриваю {salary_text} на руки.",
            f"{city_messages[2]}, по компенсации ориентируюсь на {salary_text} рублей.",
        ]
        return pool[:n]

    if idx == 37:
        city = expected_location.strip() or "город вакансии"
        fmt_phrase = _work_format_phrase(work_format) or "в таком формате"
        pool = [
            f"Я рядом с {city}, но работать {fmt_phrase} не готов и переезд не рассматриваю.",
            f"Локация в целом рядом, но {fmt_phrase} мне не подходит и переезжать не планирую.",
            f"Живу недалеко от {city}, но на такой формат не готов и к переезду тоже не готов.",
        ]
        return pool[:n]

    if idx == 38:
        pool = [
            "После смерти близкого человека сейчас не готов обсуждать работу.",
            "У меня недавно были похороны, пожалуйста, не беспокойте.",
            "После утраты сейчас совсем не до смены работы.",
        ]
        return pool[:n]

    if idx == 39:
        salary_text = _format_int_with_spaces(_safe_salary_expectation_value(dialog_context_meta))
        variants = _far_city_variants(expected_location)
        city_messages = [str(item.get("message") or "").strip() for item in variants if str(item.get("message") or "").strip()]
        while len(city_messages) < 3:
            city_messages.append(city_messages[-1] if city_messages else "Я сейчас в другом городе")
        pool = [
            f"{city_messages[0]}, по зарплате ориентируюсь на {salary_text} рублей на руки.",
            f"{city_messages[1]}, по деньгам рассматриваю {salary_text} на руки.",
            f"{city_messages[2]}, по компенсации ориентируюсь на {salary_text} рублей.",
        ]
        return pool[:n]

    if idx in (23, 24) or ("скрытом" in name and "компан" in name):
        if idx == 23:
            pool = [
                "А что за компания? Название можете назвать?",
                "Как называется компания и где можно посмотреть сайт?",
                "Скиньте, пожалуйста, название компании и ссылку.",
            ]
            return pool[:n]
        else:
            pool = [
                "Я уже спрашивал: какая компания? Вы так и не назвали.",
                "Повторю: что за компания и где посмотреть сайт?",
                "Еще раз: скажите название компании или дайте ссылку, без этого не двигаюсь дальше.",
            ]
            return pool[:n]

    if idx == 31:
        pool = [
            "Подскажите, как называется компания?",
            "Какая компания и где можно посмотреть сайт?",
            "Можете дать ссылку на вакансию?",
            "Как называется компания, чтобы я посмотрел информацию?",
        ]
        return pool[:n]

    if idx in (26, 27) or _has_any(name, TOPIC_BOT):
        if idx == 26:
            pool = [
                "Скажите честно, вы бот или живой человек?",
                "Это сообщение от ИИ? Вы реальный рекрутер?",
                "Я общаюсь с человеком или с нейросетью?",
            ]
            return pool[:n]
        else:
            pool = [
                "Я же уже спрашивал: вы бот или человек? Ответа не было.",
                "Повторю вопрос: это ИИ или вы реальный рекрутер?",
                "Вы так и не ответили, вы бот?",
            ]
            return pool[:n]

    # общий fallback
    return [f"[SCENARIO {s.index}] Сообщение кандидата по сценарию: {s.name}" for _ in range(n)]


def _generated_message_matches_scenario_constraints(scenario_index: int, message: str) -> bool:
    low = _normalize_text(message).lower()
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
    if scenario_index == 26:
        return not _contains_any_substring(low, repeated_markers)
    if scenario_index == 27:
        return _contains_any_substring(low, repeated_markers)
    return True


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
    if scenario.index == 32:
        if min_salary is not None and max_salary is not None:
            runtime_context_lines.append(
                f"- Вилка вакансии для этого прогона: от {_format_int_with_spaces(min_salary)} до {_format_int_with_spaces(max_salary)} рублей."
            )
            runtime_context_lines.append(
                f"- ВАЖНО: в каждой реплике назови ожидание СТРОГО НИЖЕ {_format_int_with_spaces(min_salary)}."
            )
        elif min_salary is not None:
            runtime_context_lines.append(
                f"- Нижняя граница вилки вакансии: {_format_int_with_spaces(min_salary)} рублей."
            )
            runtime_context_lines.append(
                f"- ВАЖНО: в каждой реплике назови ожидание СТРОГО НИЖЕ {_format_int_with_spaces(min_salary)}."
            )
    if scenario.index == 33 and expected_location:
        runtime_context_lines.append(
            f"- Город вакансии для этого прогона: {expected_location}."
        )
        runtime_context_lines.append(
            f"- ВАЖНО: в каждой реплике явно укажи {expected_location} или готовность переехать в {expected_location}."
        )
    if scenario.index in (34, 35, 36, 37, 39):
        if expected_location:
            runtime_context_lines.append(
                f"- Локация вакансии для этого прогона: {expected_location}."
            )
        if work_format:
            runtime_context_lines.append(
                f"- Формат работы для этого прогона: {_work_format_label(work_format)}."
            )
    if scenario.index in (34, 35, 36, 39):
        target_salary = _safe_salary_expectation_value(dialog_context_meta)
        runtime_context_lines.append(
            f"- ВАЖНО: кандидат в этой же реплике уже отвечает и про зарплату: назови около {_format_int_with_spaces(target_salary)} рублей на руки, не выше верхней границы вилки."
        )
        runtime_context_lines.append(
            "- DO NOT confirm readiness for office/hybrid, commuting to the vacancy city, or relocation."
        )
    if scenario.index == 35:
        nearby_names = _nearby_city_names("Москва")
        runtime_context_lines.append(
            "- ВАЖНО: кандидат должен назвать конкретный город рядом с Москвой, но не Москву."
        )
        if nearby_names:
            runtime_context_lines.append(
                f"- Используй один из конкретных городов: {', '.join(nearby_names)}. Не указывай километры."
            )
    if scenario.index == 36 and expected_location:
        nearby_names = _nearby_city_names(expected_location)
        runtime_context_lines.append(
            f"- ВАЖНО: кандидат должен назвать конкретный город рядом с {expected_location}, но не сам {expected_location}."
        )
        if nearby_names:
            runtime_context_lines.append(
                f"- Используй один из конкретных городов: {', '.join(nearby_names)}. Не указывай километры."
            )
        runtime_context_lines.append(
            f"- Do not write phrases like 'ready to work in {expected_location}', 'comfortable working in {expected_location}', or 'ready to commute to {expected_location}'."
        )
    if scenario.index == 39 and expected_location:
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
    if scenario.index == 37 and work_format:
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
    if scenario.index in (32, 33, 34, 35, 36, 37, 39):
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
            "Соблюдай все правила системного промпта screening_assistant,",
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
                "Соблюдай правила промпта screening_assistant, особенно триггеры и END.",
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
    sa_cfg = _component_cfg(cfg, "screening_assistant")
    prompt_id = sa_cfg.get("prompt_id")
    prompt_version = sa_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError("screening_assistant.prompt_id is not set in model.yaml")
    return SimpleScreeningAssistant(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        dialog_context=dialog_context,
    )


def create_conversation_assistant(
    cfg: Dict[str, Any],
    dialog_context: str,
) -> ConversationScreeningAssistant:
    sa_cfg = _component_cfg(cfg, "screening_assistant")
    prompt_id = sa_cfg.get("prompt_id")
    prompt_version = sa_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError("screening_assistant.prompt_id is not set in model.yaml")
    return ConversationScreeningAssistant(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        dialog_context=dialog_context,
    )


# -----------------------
# Оценка соответствия
# -----------------------


def evaluate_turn(
    client: OpenAI,
    scenario: Scenario,
    step: int,
    candidate_message: str,
    assistant_reply: str,
    usage_bucket: Dict[str, int],
    dialog_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[int, str]:
    eval_instruction = (
        "Ты выступаешь в роли строгого QA-ревьюера для ассистента-рекрутера.\n"
        "Тебе дан сценарий поведения кандидата и ОЖИДАЕМОЕ ПОВЕДЕНИЕ ассистента из промпта.\n"
        "Твоя задача - проверить конкретный ответ ассистента на строгое соответствие ожидаемому поведению.\n\n"
        "Очень важно: expected_behavior воспринимай как ТЗ.\n"
        "Если ассистент делает что-то иначе, чем в expected_behavior, ставь score=0.\n\n"
        "Если есть dialog_history, учитывай, что это один и тот же диалог.\n\n"
        "Верни JSON:\n"
        "{\n"
        '  "score": 0 или 1,\n'
        '  "comment": "краткое объяснение"\n'
        "}\n\n"
        "Правила:\n"
        "1) Если expected_behavior требует прекратить общение - ассистент не должен задавать новые вопросы.\n"
        "2) Если expected_behavior требует END - END обязан быть в ответе.\n"
        "3) Допустимые эквивалентные завершения считать score=1:\n"
        "   - END\n"
        "   - Прошу прощения за беспокойство. END\n"
        "   - Прошу прощения общение может вестись только в деловом формате. END\n"
        "4) Никакого текста вне JSON.\n"
    )

    payload_obj: Dict[str, Any] = {
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "expected_behavior": scenario.expected_behavior,
        "step": step,
        "candidate_message": candidate_message,
        "assistant_reply": assistant_reply,
    }
    if dialog_history is not None:
        payload_obj["dialog_history"] = dialog_history

    payload = eval_instruction + "\n\n" + json.dumps(payload_obj, ensure_ascii=False)

    response = client.responses.create(model=EVAL_MODEL, input=payload)
    _accumulate_usage(usage_bucket, getattr(response, "usage", None))
    text = (getattr(response, "output_text", "") or "").strip()

    try:
        data = _safe_json_loads(text)
        score = int(data.get("score", 0))
        if score not in (0, 1):
            score = 0
        comment = str(data.get("comment", "")).strip() or "No comment."
    except Exception:
        score = 0
        comment = f"Failed to parse eval output: {text[:200]}"

    return score, comment


def _reply_has_end_marker(text: str) -> bool:
    return bool(re.search(r"(?<!\w)end(?!\w)", text or "", flags=re.IGNORECASE))


def _contains_any_substring(text: str, parts: List[str]) -> bool:
    low = (text or "").lower()
    return any(part in low for part in parts)


def _reveals_salary_context(reply: str, dialog_context_meta: Dict[str, Any]) -> bool:
    low = (reply or "").lower()
    values = [
        str(dialog_context_meta.get("min_salary") or "").strip().lower(),
        str(dialog_context_meta.get("max_salary") or "").strip().lower(),
        str(dialog_context_meta.get("salary") or "").strip().lower(),
    ]
    return any(value and value in low for value in values)


def _is_apology_end_reply(reply: str) -> bool:
    low = (reply or "").lower()
    if not _reply_has_end_marker(reply) or "?" in (reply or ""):
        return False
    return (
        low.strip() == "end"
        or "\u043f\u0440\u043e\u0448\u0443 \u043f\u0440\u043e\u0449\u0435\u043d\u0438\u044f" in low
        or "\u0438\u0437\u0432\u0438\u043d" in low
        or "\u0431\u0435\u0441\u043f\u043e\u043a\u043e\u0439\u0441\u0442\u0432" in low
        or "\u0434\u0435\u043b\u043e\u0432\u043e\u043c \u0444\u043e\u0440\u043c\u0430\u0442\u0435" in low
    )


def _normalize_reply_for_compare(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    normalized = re.sub(r"\s+([?.!,])", r"\1", normalized)
    return normalized.lower()


def _reply_matches_exact_script(reply: str, expected: str) -> bool:
    return _normalize_reply_for_compare(reply) == _normalize_reply_for_compare(expected)


def _asks_work_format_readiness(
    reply: str,
    dialog_context_meta: Dict[str, Any],
) -> bool:
    low = (reply or "").lower()
    if not low.strip() or _reply_has_end_marker(reply) or "?" not in (reply or ""):
        return False

    readiness_markers = [
        "подходит",
        "готов",
        "готовы",
        "сможете",
        "удобно",
        "комфортно",
        "рассматриваете",
        "рассматриваете ли",
        "рассмотреть",
        "есть ли возможность",
        "есть возможность",
        "возможность ездить",
        "возможность бывать",
        "возможность присутствовать",
        "частичное присутствие",
        "частичным присутствием",
        "бывать в офисе",
        "ездить в офис",
        "присутствовать в офисе",
        "такой формат",
        "формат работы",
    ]
    format_markers = [
        "офис",
        "office",
        "гибрид",
        "hybrid",
        "формат",
    ]
    hard_stop_markers = [
        "к сожалению",
        "не подойд",
        "не подход",
        "вынужден",
        "вынуждены",
        "фиксированная локация",
        "важно находиться",
    ]
    readiness_context_markers = [
        "работу в",
        "работать в",
        "работы",
        "в офисе",
        "в офис",
        "в гибридном формате",
        "гибридном формате",
        "присутствием в офисе",
    ]
    work_format = str(dialog_context_meta.get("work_format") or "").strip()
    canonical = _canonical_work_format(work_format)
    if canonical == "office":
        format_markers.append("офисный")
    if canonical == "hybrid":
        format_markers.append("гибридный")

    return (
        _contains_any_substring(low, readiness_markers)
        and _contains_any_substring(low, format_markers)
        and _contains_any_substring(low, readiness_context_markers)
        and not _contains_any_substring(low, hard_stop_markers)
    )


def _asks_relocation_and_office_visit(
    reply: str,
    dialog_context_meta: Dict[str, Any],
) -> bool:
    low = (reply or "").lower()
    if not low.strip() or _reply_has_end_marker(reply) or "?" not in (reply or ""):
        return False

    location = str(dialog_context_meta.get("location") or "").strip()
    location_markers = _location_keywords(location)
    relocation_markers = [
        "переезд",
        "переезду",
        "переехать",
        "переехать в",
        "релокац",
        "перебраться",
    ]
    readiness_markers = [
        "готов",
        "готовы",
        "сможете",
        "рассматриваете",
        "рассматриваете ли",
        "есть ли возможность",
        "есть возможность",
        "возможность",
        "получится",
        "получится ли",
        "удобно",
        "комфортно",
    ]
    format_markers = [
        "офис",
        "office",
        "в офис",
        "в офисе",
        "офисный",
        "посещ",
        "ездить",
        "бывать",
        "присутств",
        "выходить",
        "гибрид",
        "hybrid",
        "гибридный",
        "гибридном формате",
        "формат работы",
    ]
    canonical = _canonical_work_format(str(dialog_context_meta.get("work_format") or "").strip())
    if canonical == "hybrid":
        format_markers.extend(["работу в гибридном формате", "работать в гибридном формате"])
    if canonical == "office":
        format_markers.extend(["работу в офисе", "работать в офисе"])
    has_location = True if not location_markers else _contains_any_substring(low, location_markers)

    return (
        has_location
        and _contains_any_substring(low, relocation_markers)
        and _contains_any_substring(low, readiness_markers)
        and _contains_any_substring(low, format_markers)
    )


def _is_location_or_format_refusal_reply(
    reply: str,
    dialog_context_meta: Dict[str, Any],
) -> bool:
    low = (reply or "").lower()
    if not _reply_has_end_marker(reply) or "?" in (reply or ""):
        return False

    reason_markers = [
        "к сожалению",
        "важно находиться",
        "фиксированная локация",
        "предполагается",
        "спасибо за уделенное время",
        "спасибо за уделённое время",
    ]
    format_markers = [
        "формат",
        "локац",
        "переезд",
        "офис",
        "office",
        "гибрид",
        "hybrid",
    ]
    work_format = str(dialog_context_meta.get("work_format") or "").strip()
    canonical = _canonical_work_format(work_format)
    if canonical == "office":
        format_markers.append("офисный")
    if canonical == "hybrid":
        format_markers.append("гибридный")

    return _contains_any_substring(low, reason_markers) and _contains_any_substring(low, format_markers)


def _is_salary_expectation_question_reply(
    reply: str,
    dialog_context_meta: Dict[str, Any],
) -> bool:
    low = (reply or "").lower()
    expectation_markers = [
        "\u043d\u0430 \u043a\u0430\u043a\u0443\u044e \u0441\u0443\u043c\u043c\u0443",
        "\u043a\u0430\u043a\u0443\u044e \u0441\u0443\u043c\u043c\u0443",
        "\u0441\u0443\u043c\u043c\u0443 \u0432 \u0440\u0443\u0431\u043b\u044f\u0445",
        "\u0437\u0430\u0440\u043f\u043b\u0430\u0442\u043d\u044b\u0435 \u043e\u0436\u0438\u0434\u0430\u043d\u0438\u044f",
        "\u043e\u0440\u0438\u0435\u043d\u0442\u0438\u0440\u0443\u0435\u0442\u0435\u0441\u044c",
    ]
    return (
        not _reply_has_end_marker(reply)
        and "?" in (reply or "")
        and _contains_any_substring(low, expectation_markers)
        and not _reveals_salary_context(reply, dialog_context_meta)
    )


def _is_salary_first_reply(reply: str, dialog_context_meta: Dict[str, Any]) -> bool:
    low = (reply or "").lower()
    explanation_markers = [
        "\u0444\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u0446\u0438\u0444\u0440\u044b",
        "\u0443\u0441\u043b\u043e\u0432\u0438\u044f \u043e\u0431\u0441\u0443\u0436\u0434\u0430\u044e\u0442\u0441\u044f",
        "\u043f\u043e\u0441\u043b\u0435 \u0442\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u0441\u043e\u0431\u0435\u0441\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u044f",
        "\u043e\u0431\u0441\u0443\u0436\u0434\u0430\u044e\u0442\u0441\u044f \u0441 \u043a\u043e\u043c\u0430\u043d\u0434\u043e\u0439",
        "\u0437\u0430\u0432\u0438\u0441\u044f\u0442 \u043e\u0442 \u0443\u0440\u043e\u0432\u043d\u044f",
    ]
    return _is_salary_expectation_question_reply(reply, dialog_context_meta) and _contains_any_substring(
        low, explanation_markers
    )


def _is_bot_first_reply(reply: str) -> bool:
    low = (reply or "").lower()
    role_markers = [
        "\u0432\u043d\u0435\u0448\u043d\u0438\u0439 \u0440\u0435\u043a\u0440\u0443\u0442\u0435\u0440",
        "\u043f\u0435\u0440\u0432\u0438\u0447\u043d\u044b\u0439 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433",
    ]
    continue_markers = [
        "\u043f\u043e\u0434\u0441\u043a\u0430\u0436\u0438\u0442\u0435",
        "\u0443\u0442\u043e\u0447\u043d\u044e",
        "\u043d\u0430 \u043a\u0430\u043a\u0443\u044e \u0441\u0443\u043c\u043c\u0443",
        "\u0432 \u043a\u0430\u043a\u043e\u043c \u0433\u043e\u0440\u043e\u0434\u0435",
    ]
    return (
        not _reply_has_end_marker(reply)
        and _contains_any_substring(low, role_markers)
        and ("?" in (reply or "") or _contains_any_substring(low, continue_markers))
    )


def _is_schedule_refusal_reply(reply: str) -> bool:
    low = (reply or "").lower()
    refusal_markers = [
        "\u0437\u0430\u043d\u0438\u043c\u0430\u044e\u0441\u044c \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u0435\u0440\u0432\u0438\u0447\u043d\u044b\u043c \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u043e\u043c",
        "\u043d\u0435 \u043d\u0430\u0437\u043d\u0430\u0447\u0430\u044e \u0441\u043e\u0437\u0432\u043e\u043d\u044b",
        "\u043d\u0435 \u043d\u0430\u0437\u043d\u0430\u0447\u0430\u044e \u0437\u0432\u043e\u043d\u043a\u0438",
        "\u043d\u0435 \u043d\u0430\u0437\u043d\u0430\u0447\u0430\u044e \u0432\u0441\u0442\u0440\u0435\u0447\u0438",
    ]
    bad_markers = [
        "\u0441\u043e\u0437\u0432\u043e\u043d\u0438\u043c\u0441\u044f",
        "\u043d\u0430\u0437\u043d\u0430\u0447\u0438\u043c",
        "\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0430\u044e \u0432\u0440\u0435\u043c\u044f",
    ]
    return (
        _contains_any_substring(low, refusal_markers)
        and not _contains_any_substring(low, bad_markers)
        and ("?" in (reply or "") or _reply_has_end_marker(reply))
    )


def _question_mark_count(text: str) -> int:
    return (text or "").count("?")


def _contains_markers(text: str, markers: List[str], minimum: int = 1) -> bool:
    low = (text or "").lower()
    matched = sum(1 for marker in markers if marker and marker in low)
    return matched >= minimum


def _looks_like_restarted_intro(reply: str) -> bool:
    low = (reply or "").lower()
    intro_markers = [
        "меня зовут",
        "я внешний рекрутер",
        "обратила внимание",
        "сейчас открыта вакансия",
        "хотела бы предложить",
    ]
    return _contains_any_substring(low, intro_markers)


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


def _is_finish_reply(reply: str) -> bool:
    return (
        _reply_has_end_marker(reply)
        and "?" not in (reply or "")
        and _contains_markers(reply, FINISH_REPLY_MARKERS, minimum=2)
    )


def _has_pause_script(reply: str) -> bool:
    low = (reply or "").lower()
    if _contains_markers(reply, PAUSE_SCRIPT_MARKERS, minimum=2):
        return True
    return _contains_any_substring(low, ["хорошо, договорились", "договорились"]) and _contains_any_substring(
        low,
        [
            "вернуться к этому диалогу позже",
            "вернуться позже",
            "позже обсудим",
            "когда вам будет удобно",
        ],
    )


def _looks_like_direct_answer_request(reply: str) -> bool:
    low = (reply or "").lower()
    direct_markers = [
        "прямо в диалоге",
        "прямо здесь",
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


def _is_hidden_company_reply_valid(
    reply: str,
    dialog_context_meta: Dict[str, Any],
) -> bool:
    low = (reply or "").lower()
    original_company = str(dialog_context_meta.get("original_company_name") or "").strip().lower()
    vacancy_url = str(dialog_context_meta.get("vacancy_url") or "").strip().lower()
    hidden_markers = [
        "\u0441\u043a\u0440\u044b\u0442\u044b\u0439 \u043f\u043e\u0438\u0441\u043a",
        "\u043d\u0435 \u0440\u0430\u0441\u043a\u0440\u044b\u0432\u0430\u0435\u0442\u0441\u044f",
        "\u043d\u0435 \u0440\u0430\u0441\u043a\u0440\u044b\u0432\u0430",
        "\u043d\u0435 \u043c\u043e\u0433\u0443 \u0440\u0430\u0441\u043a\u0440\u044b\u0442\u044c",
        "\u043d\u0435 \u043c\u043e\u0433\u0443 \u043d\u0430\u0437\u0432\u0430\u0442\u044c",
        "\u043d\u0435 \u043c\u043e\u0433\u0443 \u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432\u0438\u0442\u044c \u0441\u0441\u044b\u043b\u043a\u0443",
        "\u043d\u0435 \u043c\u043e\u0433\u0443 \u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432\u0438\u0442\u044c \u0441\u0430\u0439\u0442",
    ]
    continue_markers = [
        "\u043f\u043e\u0434\u0441\u043a\u0430\u0436\u0438\u0442\u0435",
        "\u043d\u0430 \u043a\u0430\u043a\u0443\u044e \u0441\u0443\u043c\u043c\u0443",
        "\u0432 \u043a\u0430\u043a\u043e\u043c \u0433\u043e\u0440\u043e\u0434\u0435",
    ]
    reveals_company = bool(original_company and original_company != "\u0441\u043a\u0440\u044b\u0442\u043e" and original_company in low)
    reveals_url = bool(vacancy_url and vacancy_url in low)
    return (
        not _reply_has_end_marker(reply)
        and not reveals_company
        and not reveals_url
        and _contains_any_substring(low, hidden_markers)
        and ("?" in (reply or "") or _contains_any_substring(low, continue_markers))
    )


def _is_open_company_reply_valid(
    reply: str,
    dialog_context_meta: Dict[str, Any],
    candidate_message: str = "",
) -> bool:
    low = (reply or "").lower()
    candidate_low = (candidate_message or "").lower()
    company_name = (
        str(dialog_context_meta.get("original_company_name") or "").strip()
        or str(dialog_context_meta.get("company_name") or "").strip()
    ).lower()
    title = str(dialog_context_meta.get("title") or "").strip().lower()
    work_format = str(dialog_context_meta.get("work_format") or "").strip().lower()
    vacancy_url = str(dialog_context_meta.get("vacancy_url") or "").strip().lower()
    location = str(dialog_context_meta.get("location") or "").strip().lower()
    location_markers = _location_keywords(location)
    canonical = _canonical_work_format(work_format)
    company_requested = _contains_any_substring(
        candidate_low,
        ["компан", "работодател", "фирм", "кто вы", "название компании"],
    )
    hidden_search_markers = [
        "скрытый поиск",
        "компания не раскрывается",
        "не могу раскрыть компанию",
        "не могу назвать компанию",
    ]

    has_company = bool(company_name and company_name in low)
    has_title = bool(title and title in low) or _contains_any_substring(low, ["позици", "ваканси", "роль"])
    if canonical == "office":
        has_work_format = _contains_any_substring(low, ["офис", "office", "офисный"])
    elif canonical == "hybrid":
        has_work_format = _contains_any_substring(low, ["гибрид", "hybrid", "гибридный"])
    elif canonical == "remote":
        has_work_format = _contains_any_substring(low, ["удален", "удалён", "remote"])
    else:
        has_work_format = bool(work_format and work_format in low)
    has_location = True if not location_markers else _contains_any_substring(low, location_markers)
    has_tasks = _contains_any_substring(low, ["основные задачи", "ключевые задачи", "задачи", "обязанност", "требован"])
    has_url = bool(vacancy_url and vacancy_url in low)
    continues = "?" in (reply or "") or _contains_any_substring(
        low,
        [
            "\u043f\u043e\u0434\u0441\u043a\u0430\u0436\u0438\u0442\u0435",
            "\u043d\u0430 \u043a\u0430\u043a\u0443\u044e \u0441\u0443\u043c\u043c\u0443",
            "\u0432 \u043a\u0430\u043a\u043e\u043c \u0433\u043e\u0440\u043e\u0434\u0435",
        ],
    )
    fact_count = sum(int(flag) for flag in [has_company, has_title, has_work_format, has_location, has_tasks, has_url])

    return (
        not _reply_has_end_marker(reply)
        and not _contains_any_substring(low, hidden_search_markers)
        and continues
        and fact_count >= 2
        and (not company_requested or has_company)
    )


def enforce_prompt_v2_turn_rules(
    scenario: Scenario,
    candidate_message: str,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    idx = scenario.index
    candidate_low = (candidate_message or "").lower()
    reply_low = (assistant_reply or "").lower()

    if idx in (12, 29, 30):
        if idx == 30:
            if _is_apology_end_reply(assistant_reply):
                return 1, "Prompt v2 salary rule passed: third salary scenario ends the dialogue with apology and END."
            return 0, "Prompt v2 salary rule failed: scenario 30 must end the dialogue with apology and END."

        if idx == 29:
            if _is_salary_expectation_question_reply(assistant_reply, dialog_context_meta):
                return 1, "Prompt v2 salary rule passed: repeated salary scenario asks only for expectations."
            return 0, "Prompt v2 salary rule failed: scenario 29 should ask for salary expectations without END."

        if _is_salary_first_reply(assistant_reply, dialog_context_meta):
            return 1, "Prompt v2 salary rule passed: first salary scenario uses the full script and asks for expectations."
        return 0, "Prompt v2 salary rule failed: scenario 12 should use the full script and ask for expectations."

    if idx in (26, 27):
        if idx == 27:
            if _is_apology_end_reply(assistant_reply):
                return 1, "Prompt v2 bot rule passed: repeated bot/AI scenario ends the dialogue."
            return 0, "Prompt v2 bot rule failed: scenario 27 must end the dialogue."

        if _is_bot_first_reply(assistant_reply):
            return 1, "Prompt v2 bot rule passed: first bot/AI scenario explains the role and continues."
        return 0, "Prompt v2 bot rule failed: scenario 26 should explain the role and continue."

    if idx == 22:
        if _is_schedule_refusal_reply(assistant_reply):
            return 1, "Prompt v2 schedule rule passed: assistant refuses scheduling and stays in screening flow."
        return 0, "Prompt v2 schedule rule failed: assistant should refuse scheduling instead of promising a call."

    if idx == 10:
        if _reply_matches_exact_script(assistant_reply, S10_RF_ONLY_SCRIPT):
            return 1, "Prompt v2 geo rule passed: outside-RF scenario uses the exact updated script."
        return 0, "Prompt v2 geo rule failed: scenario 10 must use the exact updated outside-RF script."

    if idx in (23, 24) and bool(dialog_context_meta.get("company_hidden", False)):
        if _is_hidden_company_reply_valid(assistant_reply, dialog_context_meta):
            return 1, "Prompt v2 hidden-company rule passed: assistant keeps the search hidden and continues the dialogue."
        return 0, "Prompt v2 hidden-company rule failed: assistant should keep the search hidden and continue without END."

    if idx == 28 and _contains_any_substring(candidate_low, ["\u0443\u0441\u043b\u043e\u0432\u0438", "\u0437\u0430\u0440\u043f\u043b\u0430\u0442", "\u0432\u0438\u043b\u043a", "\u043a\u043e\u043c\u043f\u0435\u043d\u0441\u0430\u0446"]):
        if _is_salary_expectation_question_reply(assistant_reply, dialog_context_meta):
            return 1, "Prompt v2 vacancy-info rule passed: compensation/conditions question follows the salary script."
        if _contains_any_substring(reply_low, ["\u0443\u0442\u043e\u0447\u043d\u044e \u044d\u0442\u043e\u0442 \u043c\u043e\u043c\u0435\u043d\u0442 \u0443 \u043a\u043e\u043b\u043b\u0435\u0433", "\u0443\u0442\u043e\u0447\u043d\u044e \u0443 \u043a\u043e\u043b\u043b\u0435\u0433"]):
            return 1, "Prompt v2 vacancy-info rule passed: assistant defers missing conditions info to colleagues."
        return 0, "Prompt v2 vacancy-info rule failed: compensation/conditions question should not force the old company-info script."

    if idx == 28:
        if _is_open_company_reply_valid(assistant_reply, dialog_context_meta, candidate_message):
            return 1, "Prompt v2 vacancy-info rule passed: open company info is shared and the dialogue continues."
        return 0, "Prompt v2 vacancy-info rule failed: scenario 28 should provide open company/vacancy info and continue."

    if idx in (34, 35, 36):
        if _asks_work_format_readiness(assistant_reply, dialog_context_meta):
            return 1, f"Prompt v2 office/hybrid rule passed: scenario {idx} asks about readiness for the work format."
        return 0, f"Prompt v2 office/hybrid rule failed: scenario {idx} must continue and explicitly ask about readiness for office/hybrid format."

    if idx == 39:
        if _asks_relocation_and_office_visit(assistant_reply, dialog_context_meta):
            return 1, "Prompt v2 office/hybrid rule passed: scenario 39 asks about relocation to the vacancy city and office attendance."
        return 0, "Prompt v2 office/hybrid rule failed: scenario 39 must continue and ask about relocation to the vacancy city and office attendance."

    if idx == 37:
        if _is_location_or_format_refusal_reply(assistant_reply, dialog_context_meta):
            return 1, "Prompt v2 office/hybrid KO rule passed: scenario 37 rejects due to incompatible location/work format."
        return 0, "Prompt v2 office/hybrid KO rule failed: scenario 37 must end with a location/work-format refusal and END."

    if idx == 38:
        if _reply_matches_exact_script(assistant_reply, S38_DEATH_LOSS_SCRIPT):
            return 1, "Prompt v2 death/loss rule passed: assistant uses the exact apology END script."
        return 0, "Prompt v2 death/loss rule failed: scenario 38 must use the exact apology END script."

    return score, comment


def enforce_prompt_v2_hard_rules(
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if _reply_has_end_marker(assistant_reply) and "?" in (assistant_reply or ""):
        return 0, "Prompt v2 hard rule failed: END cannot be combined with questions."

    return score, comment


def enforce_contact_source_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    idx = scenario.index
    if idx not in CONTACT_SOURCE_SCENARIOS and idx not in CONTACT_SOURCE_RESUME_SCENARIOS:
        return score, comment

    reply_low = (assistant_reply or "").lower()
    if idx in CONTACT_SOURCE_SCENARIOS:
        if _reply_has_end_marker(assistant_reply):
            return 0, "Contact source rule failed: ordinary source-contact question must not end the dialogue."

        if idx in CONTACT_SOURCE_EMPTY_SCENARIOS:
            if CONTACT_SOURCE_FALLBACK_FRAGMENT.lower() not in reply_low:
                return 0, "Contact source rule failed: empty contact_source scenario must use the fallback phrase about the candidate database."
        else:
            expected_source = str(dialog_context_meta.get("contact_source") or "").strip().lower()
            if not expected_source or expected_source not in reply_low:
                return 0, "Contact source rule failed: reply must contain the current contact_source value."

        if _reply_matches_exact_script(assistant_reply, S7_LEGITIMACY_SCRIPT) or _contains_any_substring(
            reply_low, ["мошенн", "развод", "корпоративн", "легитим"]
        ):
            return 0, "Contact source rule failed: ordinary source-contact scenario must not use the legitimacy refusal flow."

        if _question_mark_count(assistant_reply) > 1:
            return 0, "Contact source rule failed: reply must contain at most one question."

        if "продолжить общение по вакансии" not in reply_low:
            return 0, "Contact source rule failed: reply must ask only about continuing the vacancy discussion."

        if _contains_any_substring(
            reply_low,
            [
                "на какую сумму",
                "в каком городе",
                "playwright",
                "ментор",
                "api-автотест",
                "api автотест",
            ],
        ):
            return 0, "Contact source rule failed: reply must not add extra screening questions."

        return 1, "Contact source rule passed: assistant explains the source and asks only whether to continue the vacancy discussion."

    if _reply_has_end_marker(assistant_reply):
        return 0, "Contact source resume rule failed: assistant should continue the current screening step without END."

    if _looks_like_restarted_intro(assistant_reply):
        return 0, "Contact source resume rule failed: assistant restarted the introduction instead of resuming the screening flow."

    if not _contains_any_substring(
        reply_low,
        ["на какую сумму", "зарплат", "в каком городе", "где вы сейчас", "в каком городе вы сейчас"],
    ):
        return 0, "Contact source resume rule failed: assistant should continue with the current priority questions."

    return 1, "Contact source resume rule passed: assistant resumes the screening flow from the current priority step."


def enforce_legitimacy_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
) -> Tuple[int, str]:
    if scenario.index not in LEGITIMACY_SCENARIOS:
        return score, comment

    if _reply_matches_exact_script(assistant_reply, S7_LEGITIMACY_SCRIPT):
        return 1, "Legitimacy rule passed: assistant uses the exact apology END script."
    return 0, "Legitimacy rule failed: legitimacy/fraud scenario must use the exact apology END script."


def enforce_salary_normalization_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    idx = scenario.index
    if idx not in SALARY_NORMALIZATION_SCENARIOS:
        return score, comment

    if idx == 44:
        if _reply_matches_exact_script(assistant_reply, S14_SALARY_REJECTION_SCRIPT):
            return 1, "Salary rule passed: explicit monthly salary above the budget triggers the standard salary rejection END script."
        return 0, "Salary rule failed: explicit monthly salary above the budget must use the standard salary rejection END script."

    if idx in (46, 47, 48, 49):
        if _reply_has_end_marker(assistant_reply):
            return 0, "Salary rule failed: ambiguous/hourly/currency salary must not end the dialogue."
        if not _is_monthly_rubles_clarification_reply(assistant_reply):
            return 0, "Salary rule failed: ambiguous/hourly/currency salary must trigger clarification in monthly net rubles."
        return 1, "Salary rule passed: assistant asks to clarify monthly net salary in rubles."

    if _reply_has_end_marker(assistant_reply):
        return 0, "Salary rule failed: explicit monthly salary within range must not end the dialogue."

    if _is_monthly_rubles_clarification_reply(assistant_reply):
        return 0, "Salary rule failed: explicit monthly salary in rubles per month must not trigger unnecessary clarification."

    return 1, "Salary rule passed: assistant accepts explicit monthly salary in rubles without unnecessary clarification."


def enforce_profile_reference_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    idx = scenario.index
    if idx not in PROFILE_REFERENCE_SCENARIOS and idx not in PROFILE_REFERENCE_RESUME_SCENARIOS:
        return score, comment

    reply_low = (assistant_reply or "").lower()
    questions = _extract_additional_questions(dialog_context_meta)
    current_question = ""
    next_question = ""
    if idx in (50, 61):
        current_question = questions[0] if questions else ""
        next_question = questions[1] if len(questions) > 1 else ""
    elif idx == 51:
        current_question = questions[1] if len(questions) > 1 else (questions[0] if questions else "")
        next_question = questions[2] if len(questions) > 2 else ""
    elif idx == 62:
        current_question = questions[0] if questions else ""
        next_question = questions[1] if len(questions) > 1 else ""

    current_markers = _question_markers_from_text(current_question)
    next_markers = _question_markers_from_text(next_question)

    if idx in PROFILE_REFERENCE_SCENARIOS:
        if _reply_has_end_marker(assistant_reply):
            return 0, "Profile reference rule failed: reply must not end the dialogue."
        if not _looks_like_direct_answer_request(assistant_reply):
            return 0, "Profile reference rule failed: assistant should ask to answer directly in the dialogue."

        if current_markers and not _reply_mentions_question_markers(reply_low, current_markers):
            return 0, "Profile reference rule failed: assistant should repeat the current unanswered question."

        if next_markers and _reply_mentions_question_markers(reply_low, next_markers):
            return 0, "Profile reference rule failed: assistant jumped to the next question instead of clarifying the current one."

        return 1, "Profile reference rule passed: assistant keeps the current question open and asks for a direct answer in chat."

    if _reply_has_end_marker(assistant_reply):
        return 0, "Profile reference resume rule failed: after a substantive answer the dialogue should continue."
    if _looks_like_restarted_intro(assistant_reply):
        return 0, "Profile reference resume rule failed: assistant restarted the whole screening instead of moving to the next question."

    if next_markers:
        if not _reply_mentions_question_markers(reply_low, next_markers):
            return 0, "Profile reference resume rule failed: assistant should move to the next unanswered question after receiving the direct answer."
        if current_markers and _reply_mentions_question_markers(reply_low, current_markers):
            return 0, "Profile reference resume rule failed: assistant repeated the previous question instead of moving forward."
    elif current_markers and _reply_mentions_question_markers(reply_low, current_markers):
        return 0, "Profile reference resume rule failed: assistant repeated the same question after receiving a direct answer."

    return 1, "Profile reference resume rule passed: assistant moves to the next question only after receiving a direct answer."


def enforce_pause_later_rules(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    idx = scenario.index
    if idx not in PAUSE_LATER_SCENARIOS and idx not in PAUSE_LATER_RESUME_SCENARIOS:
        return score, comment

    reply_low = (assistant_reply or "").lower()
    priority_markers = ["в каком городе", "где вы сейчас", "локац", "город"]
    questions = _extract_additional_questions(dialog_context_meta)
    current_question = questions[0] if questions else ""
    current_question_markers = _question_markers_from_text(current_question)
    needs_work_format_ready = _is_office_or_hybrid_work_format(str(dialog_context_meta.get("work_format") or ""))

    if idx == 54:
        if _is_finish_reply(assistant_reply):
            return 1, "Pause rule passed: once everything is already answered, the assistant finishes with END."
        return 0, "Pause rule failed: when all answers are already collected, assistant must use the final finish script with END."

    if idx in PAUSE_LATER_SCENARIOS:
        if _reply_has_end_marker(assistant_reply):
            return 0, "Pause rule failed: assistant must not end the dialogue while required answers are still missing."
        if not _has_pause_script(assistant_reply):
            return 0, "Pause rule failed: reply must contain the new merged pause script."

        if idx in (52, 55):
            if not _contains_any_substring(reply_low, priority_markers):
                return 0, "Pause rule failed: before priority answers are complete, assistant must ask the next missing priority question."
        if idx in (53, 57):
            if needs_work_format_ready and _asks_work_format_readiness(assistant_reply, dialog_context_meta):
                return 1, "Pause rule passed: assistant uses the merged pause script and continues with the required work-format readiness step."
            if current_question_markers and not _reply_mentions_question_markers(reply_low, current_question_markers):
                return 0, "Pause rule failed: after priority answers are complete, assistant must ask the next unanswered question from [questions]."
            if not current_question_markers and "?" not in assistant_reply:
                return 0, "Pause rule failed: after priority answers are complete, assistant should continue with the current unanswered question."

        return 1, "Pause rule passed: assistant uses the merged pause script and continues with the correct next step."

    if _reply_has_end_marker(assistant_reply):
        return 0, "Pause resume rule failed: after the candidate returns later, assistant must resume instead of ending the dialogue."
    if _looks_like_restarted_intro(assistant_reply):
        return 0, "Pause resume rule failed: assistant restarted the screening instead of resuming from the saved step."

    if idx == 56 and not _contains_any_substring(reply_low, priority_markers):
        return 0, "Pause resume rule failed: assistant should resume with the missing priority question."
    if idx == 58:
        if needs_work_format_ready and _asks_work_format_readiness(assistant_reply, dialog_context_meta):
            return 1, "Pause resume rule passed: assistant resumes with the unresolved work-format readiness step."
        if current_question_markers and not _reply_mentions_question_markers(reply_low, current_question_markers):
            return 0, "Pause resume rule failed: assistant should resume with the current unanswered question from [questions]."
        if not current_question_markers and "?" not in assistant_reply:
            return 0, "Pause resume rule failed: assistant should resume with the current unanswered question from [questions]."

    return 1, "Pause resume rule passed: assistant resumes the dialogue from the correct step."


def enforce_open_company_answer_for_s31(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    # Для S31: разрешаем любые доп.вопросы/продолжения,
    # главное - ассистент назвал компанию (и дал ссылку, если она есть в контексте).
    if scenario.index != 31:
        return score, comment

    expected_company = (
        str(dialog_context_meta.get("original_company_name") or "").strip()
        or str(dialog_context_meta.get("company_name") or "").strip()
    )
    expected_url = str(dialog_context_meta.get("vacancy_url") or "").strip()

    if not _is_real_company_name(expected_company):
        return score, comment

    reply_raw = assistant_reply or ""
    reply_low = reply_raw.lower()

    # 1) Компания обязана быть в ответе
    if expected_company.lower() not in reply_low:
        return (
            0,
            f"Scenario 31 strict check failed: assistant reply must contain company name '{expected_company}'.",
        )

    # 2) Если ссылка реально есть в контексте - она тоже обязана быть в ответе
    if expected_url and expected_url.lower() not in reply_low:
        return (
            0,
            f"Scenario 31 strict check failed: assistant reply must contain vacancy_url '{expected_url}'.",
        )

    # 3) Если компания (и ссылка при наличии) есть - считаем ход успешным,
    # даже если evaluator ругается на доп.вопросы.
    return 1, "Scenario 31 relaxed check passed: company (and vacancy_url if provided) present; extra questions allowed."


def enforce_positive_handling_for_s32_s33(
    scenario: Scenario,
    assistant_reply: str,
    score: int,
    comment: str,
    dialog_context_meta: Dict[str, Any],
) -> Tuple[int, str]:
    if scenario.index not in (32, 33):
        return score, comment

    reply_low = (assistant_reply or "").lower()
    if not reply_low.strip():
        return 0, f"Scenario {scenario.index} strict check failed: empty assistant reply."

    hard_stop_markers = [
        "end",
        "заверш",
        "прекрат",
        "не можем продолж",
        "не сможем продолж",
        "вынуждены заверш",
        "вынужден заверш",
        "откаж",
        "не подойд",
        "не подходит",
        "не соответствует",
    ]
    if any(marker in reply_low for marker in hard_stop_markers):
        return (
            0,
            f"Scenario {scenario.index} strict check failed: assistant reply contains refusal/stop markers.",
        )

    continue_markers = [
        "продолж",
        "давайте",
        "подскаж",
        "уточн",
        "расскаж",
        "следующ",
        "?",
    ]
    if not any(marker in reply_low for marker in continue_markers):
        return (
            0,
            f"Scenario {scenario.index} strict check failed: assistant reply should continue dialogue.",
        )

    return 1, f"Scenario {scenario.index} strict check passed: no refusal/stop and dialogue continues."


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
    assistant = create_simple_assistant(cfg, dialog_context=dialog_context)

    candidate_messages = generate_candidate_messages_for_scenario(
        client=client,
        scenario=scenario,
        messages_per_scenario=messages_per_scenario,
        usage_bucket=usage["candidate_generator"],
        dialog_context_meta=dialog_context_meta,
    )

    turns: List[Dict[str, Any]] = []
    scenario_score = 0
    turns_total = 0

    for step_idx, cand_msg in enumerate(candidate_messages, start=1):
        reply = assistant.reply_one_turn(cand_msg)
        _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))

        score, comment = evaluate_turn(
            client=client,
            scenario=scenario,
            step=step_idx,
            candidate_message=cand_msg,
            assistant_reply=reply,
            usage_bucket=usage["evaluator"],
            dialog_history=None,
        )
        score, comment = enforce_open_company_answer_for_s31(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_positive_handling_for_s32_s33(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_prompt_v2_turn_rules(
            scenario=scenario,
            candidate_message=cand_msg,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_contact_source_rules(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_legitimacy_rules(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
        )
        score, comment = enforce_salary_normalization_rules(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_profile_reference_rules(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_pause_later_rules(
            scenario=scenario,
            assistant_reply=reply,
            score=score,
            comment=comment,
            dialog_context_meta=dialog_context_meta,
        )
        score, comment = enforce_prompt_v2_hard_rules(
            assistant_reply=reply,
            score=score,
            comment=comment,
        )

        turn = {
            "step": step_idx,
            "candidate_message": cand_msg,
            "assistant_reply": reply,
            "score": score,
            "comment": comment,
        }
        turns.append(turn)

        scenario_score += score
        turns_total += 1

    case = {
        "case_id": f"S{scenario.index}",
        "type": "single",
        "scenario_index": scenario.index,
        "scenario_name": scenario.name,
        "cdm_file": str(dialog_context_meta.get("cdm_file") or ""),
        "company_hidden": bool(dialog_context_meta.get("company_hidden", False)),
        "vacancy_ref": build_vacancy_ref(dialog_context_meta),
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
    total_score = 0
    total_turns = 0

    # messages_per_scenario здесь - количество прогонов диалога
    for run_idx in range(1, messages_per_scenario + 1):
        conv_id = assistant.create_conversation()
        dialog_history: List[Dict[str, str]] = []
        turns: List[Dict[str, Any]] = []

        run_score = 0
        run_turns = 0

        for step_idx, s in enumerate(scenarios, start=1):
            cand_msg = candidate_variants[s.index][run_idx - 1]
            reply = assistant.reply_in_conversation(conv_id, cand_msg)
            _accumulate_usage(usage["screening_assistant"], getattr(assistant, "last_usage", None))

            score, comment = evaluate_turn(
                client=client,
                scenario=s,
                step=step_idx,
                candidate_message=cand_msg,
                assistant_reply=reply,
                usage_bucket=usage["evaluator"],
                dialog_history=dialog_history,
            )
            score, comment = enforce_open_company_answer_for_s31(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_positive_handling_for_s32_s33(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_prompt_v2_turn_rules(
                scenario=s,
                candidate_message=cand_msg,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_contact_source_rules(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_legitimacy_rules(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
            )
            score, comment = enforce_salary_normalization_rules(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_profile_reference_rules(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_pause_later_rules(
                scenario=s,
                assistant_reply=reply,
                score=score,
                comment=comment,
                dialog_context_meta=dialog_context_meta,
            )
            score, comment = enforce_prompt_v2_hard_rules(
                assistant_reply=reply,
                score=score,
                comment=comment,
            )

            turn = {
                "step": step_idx,
                "scenario_index": s.index,
                "scenario_name": s.name,
                "candidate_message": cand_msg,
                "assistant_reply": reply,
                "score": score,
                "comment": comment,
            }
            turns.append(turn)

            run_score += score
            run_turns += 1

            dialog_history.append({"candidate": cand_msg, "assistant": reply})

        runs.append(
            {
                "run_index": run_idx,
                "turns_total": run_turns,
                "score_total": run_score,
                "passed": run_score == run_turns,
                "turns": turns,
            }
        )

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
        "runs_total": len(runs),
        "turns_total": total_turns,
        "score_total": total_score,
        "score_rate": (total_score / total_turns) if total_turns else 0.0,
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

    if not scenarios:
        if scenario_indices:
            raise ValueError(
                "No scenarios matched --scenario-indices. "
                f"Requested: {scenario_indices}. Check CSV row indices."
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
        hide_company = _group_requires_hidden_company(group)
        contact_source_override = ""
        if not _group_has_any_scenario(group, sorted(CONTACT_SOURCE_EMPTY_SCENARIOS)):
            contact_source_override = None
        dialog_context, dialog_context_meta = build_dialog_context(
            fixture=fixture,
            hide_company=hide_company,
            contact_source_override=contact_source_override,
        )
        print(
            f"  - cdm={fixture.file_name} | "
            f"company_hidden={'YES' if hide_company else 'NO'}"
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
    score_rate = (score_total / turns_total) if turns_total else 0.0
    passed_cases = sum(1 for case in cases if bool(case.get("passed")))
    failed_cases = cases_total - passed_cases
    pass_rate = (passed_cases / cases_total * 100.0) if cases_total else 0.0

    mismatches = build_mismatches(cases)
    token_usage_total = _token_usage_total(usage)
    sa_cfg = _component_cfg(cfg, "screening_assistant")

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
            "screening_assistant": {
                "prompt_id": sa_cfg.get("prompt_id"),
                "prompt_version": sa_cfg.get("prompt_version"),
            },
            "evaluator": EVAL_MODEL,
        },
        "token_usage_total": token_usage_total,
        "token_usage": usage,
        "summary": {
            "cases_total": cases_total,
            "turns_total": turns_total,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": pass_rate,
            "score_total": score_total,
            "score_rate": score_rate,
        },
        "cases": cases,
        "mismatches": mismatches,
    }

    out_path = REPORTS_DIR / f"screening_scenarios_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] Screening scenarios report saved to: {out_path}")
    failed_turns = sum(len(item.get("dialogs") or []) for item in mismatches)
    print(
        f"[summary] cases_total={cases_total} | turns_total={turns_total} | "
        f"score_total={score_total} | score_rate={score_rate:.3f} | "
        f"failed_cases={failed_cases} | failed_turns={failed_turns}"
    )

    return out_path


# -----------------------
# CLI
# -----------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavioral screening scenarios against screening_assistant (supports chain multi-scenario dialogs)."
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

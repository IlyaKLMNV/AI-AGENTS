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
        desc_key = "Краткое описание сценария"

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

            description = row.get(desc_key) or ""

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
            raw_url = str(vacancy.get("vacancy_url") or "").strip()
            if raw_url:
                company_info = dict(vacancy_info.get("company_info") or {})
                company_info["vacancy_url"] = raw_url
                vacancy_info["company_info"] = company_info

            fixtures.append(
                CdmFixture(
                    file_name=path.name,
                    vacancy_info=vacancy_info,
                    names=names,
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


def _is_hidden_company_scenario(s: Scenario) -> bool:
    return s.index in (23, 24)


def _group_requires_hidden_company(group: ScenarioGroup) -> bool:
    return any(_is_hidden_company_scenario(s) for s in group.scenarios)


def build_dialog_context(
    fixture: CdmFixture,
    hide_company: bool,
) -> Tuple[str, Dict[str, Any]]:
    vacancy_info = fixture.vacancy_info
    names = fixture.names

    recruiter_name = str(names.get("recruiter_name") or "Рекрутер").strip()
    candidate_name = str(names.get("candidate_name") or "Кандидат").strip()
    title = str(vacancy_info.get("title") or "").strip()
    original_company_name = str(vacancy_info.get("company_name") or "").strip()
    company_name = "СКРЫТО" if hide_company else original_company_name
    responsibilities = str(vacancy_info.get("responsibilities") or "").strip()
    work_format = str(vacancy_info.get("work_format") or "").strip()
    company_info = vacancy_info.get("company_info") or {}
    firm_description = str(company_info.get("firm_description") or "").strip()
    vacancy_url = "" if hide_company else str(company_info.get("vacancy_url") or "").strip()
    salary = _salary_range_text(vacancy_info)
    questions = str(vacancy_info.get("questions") or "").strip() or "-"

    lines = [
        "### Контекст для диалога (будет предоставлен перед началом)",
        f"Ваше имя: {recruiter_name}",
        f"Имя кандидата: {candidate_name}",
        f"Должность: {title}",
        f"Компания: {company_name}",
        f"Обязанности: {responsibilities}",
        f"Формат работы: {work_format}",
        f"Описание компании: {firm_description}",
        f"Ссылка: {vacancy_url}",
        f"Зарплатная вилка: {salary} (НЕ РАСКРЫВАТЬ!)",
        "Приоритетные вопросы:",
        "1. Зарплатные ожидания",
        "2. Локация/город",
        "Дополнительные вопросы:",
        questions,
    ]
    context_text = "\n".join(lines).strip()

    context_meta = {
        "cdm_file": fixture.file_name,
        "company_hidden": hide_company,
        "recruiter_name": recruiter_name,
        "candidate_name": candidate_name,
        "title": title,
        "company_name": company_name,
        "original_company_name": original_company_name,
        "responsibilities": responsibilities,
        "work_format": work_format,
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
    }


def _short_reason(comment: str, limit: int = 140) -> str:
    normalized = re.sub(r"\s+", " ", str(comment or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


def _case_failures(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    case_type = case.get("type")

    if case_type == "single":
        scenario_index = int(case.get("scenario_index") or 0)
        for turn in case.get("turns") or []:
            if int(turn.get("score", 0)) == 0:
                failures.append(
                    {
                        "run_index": None,
                        "step": int(turn.get("step") or 0),
                        "scenario_index": scenario_index,
                        "reason": _short_reason(str(turn.get("comment") or "")),
                    }
                )
        return failures

    if case_type == "chain":
        for run in case.get("runs") or []:
            run_index = int(run.get("run_index") or 0)
            for turn in run.get("turns") or []:
                if int(turn.get("score", 0)) == 0:
                    failures.append(
                        {
                            "run_index": run_index,
                            "step": int(turn.get("step") or 0),
                            "scenario_index": int(turn.get("scenario_index") or 0),
                            "reason": _short_reason(str(turn.get("comment") or "")),
                        }
                    )
        return failures

    return failures


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

    # 12/29/30 - про зарплату
    if idx in (12, 29, 30) or _has_any(name, TOPIC_SALARY):
        return (
            "В КАЖДОЙ реплике кандидат должен ЗАДАВАТЬ вопрос про деньги/зарплату/вилку.\n"
            "Если это повтор/третий раз - явно укажи, что кандидат уже спрашивал и ответа не получил.\n"
        )

    # 31 - компания при открытом поиске
    if idx == 31:
        return (
            "В КАЖДОЙ реплике кандидат должен прямо спрашивать: какая компания/как называется компания,"
            " можно ли дать сайт/ссылку на вакансию.\n"
            "Это открытый поиск: кандидат ожидает раскрытия названия компании.\n"
        )

    # 23/24 - скрытая компания (проверяем раньше BOT, чтобы не было сюрпризов по приоритетам)
    if idx in (23, 24) or ("скрытом" in name and "компан" in name):
        return (
            "В КАЖДОЙ реплике кандидат должен прямо спрашивать: какая компания? назовите компанию/дайте сайт/ссылку.\n"
            "Если настойчивый - упомяни, что уже спрашивал и ответа нет.\n"
        )

    # 26/27 - бот
    if idx in (26, 27) or _has_any(name, TOPIC_BOT):
        return (
            "В КАЖДОЙ реплике кандидат должен спрашивать: вы бот/ИИ или человек?\n"
            "Если повторно - явно укажи, что это повторный вопрос и ранее ответа не было.\n"
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
            "- ВСЕ реплики кандидата должны быть на иностранном языке, НЕ на русском.\n"
            "- Можно использовать английский, польский, украинский и т.п."
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
            "- В КАЖДОЙ реплике явно обозначь жёсткое ограничение по локации или часовому поясу.\n"
            "- Формулировки: «живу за пределами России», «только удалёнка», «не планирую переезд»."
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
    if idx == 23 and "скрытом" in name:
        parts.append("- Кандидат прямо спрашивает, КАКАЯ это компания, просит название/сайт/ссылку.")
    if idx == 24 or ("настойчивый" in name and "скрытом" in name):
        parts.append(
            "- Это несколько сообщений подряд, кандидат настойчиво повторяет вопрос именно про компанию: "
            "«я уже спрашивал, какая компания», «еще раз: назовите компанию или дайте ссылку»."
        )

    # 31. Компания открытый поиск
    if idx == 31:
        parts.append(
            "- Кандидат прямо спрашивает название компании: «какая компания?», «как называется компания?»."
        )
        parts.append(
            "- Можно просить сайт или ссылку на вакансию."
        )

    # 25. Нет опыта
    if idx == 25 or "нет нужного опыта" in name or "отсутствие необходимого" in name:
        parts.append(
            "- Кандидат прямо говорит, что нужного опыта нет.\n"
            "- Не добавляй «но быстро разберусь» - акцент на отсутствии опыта."
        )

    # 26/27. Бот
    if idx == 26 or "бот" in name:
        parts.append("- Кандидат сомневается: «ты бот?», «это ИИ или человек?».")
    if idx == 27 or ("повторно" in name and "бот" in name):
        parts.append("- Повторный вопрос про бота: «я же уже спрашивал, вы бот?».")
    if not parts:
        return ""

    return "Дополнительные требования для этого конкретного сценария:\n" + "\n".join(parts)


def _validate_trigger(s: Scenario, msg: str) -> bool:
    """
    Очень простая проверка "попали ли в триггер".
    Не пытаемся покрыть все сценарии - только те, где чаще всего промахи критичны.
    """
    m = _normalize_text(msg).lower()
    idx = s.index
    name = s.name.lower()

    # политика/нация
    if idx == 1 or "полит" in name or "национ" in name:
        return any(k in m for k in KW_POL_NATION)

    # зарплата
    if idx in (12, 29, 30) or _has_any(name, TOPIC_SALARY):
        return any(k in m for k in KW_SALARY)

    # скрытая компания (до bot)
    if idx in (23, 24) or ("скрытом" in name and "компан" in name):
        return any(k in m for k in KW_COMPANY)

    # 31 - открытый поиск по компании
    if idx == 31:
        return any(k in m for k in KW_COMPANY)

    # бот (KW_BOT через safe matcher)
    if idx in (26, 27) or _has_any(name, TOPIC_BOT):
        return _has_any(m, KW_BOT)

    # иначе не валидируем
    return True


def _fallback_messages(s: Scenario, n: int) -> List[str]:
    """
    Жесткий fallback только для ключевых триггеров, чтобы тест не был мусорным.
    """
    idx = s.index
    name = s.name.lower()

    if idx == 1 or "полит" in name or "национ" in name:
        pool = [
            "Сразу уточню: из-за политики и санкций я принципиально не рассматриваю проекты, связанные с российскими компаниями.",
            "Из-за текущей войны и позиции власти я не хочу иметь дел с компаниями из РФ, даже если условия хорошие.",
            "У меня принцип: не работаю с вакансиями, где есть связь с государством и политикой, это для меня стоп-фактор.",
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
) -> List[str]:
    """
    Генерация максимально близкая к оригиналу:
    - примеры (если есть)
    - extra-guidelines
    - sequential, если сценарий повторный
    Плюс: trigger forcing + 1 перегенерация + fallback.
    """
    examples = extract_candidate_examples(scenario.examples_raw, max_examples=10)
    extra = _extra_generation_guidelines(scenario)
    is_repeated = _is_repeated_dialog_scenario(scenario)
    trigger_req = _trigger_requirement_text(scenario)

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

    def _do_gen(strong: bool) -> List[str]:
        prompt = _build_prompt(strong=strong).format(n=messages_per_scenario)
        payload = prompt + "\n\n" + json.dumps(payload_obj, ensure_ascii=False)

        resp = client.responses.create(model=GEN_MODEL, input=payload)
        _accumulate_usage(usage_bucket, getattr(resp, "usage", None))
        text = (getattr(resp, "output_text", "") or "").strip()

        msgs = _parse_json_string_list(text)
        cleaned = [_normalize_text(m) for m in msgs[:messages_per_scenario]]
        return cleaned

    # 1) первая попытка
    messages = _do_gen(strong=False)

    # 2) валидируем, если промахи - перегенерим 1 раз
    if messages and not all(_validate_trigger(scenario, m) for m in messages):
        messages2 = _do_gen(strong=True)
        if messages2:
            messages = messages2

    # 3) если совсем плохо - fallback
    if not messages or not all(_validate_trigger(scenario, m) for m in messages):
        messages = _fallback_messages(scenario, messages_per_scenario)

    # гарантируем длину и чистку END
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

    if not expected_company:
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
        fixture = cdm_fixtures[(gidx - 1) % len(cdm_fixtures)]
        hide_company = _group_requires_hidden_company(group)
        dialog_context, dialog_context_meta = build_dialog_context(
            fixture=fixture,
            hide_company=hide_company,
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

    errors_by_case: List[Dict[str, Any]] = []
    for case in cases:
        if bool(case.get("passed")):
            continue
        errors_by_case.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "type": str(case.get("type") or ""),
                "cdm_file": str(case.get("cdm_file") or ""),
                "company_hidden": bool(case.get("company_hidden", False)),
                "failures": _case_failures(case),
            }
        )

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
            "errors_by_case": errors_by_case,
        },
        "cases": cases,
    }

    out_path = REPORTS_DIR / f"screening_scenarios_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] Screening scenarios report saved to: {out_path}")
    failed_turns = sum(len(item.get("failures") or []) for item in errors_by_case)
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

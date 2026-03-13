from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pathlib
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "screening_autofill"

DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
DEFAULT_VARIANTS_PER_CDM = 3
DEFAULT_CDM_COUNT = None

DEFAULT_DIALOGUE_GEN_MODEL = "gpt-4.1-mini"
DIALOGUE_GEN_MAX_RETRIES = 1

REGRESSION_CASES: List[Dict[str, Any]] = [
    {
        "name": "wf_hybrid_explicit_candidate",
        "description": "Кандидат явно согласен на гибрид, но модель не должна подменять это на remote.",
        "vacancy_title": "Senior Virtualization Engineer",
        "vacancy_company": "DataGrid",
        "dialogue": (
            "Кандидат: Добрый день! Последние 6 лет занимаюсь инфраструктурой и виртуализацией, "
            "в основном VMware ESXi и vCenter, плюс сопровождал отказоустойчивые кластеры.\n"
            "Рекрутер: Добрый день! Подскажите, пожалуйста, какой уровень дохода рассматриваете "
            "и в каком городе находитесь?\n"
            "Кандидат: По деньгам ориентируюсь на 420000 рублей gross, нахожусь в Москве.\n"
            "Рекрутер: Поняла, спасибо. Готовы ли вы к гибридному формату, 1-2 дня в офисе? "
            "И отдельно расскажите, пожалуйста, был ли у вас опыт с VMware ESXi и виртуализацией?\n"
            "Кандидат: Готов к гибриду, 1-2 дня в офисе для меня нормально. "
            "С VMware ESXi работаю давно: настраивал кластеры, хранилища и миграции без даунтайма."
        ),
        "expected_json": {"work_format": "hybrid"},
    },
    {
        "name": "wf_empty_when_candidate_silent",
        "description": "Кандидат ничего не говорит про формат работы, значит work_format должен остаться пустым.",
        "vacancy_title": "Backend Python Engineer",
        "vacancy_company": "CloudCore",
        "dialogue": (
            "Кандидат: Добрый день! Я backend-разработчик, последние пять лет работаю с Python, "
            "FastAPI и PostgreSQL, плюс немного трогал Kafka.\n"
            "Рекрутер: Добрый день! Подскажите, пожалуйста, в каком городе вы сейчас находитесь "
            "и какие у вас зарплатные ожидания?\n"
            "Кандидат: Сейчас я в Санкт-Петербурге, по деньгам ориентируюсь на 360000 рублей gross.\n"
            "Рекрутер: Спасибо. А какой у вас практический опыт с highload-сервисами и очередями?\n"
            "Кандидат: На текущем проекте вел сервисы с нагрузкой порядка 20 тысяч запросов в минуту, "
            "Kafka использовал для асинхронной обработки событий и ретраев."
        ),
        "expected_json": {"work_format": ""},
    },
    {
        "name": "wf_empty_when_only_recruiter_mentions_hybrid",
        "description": "Рекрутер упоминает гибрид, но кандидат формат не подтверждает; извлекать work_format нельзя.",
        "vacancy_title": "Infrastructure Engineer",
        "vacancy_company": "InfraWave",
        "dialogue": (
            "Кандидат: Добрый день! У меня 7 лет опыта в администрировании Linux и виртуализации, "
            "последние проекты были связаны с on-prem и private cloud.\n"
            "Рекрутер: Подскажите, пожалуйста, какую зарплату рассматриваете и в каком городе вы находитесь?\n"
            "Кандидат: Я в Москве, по деньгам ориентируюсь на 400000 рублей gross.\n"
            "Рекрутер: Готовы ли вы рассматривать гибридный формат? И второй вопрос: "
            "какой у вас опыт с виртуализацией и VMware ESXi?\n"
            "Кандидат: По VMware ESXi работал около четырех лет: поднимал кластеры, "
            "настраивал HA и занимался обновлениями гипервизоров."
        ),
        "expected_json": {"work_format": ""},
    },
]


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        it = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_token_count")
            or 0
        )
        ot = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_token_count")
            or 0
        )
        tt = usage.get("total_tokens") or usage.get("token_count")
    else:
        it = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        ot = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        tt = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)

    if tt is None:
        tt = (it or 0) + (ot or 0)

    return int(it or 0), int(ot or 0), int(tt or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    it, ot, tt = _extract_usage_numbers(usage)
    bucket["input_tokens"] += it
    bucket["output_tokens"] += ot
    bucket["total_tokens"] += tt


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
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty json text")
    try:
        return json.loads(raw)
    except Exception:
        extracted = _extract_json_substring(raw)
        if not extracted:
            raise
        return json.loads(extracted)


def _only_digits_or_empty(s: Any) -> bool:
    if s is None:
        return True
    if not isinstance(s, str):
        return False
    if s == "":
        return True
    return s.isdigit()


def _validate_schema(obj: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["output is not a JSON object"]

    required = ["preferred_location", "min_salary", "max_salary", "work_format", "additional_info"]
    for k in required:
        if k not in obj:
            errors.append(f"missing key:{k}")

    if "preferred_location" in obj and not isinstance(obj.get("preferred_location"), str):
        errors.append("preferred_location_must_be_string")

    if "work_format" in obj:
        wf = obj.get("work_format")
        if not isinstance(wf, str):
            errors.append("work_format_must_be_string")
        elif wf not in ("", "remote", "office", "hybrid"):
            errors.append("work_format_invalid_value")

    if "min_salary" in obj and not _only_digits_or_empty(obj.get("min_salary")):
        errors.append("min_salary_must_be_digits_or_empty")

    if "max_salary" in obj and not _only_digits_or_empty(obj.get("max_salary")):
        errors.append("max_salary_must_be_digits_or_empty")

    if "additional_info" in obj:
        ai = obj.get("additional_info")
        if not isinstance(ai, list):
            errors.append("additional_info_must_be_list")
        else:
            for i, item in enumerate(ai):
                if not isinstance(item, dict):
                    errors.append(f"additional_info[{i}]_must_be_object")
                    continue
                q = item.get("question")
                a = item.get("answer")
                if not isinstance(q, str) or not isinstance(a, str):
                    errors.append(f"additional_info[{i}]_question_answer_must_be_strings")

    return errors


def _json_debug_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _validate_expected_json_subset(parsed: Any, expected: Optional[Dict[str, Any]]) -> List[str]:
    if not expected:
        return []
    if not isinstance(parsed, dict):
        return ["expected_json_subset_output_not_object"]

    errors: List[str] = []
    for key, expected_value in expected.items():
        actual_value = parsed.get(key)
        if actual_value != expected_value:
            errors.append(
                "expected_field_mismatch:"
                f"{key}:expected={_json_debug_value(expected_value)}:"
                f"actual={_json_debug_value(actual_value)}"
            )
    return errors


def _parse_case_names_filter(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    out = {item.strip() for item in raw.split(",") if item.strip()}
    return out or None


def _select_regression_cases(case_names: Optional[Set[str]]) -> List[Dict[str, Any]]:
    if not case_names:
        return [dict(case) for case in REGRESSION_CASES]

    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for case in REGRESSION_CASES:
        name = str(case.get("name") or "")
        if name in case_names:
            selected.append(dict(case))
            seen.add(name)

    missing = sorted(case_names - seen)
    if missing:
        raise ValueError(
            "unknown regression case names: " + ", ".join(missing)
        )
    return selected


def _resolve_prompt_from_cfg(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    block = cfg.get("screening_autofill") or {}
    pid = block.get("prompt_id")
    pver = block.get("prompt_version")
    return (str(pid) if pid else None, str(pver) if pver else None)


def _resolve_dialogue_gen_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    block = cfg.get("screening_autofill") or {}
    m = block.get("dialogue_gen_model")
    return str(m) if m else None


def load_cdm_files(cdm_dir: pathlib.Path, cdm_count: Optional[int]) -> List[pathlib.Path]:
    if not cdm_dir.exists():
        raise FileNotFoundError(f"CDM dir not found: {cdm_dir}")

    paths = [pathlib.Path(p) for p in sorted(glob.glob(str(cdm_dir / "cdm_*.json")))]
    if not paths:
        raise FileNotFoundError(f"No cdm_*.json found in: {cdm_dir}")

    if cdm_count is not None:
        if cdm_count <= 0:
            raise ValueError("--cdm-count must be > 0")
        paths = paths[:cdm_count]

    return paths


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_questions(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", ln).strip()
        if ln:
            out.append(ln)
    return out


def _format_dialogue(turns: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for t in turns:
        who = (t.get("speaker", "") or "").strip().lower()
        msg = (t.get("text") or "").strip()
        if not who or not msg:
            continue
        if who == "recruiter":
            lines.append(f"Рекрутер: {msg}")
        elif who == "candidate":
            lines.append(f"Кандидат: {msg}")
    return "\n".join(lines).strip()


def _flatten_like_prod(dialogue: str) -> str:
    s = (dialogue or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_dialogue_to_turns(dialogue: str) -> List[Tuple[str, str]]:
    text = (dialogue or "").strip()
    if not text:
        return []
    pattern = re.compile(r"(Рекрутер|Кандидат)\s*:\s*", flags=re.UNICODE)
    parts = pattern.split(text)
    if len(parts) < 3:
        return []
    turns: List[Tuple[str, str]] = []
    i = 1
    while i < len(parts):
        speaker = parts[i].strip()
        msg = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2
        if msg:
            turns.append((speaker, msg))
    return turns


def _candidate_text(dialogue: str) -> str:
    turns = _split_dialogue_to_turns(dialogue)
    return " ".join(msg for speaker, msg in turns if speaker == "Кандидат").strip()


def _recruiter_questions(dialogue: str) -> List[str]:
    turns = _split_dialogue_to_turns(dialogue)
    qs: List[str] = []
    for speaker, msg in turns:
        if speaker != "Рекрутер":
            continue
        for q in msg.split("?"):
            q = q.strip()
            if q:
                qs.append(q + "?")
    return qs


_COMMON_CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Петербург",
    "СПб",
    "Казань",
    "Екатеринбург",
    "Новосибирск",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов",
    "Воронеж",
    "Пермь",
    "Минск",
    "Алматы",
    "Астана",
    "Тбилиси",
]


def _mentions_city(text: str) -> bool:
    t = text or ""
    for c in _COMMON_CITIES:
        if re.search(rf"\b{re.escape(c)}\b", t, flags=re.IGNORECASE):
            return True
    return False


_SALARY_WORDS = re.compile(
    r"(зарплат|оклад|компенсац|вилк|вознагражден|на руки|нетто|netto|gross|гросс|брутто)",
    re.IGNORECASE,
)
_CURRENCY_WORDS = re.compile(r"(₽|\$|€|\bруб\b|\bрубл)", re.IGNORECASE)
_THOUSAND_NUMBER_SUFFIX = re.compile(r"\b\d+\s*(тыс|тысяч|т\.?р\.?|к|k)\b", re.IGNORECASE)

_LOCATION_WORDS = re.compile(r"(город|локац|находит|жив[еу]|прожива|переезд|релокац)", re.IGNORECASE)
_WORKFORMAT_WORDS = re.compile(
    r"(удален|удалён|дистанцион|remote|офис|\bочно\b|гибрид|смешан|формат работы|режим работы)",
    re.IGNORECASE,
)
_WORKFORMAT_CONCRETE = re.compile(
    r"(удален|удалён|удаленно|удалённо|удаленка|удалёнка|дистанцион|remote|офис|гибрид|смешан|\bочно\b)",
    re.IGNORECASE,
)

# NEW: salary expectation detection
_SALARY_RANGE_DASH = re.compile(r"\b(\d{2,3})\s*[-–—]\s*(\d{2,3})\b", re.UNICODE)  # e.g. 350-400 (usually thousands)
_SALARY_RANGE_FULL = re.compile(r"\b(\d{5,7})\s*[-–—]\s*(\d{5,7})\b", re.UNICODE)  # e.g. 350000-400000
_SALARY_FROM_TO = re.compile(r"\b(от|до|в районе|примерно|порядка|диапазон)\b", re.IGNORECASE)


def _salary_topic(text: str) -> bool:
    t = text or ""
    if _SALARY_WORDS.search(t):
        return True
    if _CURRENCY_WORDS.search(t):
        return True
    if _THOUSAND_NUMBER_SUFFIX.search(t):
        return True
    return False


def _salary_expectation_provided(text: str) -> bool:
    """
    True only if candidate actually provided numeric expectations/range,
    not just discussed salary as a topic.
    """
    t = (text or "").strip()
    if not t:
        return False

    # any explicit "N тыс/к/k" is an expectation (350к, 350 тыс)
    if _THOUSAND_NUMBER_SUFFIX.search(t):
        return True

    # full money numbers with currency/context nearby
    if re.search(r"\b\d{5,7}\b", t):
        if _SALARY_WORDS.search(t) or _CURRENCY_WORDS.search(t) or _SALARY_FROM_TO.search(t):
            return True
        # sometimes candidate writes "350000" without 'руб' but with "от/до/диапазон"
        if _SALARY_FROM_TO.search(t):
            return True

    # ranges like 350000-400000
    if _SALARY_RANGE_FULL.search(t):
        return True

    # ranges like 350-400 with salary context (usually "к/тыс/руб/зарплата")
    m = _SALARY_RANGE_DASH.search(t)
    if m:
        if _SALARY_WORDS.search(t) or _CURRENCY_WORDS.search(t) or _THOUSAND_NUMBER_SUFFIX.search(t) or _SALARY_FROM_TO.search(t):
            return True

    # "от 350" with context
    if re.search(r"\bот\s+\d{2,7}\b", t, flags=re.IGNORECASE):
        if _SALARY_WORDS.search(t) or _CURRENCY_WORDS.search(t) or _THOUSAND_NUMBER_SUFFIX.search(t):
            return True

    return False


def _location_topic(text: str) -> bool:
    t = text or ""
    if _mentions_city(t):
        return True
    return bool(_LOCATION_WORDS.search(t))


def _workformat_topic(text: str) -> bool:
    t = text or ""
    return bool(_WORKFORMAT_WORDS.search(t))


def _topics_in_text(text: str) -> Set[str]:
    topics: Set[str] = set()
    if _salary_topic(text):
        topics.add("salary")
    if _location_topic(text):
        topics.add("location")
    if _workformat_topic(text):
        topics.add("work_format")
    return topics


def _semantic_validate(dialogue: str, parsed: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(parsed, dict):
        return ["output_not_object"]

    cand = _candidate_text(dialogue)

    # FIX: require salary only if expectation was actually provided
    expects_salary = _salary_expectation_provided(cand)
    expects_location = _location_topic(cand)
    expects_workformat = bool(_WORKFORMAT_CONCRETE.search(cand))

    min_s = str(parsed.get("min_salary") or "")
    max_s = str(parsed.get("max_salary") or "")
    loc = str(parsed.get("preferred_location") or "")
    wf = str(parsed.get("work_format") or "")
    ai = parsed.get("additional_info")

    if expects_salary and not (min_s or max_s):
        errors.append("salary_missing")
    if expects_location and not loc.strip():
        errors.append("location_missing")
    if expects_workformat and not wf.strip():
        errors.append("work_format_missing")

    if not isinstance(ai, list):
        errors.append("additional_info_not_list")
        return errors

    recruiter_qs = _recruiter_questions(dialogue)
    recruiter_qs_non_excluded = [q for q in recruiter_qs if not _topics_in_text(q)]
    if recruiter_qs_non_excluded and len(ai) == 0:
        errors.append("additional_info_empty_but_questions_exist")

    for i, item in enumerate(ai):
        if not isinstance(item, dict):
            errors.append(f"additional_info[{i}]_not_object")
            continue

        q = str(item.get("question") or "")
        a = str(item.get("answer") or "")

        if not q.strip() or not a.strip():
            errors.append(f"additional_info[{i}]_empty_question_or_answer")

        if "Рекрутер:" in a or "Кандидат:" in a:
            errors.append(f"additional_info[{i}]_speaker_labels_leaked_in_answer")
        if "Рекрутер:" in q or "Кандидат:" in q:
            errors.append(f"additional_info[{i}]_speaker_labels_leaked_in_question")

        topics = _topics_in_text(q) | _topics_in_text(a)
        if topics:
            errors.append(f"additional_info[{i}]_contains_excluded_topic:{','.join(sorted(topics))}")

    return errors


def _normalize_speaker(s: Any) -> str:
    t = str(s or "").strip().lower()
    if t in ("recruiter", "рекрутер"):
        return "recruiter"
    if t in ("candidate", "кандидат"):
        return "candidate"
    return t


def _validate_turns_structure(turns: Any) -> bool:
    if not isinstance(turns, list):
        return False
    if len(turns) < 4:
        return False
    expected = "recruiter"
    for t in turns:
        if not isinstance(t, dict):
            return False
        sp = _normalize_speaker(t.get("speaker"))
        tx = str(t.get("text") or "").strip()
        if sp not in ("recruiter", "candidate"):
            return False
        if not tx:
            return False
        if sp != expected:
            return False
        expected = "candidate" if expected == "recruiter" else "recruiter"
    return True


def _ordered_parsed_json(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {
        "preferred_location": str(obj.get("preferred_location") or ""),
        "min_salary": str(obj.get("min_salary") or ""),
        "max_salary": str(obj.get("max_salary") or ""),
        "work_format": str(obj.get("work_format") or ""),
        "additional_info": obj.get("additional_info") if isinstance(obj.get("additional_info"), list) else [],
    }


class DialogueSynthesizer:
    def __init__(self, model: str, seed: Optional[int]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.last_usage: Any = None
        self.seed = seed

    def _instruction(self, variants: int) -> str:
        return (
            "Ты генерируешь реалистичный диалог рекрутера и кандидата для первичного скрининга.\n"
            "Важно: диалог нужен для тестирования авто-заполнения формы скрининга.\n\n"
            "Жесткие требования к формату:\n"
            f"0) Верни JSON МАССИВ ровно из {variants} элементов.\n"
            "1) Каждый элемент массива имеет ключ turns.\n"
            "2) turns - список объектов вида {\"speaker\":\"recruiter|candidate\",\"text\":\"...\"}.\n"
            "3) В КАЖДОМ элементе turns имеет длину >= 4.\n"
            "4) В КАЖДОМ элементе speaker строго чередуется: recruiter, candidate, recruiter, candidate, ...\n"
            "5) Первый speaker в turns всегда recruiter.\n"
            "6) Никаких других полей, никакого текста вне JSON.\n\n"
            "Требования к смыслу:\n"
            "1) Рекрутер задает вопросы строго из списка vacancy.questions (можно перефразировать), но не добавляй новые темы.\n"
            "2) В одном сообщении рекрутера максимум 2 вопроса, если включен mix_two_questions.\n"
            "3) Кандидат отвечает по смыслу. Иногда отвечает не прямолинейно (answer_indirect), добавляет детали и шум.\n"
            "4) В ответах кандидата должны иногда встречаться:\n"
            "   - город/локация\n"
            "   - ожидания по зарплате (с явным контекстом денег, например рубли/тыс/зарплата)\n"
            "   - формат работы: удаленно/офис/гибрид\n"
            "5) Иногда кандидат может дать часть ответа сразу, а часть уточнить позже.\n"
            "6) Не используй Markdown. Не добавляй никаких служебных меток.\n"
            "7) Иногда кандидат в ответ задает 1-3 уточняющих вопроса о роли/загрузке/подчинении/процессе.\n"
            "   - Эти вопросы пишет Кандидат: в конце своей реплики.\n"
            "   - Не задавай вопросы кандидата про зарплату/локацию/формат работы.\n"
            "8) Если noise == high: кандидат ОБЯЗАТЕЛЬНО задает уточняющие вопросы хотя бы в одной своей реплике.\n"

        )

    def _payload(
        self,
        cdm: Dict[str, Any],
        variants: int,
        noise_level: int,
        allow_two_questions: bool,
        prior_dialogues: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        vacancy = cdm.get("vacancy") or {}
        candidate = cdm.get("candidate") or {}

        questions_raw = vacancy.get("questions") or ""
        questions = _parse_questions(questions_raw)

        rnd = random.Random(self.seed)
        style_pool = ["short", "medium", "verbose"]
        noise_pool = ["low", "medium", "high"]
        want_two_q = allow_two_questions and (len(questions) >= 2)

        gen_cases: List[Dict[str, Any]] = []
        for i in range(variants):
            nl = min(max(noise_level, 0), 2)
            ask_prob = [0.10, 0.40, 1.00][nl]
            gen_cases.append(
                {
                    "variant_index": i + 1,
                    "answer_volume": rnd.choice(style_pool),
                    "noise": noise_pool[nl],
                    "mix_two_questions": want_two_q and (rnd.random() < 0.55),
                    "answer_indirect": rnd.random() < (0.25 + 0.15 * nl),
                    "include_extra_chitchat": rnd.random() < (0.20 + 0.20 * nl),
                    "include_link_or_nda": rnd.random() < (0.10 + 0.20 * nl),
                    "format_synonyms": True,
                    "salary_synonyms": True,
                    "candidate_asks_questions": (rnd.random() < ask_prob),
                    "candidate_questions_count": (1 if nl == 0 else (rnd.randint(1, 2) if nl == 1 else rnd.randint(1, 3))),
                    "candidate_questions_topics": ["роль", "подчинение", "команда", "нагрузка", "процессы", "стек", "задачи"],

                }
            )

        payload: Dict[str, Any] = {
            "vacancy": {
                "title": vacancy.get("title"),
                "company_name": vacancy.get("company_name"),
                "company_description": vacancy.get("company_description"),
                "company_industry": vacancy.get("company_industry"),
                "location": vacancy.get("location"),
                "work_format": vacancy.get("work_format"),
                "salary_range_from": vacancy.get("salary_range_from"),
                "salary_range_to": vacancy.get("salary_range_to"),
                "responsibilities": vacancy.get("responsibilities"),
                "vacancy_stack": vacancy.get("vacancy_stack"),
                "vacancy_skills": vacancy.get("vacancy_skills"),
                "questions": questions,
            },
            "candidate": {
                "recruiter_name": candidate.get("recruiter_name"),
                "candidate_name": candidate.get("candidate_name"),
                "candidate_job_list": candidate.get("candidate_job_list"),
                "candidate_skills": candidate.get("candidate_skills"),
            },
            "variants": gen_cases,
        }

        if prior_dialogues:
            payload["avoid_repeating"] = prior_dialogues[:5]

        return payload

    def _call(self, instruction: str, payload: Dict[str, Any]) -> Any:
        resp = self.client.responses.create(
            model=self.model,
            input=instruction + "\n\n" + json.dumps(payload, ensure_ascii=False),
        )
        self.last_usage = getattr(resp, "usage", None)
        text = (getattr(resp, "output_text", "") or "").strip()
        return _safe_json_loads(text)

    def synthesize(
        self,
        cdm: Dict[str, Any],
        variants: int,
        noise_level: int,
        allow_two_questions: bool,
    ) -> List[str]:
        out_dialogues: List[str] = []
        prior_for_retry: List[str] = []

        for attempt in range(DIALOGUE_GEN_MAX_RETRIES + 1):
            need = variants - len(out_dialogues)
            if need <= 0:
                break

            instruction = self._instruction(need)
            payload = self._payload(
                cdm=cdm,
                variants=need,
                noise_level=noise_level,
                allow_two_questions=allow_two_questions,
                prior_dialogues=prior_for_retry if prior_for_retry else None,
            )

            data = self._call(instruction, payload)
            if not isinstance(data, list):
                raise ValueError("dialogue generator did not return a JSON array")

            for item in data:
                if len(out_dialogues) >= variants:
                    break
                if not isinstance(item, dict):
                    continue
                turns = item.get("turns")
                if not _validate_turns_structure(turns):
                    continue
                dlg = _format_dialogue(turns)
                if not dlg:
                    continue
                out_dialogues.append(dlg)

            if len(out_dialogues) < variants and attempt < DIALOGUE_GEN_MAX_RETRIES:
                prior_for_retry = out_dialogues[:]
                continue

        if len(out_dialogues) < variants:
            raise ValueError(f"dialogue generator returned only {len(out_dialogues)}/{variants} dialogues")

        return out_dialogues


class ScreeningAutofillPromptRunner:
    def __init__(self, prompt_id: str, prompt_version: Optional[str]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def run_once(self, dialogue: str) -> str:
        payload = "\n".join(
            [
                "Fill the screening form based on the dialogue below.",
                "",
                dialogue.strip(),
            ]
        ).strip()

        resp = self.client.responses.create(
            prompt=self.prompt,
            input=payload,
        )
        self.last_usage = getattr(resp, "usage", None)
        return (getattr(resp, "output_text", "") or "").strip()


def _run_single_autofill_case(
    autofill: ScreeningAutofillPromptRunner,
    dialogue: str,
    flatten_like_prod: bool,
    token_usage_total: Dict[str, int],
    expected_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    final_dialogue = _flatten_like_prod(dialogue) if flatten_like_prod else dialogue

    parsed: Any = None
    schema_errors: List[str] = []
    semantic_errors: List[str] = []
    expectation_errors: List[str] = []
    error: Optional[str] = None

    try:
        raw_out = autofill.run_once(final_dialogue)
        _accumulate_usage(token_usage_total, autofill.last_usage)

        parsed = _safe_json_loads(raw_out)
        schema_errors = _validate_schema(parsed)
        expectation_errors = _validate_expected_json_subset(parsed, expected_json)
        if not schema_errors:
            semantic_errors = _semantic_validate(final_dialogue, parsed)
    except Exception as e:
        error = repr(e)

    if isinstance(parsed, dict) and not schema_errors:
        parsed = _ordered_parsed_json(parsed)

    return {
        "dialogue": final_dialogue,
        "parsed_json": parsed,
        "schema_errors": schema_errors or [],
        "semantic_errors": semantic_errors or [],
        "expectation_errors": expectation_errors or [],
        "error": error,
    }


def _collect_errors(*error_groups: Optional[List[str]], exc: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for group in error_groups:
        for e in group or []:
            if e:
                out.append(e)
    if exc:
        out.append(f"exception:{exc}")
    return out


def run_autofill_from_cdm(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    variants_per_cdm: int,
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    dialogue_gen_model: Optional[str],
    noise_level: int,
    allow_two_questions: bool,
    flatten_like_prod: bool,
    seed: Optional[int],
    quiet: bool,
    include_regression_cases: bool = False,
    regression_only: bool = False,
    regression_case_names: Optional[Set[str]] = None,
) -> pathlib.Path:
    ensure_dirs()

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_generated_cases = not regression_only
    run_regression_cases = bool(include_regression_cases or regression_only or regression_case_names)
    selected_regression_cases = _select_regression_cases(regression_case_names) if run_regression_cases else []

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
        f"cdm_count={cdm_count} "
        f"variants_per_cdm={variants_per_cdm} "
        f"noise_level={noise_level} "
        f"allow_two_questions={allow_two_questions} "
        f"flatten_like_prod={flatten_like_prod} "
        f"seed={seed} "
        f"run_generated_cases={run_generated_cases} "
        f"run_regression_cases={run_regression_cases}",
    )

    cfg: Dict[str, Any] = {}
    if CFG_PATH.is_file():
        cfg = load_yaml(CFG_PATH) or {}
        _log(quiet, f"[init] loaded cfg: {CFG_PATH}")
    else:
        _log(quiet, f"[init] cfg not found: {CFG_PATH} (ok, will use env/cli)")

    cfg_pid, cfg_pver = _resolve_prompt_from_cfg(cfg)
    cfg_gen_model = _resolve_dialogue_gen_from_cfg(cfg)

    env_pid = os.environ.get("SCREENING_AUTOFILL_PROMPT_ID")
    env_pver = os.environ.get("SCREENING_AUTOFILL_PROMPT_VERSION")

    final_pid = prompt_id or cfg_pid or env_pid
    final_pver = prompt_version or cfg_pver or env_pver

    if not final_pid:
        raise EnvironmentError(
            "No prompt_id found. Provide --prompt-id, or set SCREENING_AUTOFILL_PROMPT_ID, "
            "or add tests/tools/model.yaml -> screening_autofill.prompt_id"
        )

    final_gen_model = dialogue_gen_model or cfg_gen_model or DEFAULT_DIALOGUE_GEN_MODEL

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count) if run_generated_cases else []

    _log(
        quiet,
        "[init] "
        f"prompt_id={final_pid} "
        f"prompt_version={final_pver} "
        f"dialogue_gen_model={final_gen_model} "
        f"dialogue_gen_retries={DIALOGUE_GEN_MAX_RETRIES} "
        f"regression_cases_selected={len(selected_regression_cases)}",
    )

    synth = DialogueSynthesizer(model=final_gen_model, seed=seed) if run_generated_cases else None
    autofill = ScreeningAutofillPromptRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()

    results: List[Dict[str, Any]] = []
    errors_by_dialogue: List[Dict[str, Any]] = []
    all_error_counts: Counter[str] = Counter()

    passed = 0
    failed = 0

    source_counts = {"cdm": 0, "regression": 0}
    total_cdm_cases = len(cdm_paths)

    if run_generated_cases:
        if synth is None:
            raise RuntimeError("dialogue synthesizer is not initialized")

        for case_idx, cdm_path in enumerate(cdm_paths, start=1):
            cdm = load_json(cdm_path)
            vacancy = cdm.get("vacancy") or {}
            v_title = vacancy.get("title")
            v_company = vacancy.get("company_name") if "company_name" in vacancy else vacancy.get("company_name")

            _log(
                quiet,
                f"[run] case {case_idx}/{total_cdm_cases} ({cdm_path.name}) title={v_title} company={v_company}",
            )

            try:
                dialogues = synth.synthesize(
                    cdm=cdm,
                    variants=variants_per_cdm,
                    noise_level=noise_level,
                    allow_two_questions=allow_two_questions,
                )
                _accumulate_usage(token_usage_total, synth.last_usage)
            except Exception as e:
                err = repr(e)
                failed += 1

                _log(quiet, f"[warn] dialogue synthesis failed: {cdm_path.name}: {err}")

                result = {
                    "case_source": "cdm",
                    "case_name": cdm_path.name,
                    "case_description": None,
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "variant_index": None,
                    "dialogue": "",
                    "parsed_json": None,
                    "expected_json": None,
                    "schema_errors": ["dialogue_synthesis_failed"],
                    "semantic_errors": ["dialogue_synthesis_failed"],
                    "expectation_errors": [],
                    "error": err,
                }
                results.append(result)
                source_counts["cdm"] += 1
                errors_by_dialogue.append(
                    {
                        "cdm_file": str(cdm_path),
                        "variant_index": None,
                        "errors": ["dialogue_synthesis_failed", f"exception:{err}"],
                    }
                )
                all_error_counts["dialogue_synthesis_failed"] += 1
                continue

            for v_idx, dialogue in enumerate(dialogues, start=1):
                _log(quiet, f"  [variant {v_idx}/{variants_per_cdm}] running screening_autofill...")

                evaluated = _run_single_autofill_case(
                    autofill=autofill,
                    dialogue=dialogue,
                    flatten_like_prod=flatten_like_prod,
                    token_usage_total=token_usage_total,
                )
                schema_errors = evaluated["schema_errors"]
                semantic_errors = evaluated["semantic_errors"]
                expectation_errors = evaluated["expectation_errors"]
                error = evaluated["error"]

                combined_errors = _collect_errors(
                    schema_errors,
                    semantic_errors,
                    expectation_errors,
                    exc=error,
                )

                if combined_errors:
                    failed += 1
                    errors_by_dialogue.append(
                        {
                            "cdm_file": str(cdm_path),
                            "variant_index": v_idx,
                            "errors": combined_errors,
                        }
                    )
                    for ce in combined_errors:
                        all_error_counts[ce] += 1

                    if error is not None:
                        _log(quiet, f"    [fail] error={error}")
                    elif schema_errors:
                        _log(quiet, f"    [fail] schema_errors={schema_errors}")
                    elif expectation_errors:
                        _log(quiet, f"    [fail] expectation_errors={expectation_errors}")
                    else:
                        _log(quiet, f"    [fail] semantic_errors={semantic_errors}")
                else:
                    passed += 1
                    _log(quiet, "    [ok] semantic_errors=[]")

                results.append(
                    {
                        "case_source": "cdm",
                        "case_name": cdm_path.name,
                        "case_description": None,
                        "cdm_file": str(cdm_path),
                        "vacancy_title": v_title,
                        "vacancy_company": v_company,
                        "variant_index": v_idx,
                        "dialogue": evaluated["dialogue"],
                        "parsed_json": evaluated["parsed_json"],
                        "expected_json": None,
                        "schema_errors": schema_errors,
                        "semantic_errors": semantic_errors,
                        "expectation_errors": expectation_errors,
                        "error": error,
                    }
                )
                source_counts["cdm"] += 1

    if run_regression_cases:
        total_regression_cases = len(selected_regression_cases)
        for case_idx, regression_case in enumerate(selected_regression_cases, start=1):
            case_name = str(regression_case.get("name") or f"regression_{case_idx:04d}")
            case_ref = f"regression_case::{case_name}"
            description = regression_case.get("description")
            dialogue = str(regression_case.get("dialogue") or "").strip()
            expected_json = regression_case.get("expected_json")
            v_title = regression_case.get("vacancy_title")
            v_company = regression_case.get("vacancy_company")

            _log(
                quiet,
                f"[run] regression {case_idx}/{total_regression_cases} "
                f"({case_name}) expected={expected_json}",
            )

            if not dialogue:
                err = "empty_regression_dialogue"
                failed += 1
                errors_by_dialogue.append(
                    {
                        "cdm_file": case_ref,
                        "variant_index": 1,
                        "errors": [err],
                    }
                )
                all_error_counts[err] += 1
                results.append(
                    {
                        "case_source": "regression",
                        "case_name": case_name,
                        "case_description": description,
                        "cdm_file": case_ref,
                        "vacancy_title": v_title,
                        "vacancy_company": v_company,
                        "variant_index": 1,
                        "dialogue": dialogue,
                        "parsed_json": None,
                        "expected_json": expected_json,
                        "schema_errors": [],
                        "semantic_errors": [],
                        "expectation_errors": [err],
                        "error": None,
                    }
                )
                source_counts["regression"] += 1
                continue

            evaluated = _run_single_autofill_case(
                autofill=autofill,
                dialogue=dialogue,
                flatten_like_prod=flatten_like_prod,
                token_usage_total=token_usage_total,
                expected_json=expected_json if isinstance(expected_json, dict) else None,
            )
            schema_errors = evaluated["schema_errors"]
            semantic_errors = evaluated["semantic_errors"]
            expectation_errors = evaluated["expectation_errors"]
            error = evaluated["error"]
            combined_errors = _collect_errors(
                schema_errors,
                semantic_errors,
                expectation_errors,
                exc=error,
            )

            if combined_errors:
                failed += 1
                errors_by_dialogue.append(
                    {
                        "cdm_file": case_ref,
                        "variant_index": 1,
                        "errors": combined_errors,
                    }
                )
                for ce in combined_errors:
                    all_error_counts[ce] += 1

                if error is not None:
                    _log(quiet, f"    [fail] error={error}")
                elif schema_errors:
                    _log(quiet, f"    [fail] schema_errors={schema_errors}")
                elif expectation_errors:
                    _log(quiet, f"    [fail] expectation_errors={expectation_errors}")
                else:
                    _log(quiet, f"    [fail] semantic_errors={semantic_errors}")
            else:
                passed += 1
                _log(quiet, "    [ok] regression matched expected_json")

            results.append(
                {
                    "case_source": "regression",
                    "case_name": case_name,
                    "case_description": description,
                    "cdm_file": case_ref,
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "variant_index": 1,
                    "dialogue": evaluated["dialogue"],
                    "parsed_json": evaluated["parsed_json"],
                    "expected_json": expected_json,
                    "schema_errors": schema_errors,
                    "semantic_errors": semantic_errors,
                    "expectation_errors": expectation_errors,
                    "error": error,
                }
            )
            source_counts["regression"] += 1

    total = passed + failed
    pass_rate = round((passed / total * 100.0), 2) if total else 0.0

    # ---- NEW report structure: cases / mismatches / errors (verdict-like) ----
    cases: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for r in results:
        schema_errors = r.get("schema_errors") or []
        semantic_errors = r.get("semantic_errors") or []
        expectation_errors = r.get("expectation_errors") or []
        exc = r.get("error")
        combined = _collect_errors(schema_errors, semantic_errors, expectation_errors, exc=exc)

        match = len(combined) == 0

        case = {
            "match": match,
            "case_source": r.get("case_source"),
            "case_name": r.get("case_name"),
            "case_description": r.get("case_description"),
            "cdm_file": r.get("cdm_file"),
            "variant_index": r.get("variant_index"),
            "vacancy_title": r.get("vacancy_title"),
            "vacancy_company": r.get("vacancy_company"),
            "dialogue": r.get("dialogue"),
            "parsed_json": r.get("parsed_json"),
            "expected_json": r.get("expected_json"),
            "schema_errors": schema_errors,
            "semantic_errors": semantic_errors,
            "expectation_errors": expectation_errors,
            "error": exc,
        }
        cases.append(case)

        if not match:
            mismatches.append(
                {
                    "case_source": case["case_source"],
                    "case_name": case["case_name"],
                    "case_description": case["case_description"],
                    "cdm_file": case["cdm_file"],
                    "variant_index": case["variant_index"],
                    "vacancy_title": case["vacancy_title"],
                    "vacancy_company": case["vacancy_company"],
                    "errors": combined,
                    "dialogue": case["dialogue"],
                    "parsed_json": case["parsed_json"],
                    "expected_json": case["expected_json"],
                }
            )

        # "errors" bucket: exceptions or synthesis failures (procedural errors)
        if exc is not None:
            errors.append(
                {
                    "case_source": case["case_source"],
                    "case_name": case["case_name"],
                    "cdm_file": case["cdm_file"],
                    "variant_index": case["variant_index"],
                    "vacancy_title": case["vacancy_title"],
                    "vacancy_company": case["vacancy_company"],
                    "error": exc,
                }
            )
        elif "dialogue_synthesis_failed" in schema_errors or "dialogue_synthesis_failed" in semantic_errors:
            errors.append(
                {
                    "case_source": case["case_source"],
                    "case_name": case["case_name"],
                    "cdm_file": case["cdm_file"],
                    "variant_index": case["variant_index"],
                    "vacancy_title": case["vacancy_title"],
                    "vacancy_company": case["vacancy_company"],
                    "error": "dialogue_synthesis_failed",
                }
            )

    summary = {
        # legacy-compatible
        "results_total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "errors_by_dialogue": errors_by_dialogue,
        # verdict-like
        "total_cases": len(cases),
        "mismatches_count": len(mismatches),
        "errors_count": len(errors),
        "error_counts": dict(all_error_counts),
        "source_counts": source_counts,
    }

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_count": cdm_count,
        "variants_per_cdm": variants_per_cdm,
        "noise_level": noise_level,
        "allow_two_questions": allow_two_questions,
        "flatten_like_prod": flatten_like_prod,
        "seed": seed,
        "include_regression_cases": include_regression_cases,
        "regression_only": regression_only,
        "regression_case_names": sorted(regression_case_names) if regression_case_names else None,
        "regression_cases_selected": [case.get("name") for case in selected_regression_cases],
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "dialogue_gen_model": final_gen_model,
        "dialogue_gen_retries": DIALOGUE_GEN_MAX_RETRIES,
        "token_usage_total": token_usage_total,
        "summary": summary,

        # NEW: verdict-like top-level lists
        "cases": cases,
        "mismatches": mismatches,
        "errors": errors,

        # OLD: keep for backward compatibility
        "results": results,
    }

    out_path = REPORTS_DIR / f"screening_autofill_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"results_total={len(results)} "
        f"passed={passed} "
        f"failed={failed} "
        f"pass_rate={pass_rate:.2f} "
        f"mismatches={len(mismatches)} "
        f"errors={len(errors)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run screening_autofill_prompt on synthesized CDM dialogues and deterministic regression cases."
    )
    parser.add_argument(
        "--cdm-dir",
        type=str,
        default=str(DEFAULT_CDM_DIR),
        help=f"CDM fixtures dir (default: {DEFAULT_CDM_DIR})",
    )
    parser.add_argument(
        "--cdm-count",
        type=int,
        default=DEFAULT_CDM_COUNT,
        help="Take first N CDM fixtures (sorted by filename). Default: all.",
    )
    parser.add_argument(
        "--variants-per-cdm",
        type=int,
        default=DEFAULT_VARIANTS_PER_CDM,
        help=f"How many dialogue variants to synthesize per CDM (default: {DEFAULT_VARIANTS_PER_CDM}).",
    )
    parser.add_argument(
        "--noise-level",
        type=int,
        default=2,
        help="0..2. Higher means more noise, longer answers, more indirectness.",
    )
    parser.add_argument(
        "--allow-two-questions",
        action="store_true",
        help="Allow recruiter messages to contain up to 2 questions (generator decides per variant).",
    )
    parser.add_argument(
        "--flatten-like-prod",
        action="store_true",
        help="Flatten dialogue to one line with spaces, similar to production wrap-up.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for variant settings (for reproducibility).",
    )
    parser.add_argument(
        "--dialogue-gen-model",
        type=str,
        default=None,
        help=f"Dialogue generation model override (default: {DEFAULT_DIALOGUE_GEN_MODEL}).",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Override screening_autofill prompt id (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override screening_autofill prompt version (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console progress output.",
    )
    parser.add_argument(
        "--include-regression-cases",
        action="store_true",
        help="Also run built-in deterministic regression dialogues for work_format extraction.",
    )
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="Run only built-in deterministic regression dialogues and skip CDM synthesis.",
    )
    parser.add_argument(
        "--regression-case-names",
        type=str,
        default=None,
        help="Comma-separated built-in regression case names to run (implies regression cases).",
    )

    args = parser.parse_args()
    regression_case_names = _parse_case_names_filter(args.regression_case_names)

    run_autofill_from_cdm(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        variants_per_cdm=args.variants_per_cdm,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        dialogue_gen_model=args.dialogue_gen_model,
        noise_level=args.noise_level,
        allow_two_questions=bool(args.allow_two_questions),
        flatten_like_prod=bool(args.flatten_like_prod),
        seed=args.seed,
        quiet=bool(args.quiet),
        include_regression_cases=bool(args.include_regression_cases),
        regression_only=bool(args.regression_only),
        regression_case_names=regression_case_names,
    )


if __name__ == "__main__":
    main()

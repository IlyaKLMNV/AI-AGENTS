# message_classifier_runner.py
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
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

# Repo root: if this file is in app/, parents[1] is repo root.
ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm" / "std"
DEFAULT_REGRESSION_CASES_PATH = ROOT / "tests" / "fixtures" / "message_classifier" / "regression_cases.json"
REPORTS_DIR = ROOT / "tests" / "reports" / "message_classifier"
TEXT_FILE_ENCODING = "utf-8-sig"

DEFAULT_MESSAGE_GEN_MODEL = "gpt-4.1-mini"
MESSAGE_GEN_MAX_RETRIES = 1

CLASSES = ("reason_farewell", "no_reason", "acceptance", "human_needed")


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding=TEXT_FILE_ENCODING))


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding=TEXT_FILE_ENCODING))


def load_cdm_files(cdm_dir: pathlib.Path, cdm_count: Optional[int]) -> List[pathlib.Path]:
    if not cdm_dir.exists():
        raise FileNotFoundError(f"CDM dir not found: {cdm_dir}")

    paths = [pathlib.Path(p) for p in sorted(glob.glob(str(cdm_dir / "cdm_*.json")))]
    if not paths and (cdm_dir / "std").is_dir():
        paths = [pathlib.Path(p) for p in sorted(glob.glob(str((cdm_dir / "std") / "cdm_*.json")))]
    if not paths:
        raise FileNotFoundError(f"No cdm_*.json found in: {cdm_dir}")

    if cdm_count is not None:
        if cdm_count <= 0:
            raise ValueError("--cdm-count must be > 0")
        paths = paths[:cdm_count]

    return paths


def load_regression_cases(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Regression cases file not found: {path}")

    raw = json.loads(path.read_text(encoding=TEXT_FILE_ENCODING))
    if not isinstance(raw, list):
        raise ValueError("Regression cases JSON must be a list")

    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Regression case #{idx} must be an object")

        case_id = str(item.get("id") or "").strip()
        target_class = str(item.get("target_class") or "").strip().lower()
        message = str(item.get("message") or "").strip()

        if not case_id:
            raise ValueError(f"Regression case #{idx} is missing required field 'id'")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate regression case id: {case_id}")
        if target_class not in CLASSES:
            raise ValueError(
                f"Regression case '{case_id}' has invalid target_class={target_class!r}; "
                f"expected one of {CLASSES}"
            )
        if not message:
            raise ValueError(f"Regression case '{case_id}' is missing required field 'message'")

        seen_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "target_class": target_class,
                "scenario": str(item.get("scenario") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "message": message,
            }
        )

    return cases


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        it = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input_token_count") or 0
        ot = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output_token_count") or 0
        tt = usage.get("total_tokens") or usage.get("token_count")
    else:
        it = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or getattr(usage, "input_token_count", None) or 0
        ot = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or getattr(usage, "output_token_count", None) or 0
        tt = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)

    if tt is None:
        tt = (it or 0) + (ot or 0)

    return int(it or 0), int(ot or 0), int(tt or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    it, ot, tt = _extract_usage_numbers(usage)
    bucket["input_tokens"] += it
    bucket["output_tokens"] += ot
    bucket["total_tokens"] += tt


def _resolve_prompt_from_cfg(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    block = cfg.get("message_classifier") or {}
    pid = block.get("prompt_id")
    pver = block.get("prompt_version")
    seed = block.get("seed")
    return (str(pid) if pid else None, str(pver) if pver else None, int(seed) if seed is not None else None)


def _resolve_message_gen_model_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    # Optional in tests/tools/model.yaml:
    # message_classifier:
    #   message_gen_model: gpt-4.1-mini
    block = cfg.get("message_classifier") or {}
    m = block.get("message_gen_model")
    return str(m) if m else None


def _extract_label(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    m = re.search(r"\b(reason_farewell|no_reason|acceptance|human_needed)\b", t)
    return m.group(1) if m else None


DECLINE_PATTERNS = (
    r"\bне\s+интерес",
    r"\bне\s+рассматрива",
    r"\bне\s+подходит",
    r"\bвынужден\s+отказ",
    r"\bоткаж",
    r"\bотказ",
    r"\bне\s+готов",
    r"\bне\s+смогу",
    r"\bнет,\s*спасибо\b",
)
REASON_PATTERNS = (
    r"\bпотому\s+что\b",
    r"\bтак\s+как\b",
    r"\bпоскольку\b",
    r"\bуже\b",
    r"\bоффер",
    r"\bзарплат",
    r"\bформат",
    r"\bофис",
    r"\bгибрид",
    r"\bудален",
    r"\bлокац",
    r"\bпереезд",
    r"\bстек",
    r"\bсфера",
    r"\bработаю\b",
    r"\bвышел\s+на\s+работу\b",
    r"\bпринял\s+оффер\b",
)
ACCEPTANCE_PATTERNS = (
    r"\bинтерес",
    r"\bваканси",
    r"\bподскажите\b",
    r"\bрасскажите\b",
    r"\bможете\s+уточнить\b",
    r"\bкакая\s+компания\b",
    r"\bкак\s+ваша\s+компания\s+называется\b",
    r"\bссылка\s+на\s+ваканси",
    r"\bописани[ея]\b",
    r"\bкоманд",
    r"\bзадач",
    r"\bстек",
    r"\bформат",
    r"\bзарплат",
    r"\bсозвон",
    r"\bготов\s+обсудить\b",
)
HUMAN_NEEDED_PATTERNS = (
    r"\bстранн",
    r"\bчто\s+за\s+ерунд",
    r"\bмошенн",
    r"\bразвод",
    r"\bденьги\b",
    r"\bскиньте\b",
    r"\bоткуда\s+нашли\s+контакт\b",
    r"\bзачем\s+мне\s+тратить\s+время\b",
    r"\bне\s+совсем\s+понимаю\b",
    r"\bбред\b",
    r"\bхрень\b",
    r"\bено[тт]\b",
    r"[🦝😕🤨]",
)


def _has_any_pattern(text: str, patterns: Tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _validate_generated_message(target_class: str, message: str) -> Optional[str]:
    text = message.strip()
    if not text:
        return "generated empty message"

    has_decline = _has_any_pattern(text, DECLINE_PATTERNS)
    has_reason = _has_any_pattern(text, REASON_PATTERNS)
    has_acceptance = _has_any_pattern(text, ACCEPTANCE_PATTERNS) or "?" in text
    has_human_needed = _has_any_pattern(text, HUMAN_NEEDED_PATTERNS)

    if target_class == "reason_farewell":
        if not has_decline:
            return "reason_farewell message has no explicit refusal"
        if not has_reason:
            return "reason_farewell message has no clear reason"
        return None

    if target_class == "no_reason":
        if not has_decline:
            return "no_reason message has no explicit refusal"
        if has_reason:
            return "no_reason message leaks a reason"
        return None

    if target_class == "acceptance":
        if has_decline:
            return "acceptance message contains refusal markers"
        if has_human_needed:
            return "acceptance message contains human_needed markers"
        if not has_acceptance:
            return "acceptance message lacks clear interest or relevant vacancy question"
        return None

    if target_class == "human_needed":
        if has_decline:
            return "human_needed message looks like a refusal instead of escalation"
        if has_acceptance and not has_human_needed:
            return "human_needed message looks like a normal acceptance/clarification"
        if not has_human_needed:
            return "human_needed message lacks clear escalation markers"
        return None

    return None


SCENARIO_HINTS_BY_CLASS: Dict[str, List[str]] = {
    "reason_farewell": [
        "Вежливый отказ с причиной: уже вышел на работу/принял оффер.",
        "Отказ с причиной: не рассматривает смену сферы/не тот стек.",
        "Отказ с причиной: не подходит формат (офис/гибрид), не готов к переезду.",
        "Отказ с причиной: ожидания по зарплате выше, чем обычно предлагают.",
        "Отказ с причиной: сейчас не в поиске, вернется позже.",
    ],
    "no_reason": [
        "Короткий отказ без объяснений: 'не интересно/не подходит/нет, спасибо'.",
        "Формальный отказ без причины: 'вынужден отказаться, спасибо'.",
        "Очень кратко: 'нет'.",
    ],
    "acceptance": [
        "Кандидат согласен и готов созвониться: предлагает время.",
        "Кандидат заинтересован и просит детали по вилке/графику/формату.",
        "Кандидат задает релевантный вопрос по вакансии (обязанности/команда/стек) и выражает интерес.",
        "Кандидат просит прислать описание и подтверждает интерес.",
        "Кандидат просит ссылку на вакансию или название компании в нейтральной деловой форме.",
        "Кандидат задает спокойные вопросы по вакансии (обязанности/команда/стек) без негатива и подозрений.",
        "Кандидат вежливо спрашивает про формат работы и ориентир по компенсации.",
    ],
    "human_needed": [
        "Раздражение/жалоба/негатив к рекрутеру или компании.",
        "Странные или нерелевантные вопросы (не про вакансию), либо непонятный смысл.",
        "Просьба денег/мошеннический оттенок/обвинения, без явного согласия или отказа.",
        "Сообщение не по теме или набор слов/эмодзи так, что смысл неясен.",
        "Резкий или подозрительный вопрос о том, откуда взяли контакт.",
        "Нейтральные вопросы про вакансию здесь запрещены: нужен явный дискомфорт, подозрение или странность.",
    ],
}


def _class_generation_requirements(target_class: str) -> str:
    if target_class == "reason_farewell":
        return (
            "Hard requirements for reason_farewell:\n"
            "- Include an explicit refusal.\n"
            "- Use direct refusal wording such as 'не рассматриваю', 'вынужден отказаться', 'мне не подходит', 'не готов'.\n"
            "- Add one short concrete reason: salary, format, location, stack, sphere, accepted offer, already working, not looking now.\n"
            "- Do not ask questions.\n"
            "- Do not sound ambiguous."
        )
    if target_class == "no_reason":
        return (
            "Hard requirements for no_reason:\n"
            "- Include an explicit refusal.\n"
            "- Do not include any reason or explanation.\n"
            "- Do not mention salary, format, location, stack, current work, offers, or plans.\n"
            "- Do not ask questions.\n"
            "- Keep it short and final."
        )
    if target_class == "acceptance":
        return (
            "Hard requirements for acceptance:\n"
            "- Show clear interest in the vacancy.\n"
            "- You may ask 1-2 normal business questions.\n"
            "- Allowed topics: company name, vacancy link, vacancy description, team, responsibilities, stack, format, schedule, salary range, next steps.\n"
            "- Tone must be calm, constructive, and business-like.\n"
            "- No irritation, suspicion, accusations, contact-source complaints, money requests, nonsense, or hostility.\n"
            "- Do not make it mixed or borderline."
        )
    return (
        "Hard requirements for human_needed:\n"
        "- The message must require manual handling because it is suspicious, irritated, accusatory, confusing, scam-like, off-topic, or strange.\n"
        "- It must not look like a normal vacancy clarification.\n"
        "- Avoid ordinary neutral questions about company name, vacancy link, salary, format, team, stack, or responsibilities unless they are clearly framed with irritation or suspicion.\n"
        "- If using contact-source theme, make it sharp or uncomfortable, not neutral.\n"
        "- Do not turn it into a clean refusal."
    )


def _class_generation_examples(target_class: str) -> str:
    if target_class == "reason_farewell":
        return "Example: Спасибо за предложение, но мне не подходит офисный формат, поэтому вынужден отказаться."
    if target_class == "no_reason":
        return "Example: Спасибо, но вынужден отказаться."
    if target_class == "acceptance":
        return "Example: Здравствуйте! Вакансия выглядит интересно. Можете прислать ссылку на описание позиции и уточнить формат работы?"
    return "Example: Откуда вы вообще взяли мой контакт и почему пишете без предупреждения?"


def _pick_scenario_hint(
    target_class: str,
    rng: random.Random,
    scenario_mode: str,
    scenario_count_per_class: Optional[int],
    cycle_state: Dict[str, int],
) -> str:
    pool = SCENARIO_HINTS_BY_CLASS.get(target_class) or ["Нейтральное сообщение."]
    if scenario_count_per_class is not None and scenario_count_per_class > 0 and scenario_count_per_class < len(pool):
        pool = pool[:scenario_count_per_class]

    if scenario_mode == "random":
        return rng.choice(pool)

    idx = cycle_state.get(target_class, 0) % len(pool)
    cycle_state[target_class] = idx + 1
    return pool[idx]


class CandidateMessageSynthesizer:
    """
    Generates ONE Russian candidate message with a known TARGET_CLASS.
    This is the "ground truth label" used to test message_classifier prompt.

    Important: message_classifier should NOT be used to build the dataset.
    """

    def __init__(self, model: str, seed: Optional[int]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.seed = seed
        self.last_usage: Any = None

    def _instruction(self) -> str:
        return (
            "You generate exactly ONE candidate message in Russian after the recruiter's first outreach.\n"
            "You will be given TARGET_CLASS: reason_farewell / no_reason / acceptance / human_needed.\n"
            "Generate one message so that it is unambiguous and easy to classify into TARGET_CLASS.\n\n"
            "Global rules:\n"
            "- Return only the candidate message text.\n"
            "- No JSON, no quotes, no markdown, no explanations.\n"
            "- Keep the message realistic and concise.\n"
            "- Avoid class ambiguity.\n"
            "- Follow the class-specific hard requirements exactly.\n"
        )

    def _payload(self, cdm: Dict[str, Any], target_class: str, scenario_hint: str, noise_level: int) -> str:
        vacancy = cdm.get("vacancy") or {}
        candidate = cdm.get("candidate") or {}

        noise_desc = ["low", "medium", "high"][min(max(noise_level, 0), 2)]

        ctx = {
            "TARGET_CLASS": target_class,
            "SCENARIO_HINT": scenario_hint,
            "noise_level": noise_desc,
            "vacancy": {
                "title": vacancy.get("title"),
                "company_name": vacancy.get("company_name"),
                "company_description": vacancy.get("company_description") or vacancy.get("firm_description"),
                "responsibilities": vacancy.get("responsibilities"),
                "work_format": vacancy.get("work_format"),
                "location": vacancy.get("location"),
                "salary_range_from": vacancy.get("salary_range_from"),
                "salary_range_to": vacancy.get("salary_range_to"),
                "salary": vacancy.get("salary"),
                "stack": vacancy.get("vacancy_stack") or vacancy.get("stack"),
            },
            "candidate": {
                "candidate_name": candidate.get("candidate_name"),
                "candidate_job_list": candidate.get("candidate_job_list"),
                "candidate_skills": candidate.get("candidate_skills"),
            },
        }

        return (
            "CONTEXT_JSON:\n"
            f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
            "INSTRUCTIONS:\n"
            f"1) TARGET_CLASS = {target_class}\n"
            f"2) SCENARIO_HINT = {scenario_hint}\n"
            "3) Use vacancy context only to make the message realistic.\n"
            "4) Do not invent a different class than requested.\n"
            f"5) {_class_generation_requirements(target_class)}\n"
            f"6) {_class_generation_examples(target_class)}\n"
            "7) Return exactly one message in Russian.\n"
        )

    def synthesize_one(self, cdm: Dict[str, Any], target_class: str, scenario_hint: str, noise_level: int) -> str:
        instruction = self._instruction()
        payload = self._payload(cdm=cdm, target_class=target_class, scenario_hint=scenario_hint, noise_level=noise_level)

        resp = self.client.responses.create(
            model=self.model,
            input=instruction + "\n\n" + payload,
        )
        self.last_usage = getattr(resp, "usage", None)
        text = (getattr(resp, "output_text", "") or "").strip()

        if not text or len(text) < 1:
            raise ValueError("message generator returned empty message")
        if "\n" in text.strip():
            text = " ".join(x.strip() for x in text.splitlines() if x.strip()).strip()
        return text


class MessageClassifierRunner:
    def __init__(self, prompt_id: str, prompt_version: Optional[str]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None
        self.last_raw_output: str = ""

    def classify(self, message: str) -> str:
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=message.strip(),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
        self.last_raw_output = raw
        label = _extract_label(raw)
        if label not in CLASSES:
            raise ValueError(f"message_classifier returned invalid output: {raw!r}")
        return label


def _confusion_matrix(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    m: Dict[str, Dict[str, int]] = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for c in cases:
        t = c.get("target_class")
        p = c.get("predicted_class")
        if t in CLASSES and p in CLASSES:
            m[t][p] += 1
    return m


def _accuracy(cases: List[Dict[str, Any]]) -> Optional[float]:
    if not cases:
        return None
    ok = sum(1 for c in cases if c.get("target_class") == c.get("predicted_class"))
    return round(ok / len(cases) * 100.0, 2)


def _per_class_accuracy(cases: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for cls in CLASSES:
        items = [c for c in cases if c.get("target_class") == cls]
        if not items:
            out[cls] = None
            continue
        ok = sum(1 for c in items if c.get("target_class") == c.get("predicted_class"))
        out[cls] = round(ok / len(items) * 100.0, 2)
    return out


def _counts_by_key(cases: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(c.get(key)) for c in cases if c.get(key) in CLASSES))


def _mismatches_from_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cases if not c.get("match")]


def run_message_classifier_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    messages_per_class: int,
    mode: str,
    regression_cases_path: Optional[pathlib.Path],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    message_gen_model: Optional[str],
    noise_level: int,
    seed: Optional[int],
    scenario_mode: str,
    scenario_count_per_class: Optional[int],
    max_attempts_multiplier: int,
    quiet: bool,
) -> pathlib.Path:
    ensure_dirs()

    if mode not in ("synthetic", "regression", "all"):
        raise ValueError("--mode must be synthetic|regression|all")
    if mode in ("synthetic", "all") and messages_per_class <= 0:
        raise ValueError("--messages-per-class must be > 0 when mode includes synthetic")
    if scenario_mode not in ("random", "cycle"):
        raise ValueError("--scenario-mode must be random|cycle")
    if max_attempts_multiplier <= 0:
        raise ValueError("--max-attempts-multiplier must be > 0")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    cfg: Dict[str, Any] = {}
    if CFG_PATH.is_file():
        cfg = load_yaml(CFG_PATH) or {}
        _log(quiet, f"[init] loaded cfg: {CFG_PATH}")
    else:
        _log(quiet, f"[init] cfg not found: {CFG_PATH} (ok, will use env/cli)")

    cfg_pid, cfg_pver, cfg_seed = _resolve_prompt_from_cfg(cfg)
    cfg_gen_model = _resolve_message_gen_model_from_cfg(cfg)

    env_pid = os.environ.get("MESSAGE_CLASSIFIER_PROMPT_ID")
    env_pver = os.environ.get("MESSAGE_CLASSIFIER_PROMPT_VERSION")

    final_pid = prompt_id or cfg_pid or env_pid
    final_pver = prompt_version or cfg_pver or env_pver

    if not final_pid:
        raise EnvironmentError(
            "No prompt_id found. Provide --prompt-id, or set MESSAGE_CLASSIFIER_PROMPT_ID, "
            "or add tests/tools/model.yaml -> message_classifier.prompt_id"
        )

    final_seed = seed if seed is not None else cfg_seed
    final_gen_model = message_gen_model or cfg_gen_model or DEFAULT_MESSAGE_GEN_MODEL

    cdm_paths: List[pathlib.Path] = []
    if mode in ("synthetic", "all"):
        cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)

    final_regression_cases_path = regression_cases_path or DEFAULT_REGRESSION_CASES_PATH
    regression_cases: List[Dict[str, Any]] = []
    if mode in ("regression", "all"):
        regression_cases = load_regression_cases(final_regression_cases_path)

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
        f"mode={mode} "
        f"cdm_count={cdm_count} "
        f"messages_per_class={messages_per_class} "
        f"noise_level={noise_level} "
        f"seed={final_seed} "
        f"scenario_mode={scenario_mode} "
        f"scenario_count_per_class={scenario_count_per_class} "
        f"max_attempts_multiplier={max_attempts_multiplier}",
    )
    _log(
        quiet,
        "[init] "
        f"prompt_id={final_pid} "
        f"prompt_version={final_pver} "
        f"message_gen_model={final_gen_model} "
        f"message_gen_retries={MESSAGE_GEN_MAX_RETRIES}",
    )

    rng = random.Random(final_seed)
    cycle_state: Dict[str, int] = {cls: 0 for cls in CLASSES}
    synth = CandidateMessageSynthesizer(model=final_gen_model, seed=final_seed) if mode in ("synthetic", "all") else None
    clf = MessageClassifierRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()
    token_usage = {
        "message_generator": _blank_usage(),
        "message_classifier": _blank_usage(),
    }
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if mode in ("synthetic", "all"):
        assert synth is not None
        for target in CLASSES:
            need = messages_per_class
            attempts_limit = messages_per_class * max_attempts_multiplier

            _log(quiet, f"[target] {target}: need={need}")

            got = 0
            attempts = 0

            while got < need and attempts < attempts_limit:
                attempts += 1

                cdm_path = rng.choice(cdm_paths)
                cdm = load_json(cdm_path)
                vacancy = cdm.get("vacancy") or {}
                v_title = vacancy.get("title")
                v_company = vacancy.get("company_name")

                scenario_hint = _pick_scenario_hint(
                    target_class=target,
                    rng=rng,
                    scenario_mode=scenario_mode,
                    scenario_count_per_class=scenario_count_per_class,
                    cycle_state=cycle_state,
                )

                _log(quiet, f"  [gen] target={target} cdm={cdm_path.name} title={v_title} company={v_company}")
                _log(quiet, f"    [hint] {scenario_hint}")

                message = ""
                predicted: Optional[str] = None
                raw_error: Optional[str] = None

                try:
                    message = synth.synthesize_one(
                        cdm=cdm,
                        target_class=target,
                        scenario_hint=scenario_hint,
                        noise_level=noise_level,
                    )
                    validation_error = _validate_generated_message(target, message)
                    if validation_error is not None:
                        raise ValueError(validation_error)
                    _accumulate_usage(token_usage_total, synth.last_usage)
                    _accumulate_usage(token_usage["message_generator"], synth.last_usage)

                    predicted = clf.classify(message)
                    _accumulate_usage(token_usage_total, clf.last_usage)
                    _accumulate_usage(token_usage["message_classifier"], clf.last_usage)

                except Exception as e:
                    raw_error = repr(e)

                if raw_error is not None:
                    errors.append(
                        {
                            "case_type": "synthetic",
                            "target_class": target,
                            "cdm_file": str(cdm_path),
                            "scenario_hint": scenario_hint,
                            "error": raw_error,
                        }
                    )
                    _log(quiet, f"    [err] {raw_error}")
                    continue

                assert predicted is not None

                case = {
                    "case_type": "synthetic",
                    "target_class": target,
                    "predicted_class": predicted,
                    "match": bool(predicted == target),
                    "scenario_hint": scenario_hint,
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "message": message,
                    "raw_classifier_output": clf.last_raw_output,
                }
                cases.append(case)

                got += 1
                _log(quiet, f"    [ok] case={got}/{need} predicted={predicted} match={case['match']}")

            if got < need:
                raise RuntimeError(
                    f"Could not generate enough messages for target={target}: got {got}/{need} "
                    f"within {attempts_limit} attempts. Consider increasing --max-attempts-multiplier "
                    f"or adjusting scenario hints."
                )

    if mode in ("regression", "all"):
        _log(quiet, f"[regression] cases={len(regression_cases)}")
        for idx, regression_case in enumerate(regression_cases, start=1):
            predicted: Optional[str] = None
            raw_error: Optional[str] = None

            try:
                predicted = clf.classify(regression_case["message"])
                _accumulate_usage(token_usage_total, clf.last_usage)
                _accumulate_usage(token_usage["message_classifier"], clf.last_usage)
            except Exception as e:
                raw_error = repr(e)

            if raw_error is not None:
                errors.append(
                    {
                        "case_type": "regression",
                        "id": regression_case["id"],
                        "target_class": regression_case["target_class"],
                        "scenario": regression_case["scenario"],
                        "error": raw_error,
                    }
                )
                _log(quiet, f"  [reg-err] id={regression_case['id']} error={raw_error}")
                continue

            assert predicted is not None

            case = {
                "case_type": "regression",
                "id": regression_case["id"],
                "description": regression_case["description"],
                "scenario": regression_case["scenario"],
                "target_class": regression_case["target_class"],
                "predicted_class": predicted,
                "match": bool(predicted == regression_case["target_class"]),
                "message": regression_case["message"],
                "raw_classifier_output": clf.last_raw_output,
            }
            cases.append(case)
            _log(
                quiet,
                f"  [reg-ok] case={idx}/{len(regression_cases)} id={regression_case['id']} "
                f"predicted={predicted} match={case['match']}",
            )

    synthetic_cases = [c for c in cases if c.get("case_type") == "synthetic"]
    regression_result_cases = [c for c in cases if c.get("case_type") == "regression"]

    accuracy = _accuracy(cases)
    synthetic_accuracy = _accuracy(synthetic_cases)
    regression_accuracy = _accuracy(regression_result_cases)
    per_class_acc = _per_class_accuracy(cases)
    synthetic_per_class_acc = _per_class_accuracy(synthetic_cases)
    regression_per_class_acc = _per_class_accuracy(regression_result_cases)
    cm = _confusion_matrix(cases)
    synthetic_cm = _confusion_matrix(synthetic_cases)
    regression_cm = _confusion_matrix(regression_result_cases)

    counts_target = _counts_by_key(cases, "target_class")
    counts_pred = _counts_by_key(cases, "predicted_class")
    synthetic_counts_target = _counts_by_key(synthetic_cases, "target_class")
    synthetic_counts_pred = _counts_by_key(synthetic_cases, "predicted_class")
    regression_counts_target = _counts_by_key(regression_result_cases, "target_class")
    regression_counts_pred = _counts_by_key(regression_result_cases, "predicted_class")

    mismatches = _mismatches_from_cases(cases)
    synthetic_mismatches = _mismatches_from_cases(synthetic_cases)
    regression_mismatches = _mismatches_from_cases(regression_result_cases)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "mode": mode,
        "cdm_count": cdm_count,
        "regression_cases_path": str(final_regression_cases_path) if mode in ("regression", "all") else None,
        "messages_per_class": messages_per_class,
        "noise_level": noise_level,
        "seed": final_seed,
        "scenario_mode": scenario_mode,
        "scenario_count_per_class": scenario_count_per_class,
        "max_attempts_multiplier": max_attempts_multiplier,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "message_gen_model": final_gen_model,
        "message_gen_retries": MESSAGE_GEN_MAX_RETRIES,
        "token_usage_total": token_usage_total,
        "token_usage": token_usage,
        "summary": {
            "total_cases": len(cases),
            "accuracy": accuracy,
            "synthetic_accuracy": synthetic_accuracy,
            "regression_accuracy": regression_accuracy,
            "per_class_accuracy": per_class_acc,
            "synthetic_per_class_accuracy": synthetic_per_class_acc,
            "regression_per_class_accuracy": regression_per_class_acc,
            "counts_target": counts_target,
            "counts_predicted": counts_pred,
            "synthetic_counts_target": synthetic_counts_target,
            "synthetic_counts_predicted": synthetic_counts_pred,
            "regression_counts_target": regression_counts_target,
            "regression_counts_predicted": regression_counts_pred,
            "confusion_matrix": cm,
            "synthetic_confusion_matrix": synthetic_cm,
            "regression_confusion_matrix": regression_cm,
            "errors_count": len(errors),
            "mismatches_count": len(mismatches),
            "synthetic_mismatches_count": len(synthetic_mismatches),
            "regression_mismatches_count": len(regression_mismatches),
        },
        "cases": cases,
        "mismatches": mismatches,
        "synthetic_mismatches": synthetic_mismatches,
        "regression_mismatches": regression_mismatches,
        "errors": errors,
    }

    out_path = REPORTS_DIR / f"message_classifier_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding=TEXT_FILE_ENCODING)

    _log(
        quiet,
        "[summary] "
        f"total_cases={len(cases)} "
        f"accuracy={(f'{accuracy:.2f}%' if accuracy is not None else 'n/a')} "
        f"mismatches={len(mismatches)} "
        f"errors={len(errors)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate messages with known TARGET classes and evaluate message_classifier prompt accuracy."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "regression", "all"],
        help="Dataset mode: generate synthetic cases, run manual regression cases, or both.",
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
        default=None,
        help="Use first N CDM fixtures (sorted by filename). Default: all.",
    )
    parser.add_argument(
        "--messages-per-class",
        type=int,
        default=0,
        help="How many messages to generate for EACH class in synthetic mode.",
    )
    parser.add_argument(
        "--regression-cases",
        type=str,
        default=str(DEFAULT_REGRESSION_CASES_PATH),
        help=f"Path to manual regression cases JSON (default: {DEFAULT_REGRESSION_CASES_PATH}).",
    )
    parser.add_argument(
        "--noise-level",
        type=int,
        default=2,
        help="0..2. Higher means more noise/indirectness (kept bounded to avoid ambiguity).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides cfg seed if provided).",
    )
    parser.add_argument(
        "--message-gen-model",
        type=str,
        default=None,
        help=f"Message generation model override (default: {DEFAULT_MESSAGE_GEN_MODEL}).",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Override message_classifier prompt id (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override message_classifier prompt version (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="random",
        choices=["random", "cycle"],
        help="How to select scenario hints inside each class bucket.",
    )
    parser.add_argument(
        "--scenario-count-per-class",
        type=int,
        default=None,
        help="If set, restrict to first N scenario hints in each class pool.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=30,
        help="Attempts limit per class = messages_per_class * multiplier.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console progress output.",
    )

    args = parser.parse_args()

    run_message_classifier_dataset(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        messages_per_class=int(args.messages_per_class),
        mode=str(args.mode),
        regression_cases_path=pathlib.Path(args.regression_cases) if args.regression_cases else None,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        message_gen_model=args.message_gen_model,
        noise_level=int(args.noise_level),
        seed=args.seed,
        scenario_mode=str(args.scenario_mode),
        scenario_count_per_class=args.scenario_count_per_class,
        max_attempts_multiplier=int(args.max_attempts_multiplier),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()

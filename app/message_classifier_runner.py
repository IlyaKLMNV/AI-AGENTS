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
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
REPORTS_DIR = ROOT / "tests" / "reports" / "message_classifier"

DEFAULT_MESSAGE_GEN_MODEL = "gpt-4.1-mini"
MESSAGE_GEN_MAX_RETRIES = 1

CLASSES = ("reason_farewell", "no_reason", "acceptance", "human_needed")


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "Кандидат первым вопросом спрашивают сколько платят (в грубой форме).",
        "Кандидат задает вопросы по вакансии (обязанности/команда/стек).",

    ],
    "human_needed": [
        "Смешанное намерение: вроде интересно, но много сомнений и условий, не дает ясного 'да'.",
        "Раздражение/жалоба/негатив к рекрутеру или компании.",
        "Странные или нерелевантные вопросы (не про вакансию), либо непонятный смысл.",
        "Просьба денег/мошеннический оттенок/обвинения, без явного согласия или отказа.",
        "Сообщение не по теме или набор слов/эмодзи так, что смысл неясен.",
    ],
}


def _pick_scenario_hint(
    target_class: str,
    rng: random.Random,
    scenario_mode: str,
    scenario_count_per_class: Optional[int],
) -> str:
    pool = SCENARIO_HINTS_BY_CLASS.get(target_class) or ["Нейтральное сообщение."]
    if scenario_count_per_class is not None and scenario_count_per_class > 0 and scenario_count_per_class < len(pool):
        pool = pool[:scenario_count_per_class]

    if scenario_mode == "random":
        return rng.choice(pool)

    idx = rng.randrange(0, len(pool))
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
            "Ты генерируешь одно входящее сообщение кандидата на русском языке после первого касания рекрутера.\n"
            "Тебе будет задан TARGET_CLASS: reason_farewell / no_reason / acceptance / human_needed.\n"
            "Сгенерируй ОДНО сообщение кандидата так, чтобы оно ОДНОЗНАЧНО соответствовало TARGET_CLASS.\n\n"
            "Правила:\n"
            "- Верни только текст сообщения кандидата, без JSON, без кавычек, без markdown, без пояснений.\n"
            "- Не упоминай названия классов.\n"
            "- Сообщение должно быть естественным и реалистичным.\n"
            "- Избегай двусмысленности: message_classifier должен легко распознать класс.\n\n"
            "Критерии:\n"
            "- acceptance: явный интерес/согласие, можно с релевантными вопросами по вакансии (вилка, график, формат, обязанности, команда, условия).\n"
            "- no_reason: отказ без причины.\n"
            "- reason_farewell: отказ с причиной.\n"
            "- human_needed: сомнения/смешанное намерение/негатив/странные вопросы/неясный смысл.\n"
        )

    def _payload(self, cdm: Dict[str, Any], target_class: str, scenario_hint: str, noise_level: int) -> str:
        vacancy = cdm.get("vacancy") or {}
        candidate = cdm.get("candidate") or {}

        noise_desc = ["низкий", "средний", "высокий"][min(max(noise_level, 0), 2)]

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
            "3) Учитывай контекст вакансии (название, формат, город, вилка) чтобы сообщение выглядело правдоподобно.\n"
            "4) Для acceptance: добавь 1-2 релевантных вопроса по вакансии или предложи созвон.\n"
            "5) Для no_reason: отказ без объяснений.\n"
            "6) Для reason_farewell: отказ и краткая причина.\n"
            "7) Для human_needed: сделай сомнения/негатив/нерелевантность или неясный смысл.\n"
            "8) Верни только одно сообщение кандидата.\n"
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
            # Keep it strict: one message, no multi-line dialogues.
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

    def classify(self, message: str) -> str:
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=message.strip(),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
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


def _accuracy(cases: List[Dict[str, Any]]) -> float:
    if not cases:
        return 0.0
    ok = sum(1 for c in cases if c.get("target_class") == c.get("predicted_class"))
    return round(ok / len(cases) * 100.0, 2)


def _per_class_accuracy(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls in CLASSES:
        items = [c for c in cases if c.get("target_class") == cls]
        if not items:
            out[cls] = 0.0
            continue
        ok = sum(1 for c in items if c.get("target_class") == c.get("predicted_class"))
        out[cls] = round(ok / len(items) * 100.0, 2)
    return out


def run_message_classifier_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    messages_per_class: int,
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

    if messages_per_class <= 0:
        raise ValueError("--messages-per-class must be > 0")
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

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
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
    synth = CandidateMessageSynthesizer(model=final_gen_model, seed=final_seed)
    clf = MessageClassifierRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

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
                _accumulate_usage(token_usage_total, synth.last_usage)

                predicted = clf.classify(message)
                _accumulate_usage(token_usage_total, clf.last_usage)

            except Exception as e:
                raw_error = repr(e)

            if raw_error is not None:
                errors.append(
                    {
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
                "target_class": target,
                "predicted_class": predicted,
                "match": bool(predicted == target),
                "scenario_hint": scenario_hint,
                "cdm_file": str(cdm_path),
                "vacancy_title": v_title,
                "vacancy_company": v_company,
                "message": message,
                "raw_classifier_output": predicted,
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

    accuracy = _accuracy(cases)
    per_class_acc = _per_class_accuracy(cases)
    cm = _confusion_matrix(cases)

    counts_target = Counter(c.get("target_class") for c in cases)
    counts_pred = Counter(c.get("predicted_class") for c in cases)

    mismatches = [
        {
            "target_class": c["target_class"],
            "predicted_class": c["predicted_class"],
            "scenario_hint": c["scenario_hint"],
            "cdm_file": c["cdm_file"],
            "vacancy_title": c["vacancy_title"],
            "vacancy_company": c["vacancy_company"],
            "message": c["message"],
        }
        for c in cases
        if not c.get("match")
    ]

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_count": cdm_count,
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
        "summary": {
            "total_cases": len(cases),
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc,
            "counts_target": dict(counts_target),
            "counts_predicted": dict(counts_pred),
            "confusion_matrix": cm,
            "errors_count": len(errors),
            "mismatches_count": len(mismatches),
        },
        "cases": cases,
        "mismatches": mismatches,
        "errors": errors,
    }

    out_path = REPORTS_DIR / f"message_classifier_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"total_cases={len(cases)} "
        f"accuracy={accuracy:.2f}% "
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
        required=True,
        help="How many messages to generate for EACH class (reason_farewell, no_reason, acceptance, human_needed).",
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

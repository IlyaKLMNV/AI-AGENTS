from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pathlib
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

# Repo root: if this file is in app/, parents[1] is repo root.
ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"
DEFAULT_REGRESSION_CASES_PATH = ROOT / "tests" / "fixtures" / "verdict_classifier" / "regression_cases.json"
REPORTS_DIR = ROOT / "tests" / "reports" / "verdict_classifier"

DEFAULT_DIALOGUE_GEN_MODEL = "gpt-4.1-mini"
DIALOGUE_GEN_MAX_RETRIES = 1

VERDICTS = ("passed", "failed", "deadlock")


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


def load_regression_cases(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Regression cases file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Regression cases JSON must be a list")

    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Regression case #{idx} must be an object")

        case_id = str(item.get("id") or "").strip()
        target_verdict = str(item.get("target_verdict") or "").strip().lower()
        dialogue = str(item.get("dialogue") or "").strip()

        if not case_id:
            raise ValueError(f"Regression case #{idx} is missing required field 'id'")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate regression case id: {case_id}")
        if target_verdict not in VERDICTS:
            raise ValueError(
                f"Regression case '{case_id}' has invalid target_verdict={target_verdict!r}; "
                f"expected one of {VERDICTS}"
            )
        if not dialogue:
            raise ValueError(f"Regression case '{case_id}' is missing required field 'dialogue'")

        seen_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "target_verdict": target_verdict,
                "scenario": str(item.get("scenario") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "dialogue": dialogue,
            }
        )

    return cases


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


def _resolve_verdict_prompt_from_cfg(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    block = cfg.get("verdict_classifier") or {}
    pid = block.get("prompt_id")
    pver = block.get("prompt_version")
    seed = block.get("seed")
    return (str(pid) if pid else None, str(pver) if pver else None, int(seed) if seed is not None else None)


def _resolve_dialogue_gen_model_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    # If you want, you can add in model.yaml:
    # verdict_classifier:
    #   dialogue_gen_model: gpt-4.1-mini
    block = cfg.get("verdict_classifier") or {}
    m = block.get("dialogue_gen_model")
    return str(m) if m else None


def _extract_verdict(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    # allow extra whitespace/noise
    m = re.search(r"\b(passed|failed|deadlock)\b", t)
    return m.group(1) if m else None


# Scenario pools are now only "hints" to increase the chance of producing the target verdict.
SCENARIO_HINTS_BY_VERDICT: Dict[str, List[str]] = {
    "passed": [
        # Passed is "clean" flow: no special triggers, just normal screening until END.
        "Нормальный скрининг по алгоритму, кандидат отвечает по делу, рекрутер завершает END после получения всех ответов.",
    ],
    "failed": [
        # Failed = mismatch / KO / refusal by candidate (по сути вакансии) / toxic behavior.
        "KO по зарплате: кандидат называет ожидания выше бюджета из контекста, рекрутер вежливо отказывает и END.",
        "KO по локации для офис/гибрид: кандидат не в городе вакансии и не готов к переезду, рекрутер отказывает и END.",
        "Нет нужного опыта: кандидат прямо говорит, что опыта/стека нет, рекрутер отказывает и END.",
        "Кандидат отказывается от вакансии по сути: 'не интересно/не актуально/не рассматриваю', рекрутер завершает END.",
        "Агрессия/оскорбления/флирт/политика/просьба денег/тестовое - рекрутер завершает по скрипту и END (это failed по сути отказа).",
        "Кандидат уже трудоустроен/принял оффер - рекрутер завершает END (failed).",
        "Вопрос 'вы бот/ИИ?'",
    ],
    "deadlock": [
        # Deadlock = procedural break (legitimacy, wrong contact, can't proceed).
        "Не тот человек/ошибка контакта/уже общались/не писать: диалог обрывается процедурно, END.",
        "Неразборчиво два раза: рекрутер один раз уточняет.",
        "Иностранный язык так, что скрининг не идет: END.",        
        "Источник контакта/легитимность/корпоративная почта: кандидат требует подтверждений и не идет по скринингу, диалог заканчивается END.",
    ],
}


def _pick_scenario_hint(
    target_verdict: str,
    rng: random.Random,
    scenario_mode: str,
    scenario_count_per_verdict: Optional[int],
    cycle_state: Optional[Dict[str, int]] = None,
) -> str:
    pool = SCENARIO_HINTS_BY_VERDICT.get(target_verdict) or ["Нормальный диалог."]
    if scenario_count_per_verdict is not None and scenario_count_per_verdict > 0:
        # restrict to a subset of scenarios for this run
        if scenario_count_per_verdict < len(pool):
            pool = pool[:scenario_count_per_verdict]

    if scenario_mode == "random":
        return rng.choice(pool)

    if cycle_state is None:
        cycle_state = {}
    idx = int(cycle_state.get(target_verdict, 0))
    cycle_state[target_verdict] = idx + 1
    idx = idx % len(pool)
    return pool[idx]


class DialogueSynthesizer:
    """
    Generates a FULL dialogue text (Рекрутер:/Кандидат:) with a known TARGET verdict.
    This is the "ground truth label" used to test verdict_classifier prompt.

    Important: verdict_classifier should NOT be used to build the dataset.
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
        # Output is plain dialogue text - no JSON.
        return (
            "Ты генерируешь реалистичный диалог переписки между рекрутером и кандидатом.\n"
            "Формат:\n"
            "- Строго чередуй реплики: Рекрутер, Кандидат, Рекрутер, Кандидат, ...\n"
            "- Каждая реплика на новой строке.\n"
            "- Каждая строка начинается с 'Рекрутер:' или 'Кандидат:'.\n"
            "- Диалог должен быть на русском языке.\n"
            "- Диалог должен закончиться: последняя реплика рекрутера должна содержать 'END'.\n"
            "- Никаких markdown, никаких пояснений, только сам диалог.\n\n"
            "Требование к разметке истины:\n"
            "- Тебе будет задан TARGET_VERDICT: passed / failed / deadlock.\n"
            "- Сгенерируй диалог так, чтобы по смыслу он ОДНОЗНАЧНО соответствовал TARGET_VERDICT.\n"
            "\n"
            "Определения:\n"
            "- passed: скрининг успешно завершен, рекрутер собрал ответы на приоритетные вопросы и вопросы из [questions], затем корректно завершил END.\n"
            "- failed: кандидат не подходит по требованиям (KO: локация/зарплата/опыт) ИЛИ кандидат отказался от вакансии по сути, диалог завершен отказом и END.\n"
            "- deadlock: диалог сорвался процедурно (легитимность/источник контакта/не тот человек/не писать/неразборчиво повторно/иностранный язык и т.п.) и скрининг по сути не состоялся, END.\n"
        )

    def _payload(self, cdm: Dict[str, Any], target_verdict: str, scenario_hint: str, noise_level: int) -> str:
        vacancy = cdm.get("vacancy") or {}
        candidate = cdm.get("candidate") or {}

        # noise_level 0..2 impacts length/indirectness slightly
        noise_desc = ["низкий", "средний", "высокий"][min(max(noise_level, 0), 2)]

        ctx = {
            "TARGET_VERDICT": target_verdict,
            "SCENARIO_HINT": scenario_hint,
            "noise_level": noise_desc,
            "vacancy": {
                "title": vacancy.get("title"),
                "company_name": vacancy.get("company_name"),
                "firm_description": vacancy.get("company_description") or vacancy.get("firm_description"),
                "responsibilities": vacancy.get("responsibilities"),
                "work_format": vacancy.get("work_format"),
                "location": vacancy.get("location"),
                "salary_range_from": vacancy.get("salary_range_from"),
                "salary_range_to": vacancy.get("salary_range_to"),
                "salary": vacancy.get("salary"),
                "questions": vacancy.get("questions"),
            },
            "candidate": {
                "recruiter_name": candidate.get("recruiter_name"),
                "candidate_name": candidate.get("candidate_name"),
                "candidate_job_list": candidate.get("candidate_job_list"),
                "candidate_skills": candidate.get("candidate_skills"),
            },
        }

        # Make the generator explicitly commit to the target verdict.
        return (
            "CONTEXT_JSON:\n"
            f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
            "INSTRUCTIONS:\n"
            f"1) TARGET_VERDICT = {target_verdict}\n"
            f"2) SCENARIO_HINT = {scenario_hint}\n"
            f"3) Учитывай контекст вакансии и кандидата.\n"
            "4) Для passed: обязательно пройди по приоритетам (зарплата/город) и нескольким вопросам из questions, затем END.\n"
            "5) Для failed: сделай явный отказ по требованиям/KO или отказ кандидата по сути вакансии, затем END.\n"
            "6) Для deadlock: сделай процедурный тупик (легитимность/не тот человек/не писать/неразборчиво повторно/и т.п.), затем END.\n"
            "7) В итоге верни только диалог в нужном формате.\n"
        )

    def synthesize_one(self, cdm: Dict[str, Any], target_verdict: str, scenario_hint: str, noise_level: int) -> str:
        instruction = self._instruction()
        payload = self._payload(cdm=cdm, target_verdict=target_verdict, scenario_hint=scenario_hint, noise_level=noise_level)

        resp = self.client.responses.create(
            model=self.model,
            input=instruction + "\n\n" + payload,
        )
        self.last_usage = getattr(resp, "usage", None)
        text = (getattr(resp, "output_text", "") or "").strip()

        # minimal sanitation: ensure it contains speaker tags and END
        if "Рекрутер:" not in text or "Кандидат:" not in text or "END" not in text:
            raise ValueError("dialogue generator returned invalid dialogue (missing tags or END)")
        return text


class VerdictClassifierRunner:
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

    def classify(self, dialogue: str) -> str:
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=dialogue.strip(),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
        self.last_raw_output = raw
        verdict = _extract_verdict(raw)
        if verdict not in VERDICTS:
            raise ValueError(f"verdict_classifier returned invalid output: {raw!r}")
        return verdict


def _confusion_matrix(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    m: Dict[str, Dict[str, int]] = {t: {p: 0 for p in VERDICTS} for t in VERDICTS}
    for c in cases:
        t = c.get("target_verdict")
        p = c.get("predicted_verdict")
        if t in VERDICTS and p in VERDICTS:
            m[t][p] += 1
    return m


def _accuracy(cases: List[Dict[str, Any]]) -> float:
    if not cases:
        return 0.0
    ok = sum(1 for c in cases if c.get("target_verdict") == c.get("predicted_verdict"))
    return round(ok / len(cases) * 100.0, 2)


def _per_class_accuracy(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    by_t: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VERDICTS}
    for c in cases:
        t = c.get("target_verdict")
        if t in by_t:
            by_t[t].append(c)

    out: Dict[str, float] = {}
    for v, items in by_t.items():
        if not items:
            out[v] = 0.0
            continue
        ok = sum(1 for c in items if c.get("target_verdict") == c.get("predicted_verdict"))
        out[v] = round(ok / len(items) * 100.0, 2)
    return out


def _counts_by_key(cases: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(c.get(key)) for c in cases if c.get(key) in VERDICTS))


def _mismatches_from_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cases if not c.get("match")]


def run_verdict_classifier_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    dialogs_per_verdict: int,
    mode: str,
    regression_cases_path: Optional[pathlib.Path],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    dialogue_gen_model: Optional[str],
    noise_level: int,
    seed: Optional[int],
    scenario_mode: str,
    scenario_count_per_verdict: Optional[int],
    max_attempts_multiplier: int,
    quiet: bool,
) -> pathlib.Path:
    ensure_dirs()

    if mode not in ("synthetic", "regression", "all"):
        raise ValueError("--mode must be synthetic|regression|all")
    if mode in ("synthetic", "all") and dialogs_per_verdict <= 0:
        raise ValueError("--dialogs-per-verdict must be > 0 when mode includes synthetic")
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

    cfg_pid, cfg_pver, cfg_seed = _resolve_verdict_prompt_from_cfg(cfg)
    cfg_gen_model = _resolve_dialogue_gen_model_from_cfg(cfg)

    env_pid = os.environ.get("VERDICT_CLASSIFIER_PROMPT_ID")
    env_pver = os.environ.get("VERDICT_CLASSIFIER_PROMPT_VERSION")

    final_pid = prompt_id or cfg_pid or env_pid
    final_pver = prompt_version or cfg_pver or env_pver

    if not final_pid:
        raise EnvironmentError(
            "No prompt_id found. Provide --prompt-id, or set VERDICT_CLASSIFIER_PROMPT_ID, "
            "or add tests/tools/model.yaml -> verdict_classifier.prompt_id"
        )

    final_seed = seed if seed is not None else cfg_seed
    final_gen_model = dialogue_gen_model or cfg_gen_model or DEFAULT_DIALOGUE_GEN_MODEL

    cdm_paths: List[pathlib.Path] = []
    if mode in ("synthetic", "all"):
        cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)
        if not cdm_paths:
            raise FileNotFoundError("No CDM fixtures resolved")

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
        f"dialogs_per_verdict={dialogs_per_verdict} "
        f"noise_level={noise_level} "
        f"seed={final_seed} "
        f"scenario_mode={scenario_mode} "
        f"scenario_count_per_verdict={scenario_count_per_verdict} "
        f"max_attempts_multiplier={max_attempts_multiplier}",
    )
    _log(
        quiet,
        "[init] "
        f"prompt_id={final_pid} "
        f"prompt_version={final_pver} "
        f"dialogue_gen_model={final_gen_model} "
        f"dialogue_gen_retries={DIALOGUE_GEN_MAX_RETRIES}",
    )
    if mode in ("regression", "all"):
        _log(
            quiet,
            "[init] "
            f"regression_cases_path={final_regression_cases_path} "
            f"regression_cases={len(regression_cases)}",
        )

    rng = random.Random(final_seed)
    cycle_state: Dict[str, int] = {}
    synth = DialogueSynthesizer(model=final_gen_model, seed=final_seed) if mode in ("synthetic", "all") else None
    clf = VerdictClassifierRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()
    token_usage = {
        "dialogue_generator": _blank_usage(),
        "verdict_classifier": _blank_usage(),
    }

    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if mode in ("synthetic", "all"):
        assert synth is not None
        # sequential fill: passed -> failed -> deadlock
        for target in VERDICTS:
            need = dialogs_per_verdict
            attempts_limit = dialogs_per_verdict * max_attempts_multiplier

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
                    target_verdict=target,
                    rng=rng,
                    scenario_mode=scenario_mode,
                    scenario_count_per_verdict=scenario_count_per_verdict,
                    cycle_state=cycle_state,
                )

                _log(
                    quiet,
                    f"  [gen] target={target} cdm={cdm_path.name} title={v_title} company={v_company}",
                )
                _log(quiet, f"    [hint] {scenario_hint}")

                dialogue = ""
                predicted: Optional[str] = None
                raw_error: Optional[str] = None

                try:
                    dialogue = synth.synthesize_one(
                        cdm=cdm,
                        target_verdict=target,
                        scenario_hint=scenario_hint,
                        noise_level=noise_level,
                    )
                    _accumulate_usage(token_usage_total, synth.last_usage)
                    _accumulate_usage(token_usage["dialogue_generator"], synth.last_usage)

                    predicted = clf.classify(dialogue)
                    _accumulate_usage(token_usage_total, clf.last_usage)
                    _accumulate_usage(token_usage["verdict_classifier"], clf.last_usage)

                except Exception as e:
                    raw_error = repr(e)

                if raw_error is not None:
                    errors.append(
                        {
                            "case_type": "synthetic",
                            "target_verdict": target,
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
                    "target_verdict": target,
                    "predicted_verdict": predicted,
                    "match": bool(predicted == target),
                    "scenario_hint": scenario_hint,
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "dialogue": dialogue,
                    "raw_classifier_output": clf.last_raw_output,
                }
                cases.append(case)

                got += 1
                _log(quiet, f"    [ok] case={got}/{need} predicted={predicted} match={case['match']}")

            if got < need:
                raise RuntimeError(
                    f"Could not generate enough dialogues for target={target}: got {got}/{need} "
                    f"within {attempts_limit} attempts. Consider increasing --max-attempts-multiplier "
                    f"or adjusting scenario hints."
                )

    if mode in ("regression", "all"):
        _log(quiet, f"[regression] cases={len(regression_cases)}")
        for idx, regression_case in enumerate(regression_cases, start=1):
            predicted: Optional[str] = None
            raw_error: Optional[str] = None

            try:
                predicted = clf.classify(regression_case["dialogue"])
                _accumulate_usage(token_usage_total, clf.last_usage)
                _accumulate_usage(token_usage["verdict_classifier"], clf.last_usage)
            except Exception as e:
                raw_error = repr(e)

            if raw_error is not None:
                errors.append(
                    {
                        "case_type": "regression",
                        "id": regression_case["id"],
                        "target_verdict": regression_case["target_verdict"],
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
                "target_verdict": regression_case["target_verdict"],
                "predicted_verdict": predicted,
                "match": bool(predicted == regression_case["target_verdict"]),
                "dialogue": regression_case["dialogue"],
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

    overall_accuracy = _accuracy(cases)
    synthetic_accuracy = _accuracy(synthetic_cases)
    regression_accuracy = _accuracy(regression_result_cases)

    overall_per_class_acc = _per_class_accuracy(cases)
    synthetic_per_class_acc = _per_class_accuracy(synthetic_cases)
    regression_per_class_acc = _per_class_accuracy(regression_result_cases)

    overall_cm = _confusion_matrix(cases)
    synthetic_cm = _confusion_matrix(synthetic_cases)
    regression_cm = _confusion_matrix(regression_result_cases)

    overall_counts_target = _counts_by_key(cases, "target_verdict")
    overall_counts_pred = _counts_by_key(cases, "predicted_verdict")
    synthetic_counts_target = _counts_by_key(synthetic_cases, "target_verdict")
    synthetic_counts_pred = _counts_by_key(synthetic_cases, "predicted_verdict")
    regression_counts_target = _counts_by_key(regression_result_cases, "target_verdict")
    regression_counts_pred = _counts_by_key(regression_result_cases, "predicted_verdict")

    mismatches = _mismatches_from_cases(cases)
    synthetic_mismatches = _mismatches_from_cases(synthetic_cases)
    regression_mismatches = _mismatches_from_cases(regression_result_cases)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "mode": mode,
        "cdm_count": cdm_count,
        "regression_cases_path": str(final_regression_cases_path) if mode in ("regression", "all") else None,
        "dialogs_per_verdict": dialogs_per_verdict,
        "noise_level": noise_level,
        "seed": final_seed,
        "scenario_mode": scenario_mode,
        "scenario_count_per_verdict": scenario_count_per_verdict,
        "max_attempts_multiplier": max_attempts_multiplier,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "dialogue_gen_model": final_gen_model,
        "dialogue_gen_retries": DIALOGUE_GEN_MAX_RETRIES,
        "token_usage_total": token_usage_total,
        "token_usage": token_usage,
        "summary": {
            "total_cases": len(cases),
            "synthetic_cases_total": len(synthetic_cases),
            "regression_cases_total": len(regression_result_cases),
            "accuracy": overall_accuracy,
            "overall_accuracy": overall_accuracy,
            "synthetic_accuracy": synthetic_accuracy,
            "regression_accuracy": regression_accuracy,
            "per_class_accuracy": overall_per_class_acc,
            "overall_per_class_accuracy": overall_per_class_acc,
            "synthetic_per_class_accuracy": synthetic_per_class_acc,
            "regression_per_class_accuracy": regression_per_class_acc,
            "counts_target": overall_counts_target,
            "counts_predicted": overall_counts_pred,
            "synthetic_counts_target": synthetic_counts_target,
            "synthetic_counts_predicted": synthetic_counts_pred,
            "regression_counts_target": regression_counts_target,
            "regression_counts_predicted": regression_counts_pred,
            "confusion_matrix": overall_cm,
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

    out_path = REPORTS_DIR / f"verdict_classifier_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"total_cases={len(cases)} "
        f"overall_accuracy={overall_accuracy:.2f}% "
        f"synthetic_accuracy={synthetic_accuracy:.2f}% "
        f"regression_accuracy={regression_accuracy:.2f}% "
        f"mismatches={len(mismatches)} "
        f"errors={len(errors)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dialogues with known TARGET verdicts and evaluate verdict_classifier prompt accuracy."
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
        "--dialogs-per-verdict",
        type=int,
        default=0,
        help="How many dialogues to generate for EACH target verdict in synthetic mode.",
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
        help="0..2. Higher means more noise/indirectness.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides cfg seed if provided).",
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
        help="Override verdict_classifier prompt id (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override verdict_classifier prompt version (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="random",
        choices=["random", "cycle"],
        help="How to select scenario hints inside each verdict bucket.",
    )
    parser.add_argument(
        "--scenario-count-per-verdict",
        type=int,
        default=None,
        help="If set, restrict to first N scenario hints in each verdict pool.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=30,
        help="Attempts limit per verdict = dialogs_per_verdict * multiplier.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console progress output.",
    )

    args = parser.parse_args()

    run_verdict_classifier_dataset(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        dialogs_per_verdict=int(args.dialogs_per_verdict),
        mode=str(args.mode),
        regression_cases_path=pathlib.Path(args.regression_cases) if args.regression_cases else None,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        dialogue_gen_model=args.dialogue_gen_model,
        noise_level=int(args.noise_level),
        seed=args.seed,
        scenario_mode=str(args.scenario_mode),
        scenario_count_per_verdict=args.scenario_count_per_verdict,
        max_attempts_multiplier=int(args.max_attempts_multiplier),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()

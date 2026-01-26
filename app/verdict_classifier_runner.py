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
) -> str:
    pool = SCENARIO_HINTS_BY_VERDICT.get(target_verdict) or ["Нормальный диалог."]
    if scenario_count_per_verdict is not None and scenario_count_per_verdict > 0:
        # restrict to a subset of scenarios for this run
        if scenario_count_per_verdict < len(pool):
            pool = pool[:scenario_count_per_verdict]

    if scenario_mode == "random":
        return rng.choice(pool)

    # cycle: deterministic-ish rotation
    # We emulate cycling by using rng but on a running counter outside would be cleaner;
    # here we just use a stable selection based on rng state.
    idx = rng.randrange(0, len(pool))
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

    def classify(self, dialogue: str) -> str:
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=dialogue.strip(),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
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


def run_verdict_classifier_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    dialogs_per_verdict: int,
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

    if dialogs_per_verdict <= 0:
        raise ValueError("--dialogs-per-verdict must be > 0")
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

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)
    if not cdm_paths:
        raise FileNotFoundError("No CDM fixtures resolved")

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
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

    rng = random.Random(final_seed)
    synth = DialogueSynthesizer(model=final_gen_model, seed=final_seed)
    clf = VerdictClassifierRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()

    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # sequential fill: passed -> failed -> deadlock
    for target in VERDICTS:
        need = dialogs_per_verdict
        attempts_limit = dialogs_per_verdict * max_attempts_multiplier

        # don't show attempts as "progress"
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
            )

            # neutral "generation" log (no fake progress numbers)
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

                predicted = clf.classify(dialogue)
                _accumulate_usage(token_usage_total, clf.last_usage)

            except Exception as e:
                raw_error = repr(e)

            if raw_error is not None:
                errors.append(
                    {
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
                "target_verdict": target,               # ground truth label (what we intended to generate)
                "predicted_verdict": predicted,         # what verdict_classifier returned
                "match": bool(predicted == target),
                "scenario_hint": scenario_hint,
                "cdm_file": str(cdm_path),
                "vacancy_title": v_title,
                "vacancy_company": v_company,
                "dialogue": dialogue,
                "raw_classifier_output": predicted,
            }
            cases.append(case)

            got += 1
            # real progress: saved cases count
            _log(quiet, f"    [ok] case={got}/{need} predicted={predicted} match={case['match']}")

        if got < need:
            # hard fail: we didn't manage to generate enough cases for this target verdict
            raise RuntimeError(
                f"Could not generate enough dialogues for target={target}: got {got}/{need} "
                f"within {attempts_limit} attempts. Consider increasing --max-attempts-multiplier "
                f"or adjusting scenario hints."
            )

    # Metrics
    accuracy = _accuracy(cases)
    per_class_acc = _per_class_accuracy(cases)
    cm = _confusion_matrix(cases)

    counts_target = Counter(c.get("target_verdict") for c in cases)
    counts_pred = Counter(c.get("predicted_verdict") for c in cases)

    mismatches = [
        {
            "target_verdict": c["target_verdict"],
            "predicted_verdict": c["predicted_verdict"],
            "scenario_hint": c["scenario_hint"],
            "cdm_file": c["cdm_file"],
            "vacancy_title": c["vacancy_title"],
            "vacancy_company": c["vacancy_company"],
            "dialogue": c["dialogue"],
        }
        for c in cases
        if not c.get("match")
    ]

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_count": cdm_count,
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
        "summary": {
            "total_cases": len(cases),
            "accuracy": accuracy,  # aka pass_rate, but this is accuracy vs target labels
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

    out_path = REPORTS_DIR / f"verdict_classifier_report_{run_id}.json"
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
        description="Generate dialogues with known TARGET verdicts and evaluate verdict_classifier prompt accuracy."
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
        required=True,
        help="How many dialogues to generate for EACH target verdict (passed, failed, deadlock).",
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

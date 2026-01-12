from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pathlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "screening_autofill"

DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"

DEFAULT_VARIANTS_PER_CDM = 3
DEFAULT_CDM_COUNT = None

DEFAULT_DIALOGUE_GEN_MODEL = "gpt-4.1-mini"


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
        total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "token_count", None)
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


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

    required = ["preferred_location", "min_salary", "max_salary", "additional_info", "work_format"]
    for k in required:
        if k not in obj:
            errors.append(f"missing key: {k}")

    if "preferred_location" in obj and not isinstance(obj.get("preferred_location"), str):
        errors.append("preferred_location must be string")

    if "work_format" in obj:
        wf = obj.get("work_format")
        if not isinstance(wf, str):
            errors.append("work_format must be string")
        elif wf not in ("", "remote", "office", "hybrid"):
            errors.append("work_format must be one of: '', remote, office, hybrid")

    if "min_salary" in obj and not _only_digits_or_empty(obj.get("min_salary")):
        errors.append("min_salary must be digits-only string or empty string")

    if "max_salary" in obj and not _only_digits_or_empty(obj.get("max_salary")):
        errors.append("max_salary must be digits-only string or empty string")

    if "additional_info" in obj:
        ai = obj.get("additional_info")
        if not isinstance(ai, list):
            errors.append("additional_info must be list")
        else:
            for i, item in enumerate(ai):
                if not isinstance(item, dict):
                    errors.append(f"additional_info[{i}] must be object")
                    continue
                q = item.get("question")
                a = item.get("answer")
                if not isinstance(q, str) or not isinstance(a, str):
                    errors.append(f"additional_info[{i}] question/answer must be strings")

    return errors


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
        who = t.get("speaker", "").strip()
        msg = (t.get("text") or "").strip()
        if not who or not msg:
            continue
        if who.lower() == "recruiter":
            lines.append(f"Рекрутер: {msg}")
        else:
            lines.append(f"Кандидат: {msg}")
    return "\n".join(lines).strip()


def _flatten_like_prod(dialogue: str) -> str:
    s = (dialogue or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


class DialogueSynthesizer:
    def __init__(self, model: str, seed: Optional[int]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.last_usage: Any = None
        self.seed = seed

    def synthesize(
        self,
        cdm: Dict[str, Any],
        variants: int,
        noise_level: int,
        allow_two_questions: bool,
    ) -> List[str]:
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
            gen_cases.append(
                {
                    "variant_index": i + 1,
                    "answer_volume": rnd.choice(style_pool),
                    "noise": noise_pool[min(max(noise_level, 0), 2)],
                    "mix_two_questions": want_two_q and (rnd.random() < 0.55),
                    "answer_indirect": rnd.random() < (0.25 + 0.15 * min(max(noise_level, 0), 2)),
                    "include_extra_chitchat": rnd.random() < (0.20 + 0.20 * min(max(noise_level, 0), 2)),
                    "include_link_or_nda": rnd.random() < (0.10 + 0.20 * min(max(noise_level, 0), 2)),
                    "format_synonyms": True,
                    "salary_synonyms": True,
                }
            )

        instruction = (
            "Ты генерируешь реалистичный диалог рекрутера и кандидата для первичного скрининга.\n"
            "Важно: диалог нужен для тестирования авто-заполнения формы скрининга.\n\n"
            "Требования:\n"
            "1) Рекрутер задает вопросы строго из списка vacancy.questions (можно перефразировать), но не добавляй новые темы.\n"
            "2) В одном сообщении рекрутера может быть максимум 2 вопроса, если включен mix_two_questions.\n"
            "3) Кандидат отвечает по смыслу. Иногда отвечает не прямолинейно (answer_indirect), добавляет детали и шум.\n"
            "4) В ответах кандидата должны иногда встречаться:\n"
            "   - город/локация (preferred_location)\n"
            "   - ожидания по зарплате (min_salary/max_salary) в разных формах: 'от 170к', '170 000', 'до 200 тысяч', '100-150к'\n"
            "   - формат работы: удаленно/офис/гибрид (последнее упоминание считать предпочтением)\n"
            "5) Иногда кандидат может дать часть ответа сразу, а часть уточнить позже.\n"
            "6) Не используй Markdown. Не добавляй никаких служебных меток.\n\n"
            "Формат ответа строго JSON.\n"
            "Верни массив длиной N, где каждый элемент:\n"
            "{\n"
            '  "turns": [\n'
            '    {"speaker":"recruiter","text":"..."},\n'
            '    {"speaker":"candidate","text":"..."},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
        )

        payload = {
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

        resp = self.client.responses.create(
            model=self.model,
            input=instruction + "\n\n" + json.dumps(payload, ensure_ascii=False),
        )
        self.last_usage = getattr(resp, "usage", None)
        text = (getattr(resp, "output_text", "") or "").strip()

        data = _safe_json_loads(text)
        if not isinstance(data, list):
            raise ValueError("dialogue generator did not return a JSON array")

        out_dialogues: List[str] = []
        for item in data[:variants]:
            turns = item.get("turns") if isinstance(item, dict) else None
            if not isinstance(turns, list):
                continue
            dialogue = _format_dialogue(turns)
            if dialogue:
                out_dialogues.append(dialogue)

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
) -> pathlib.Path:
    ensure_dirs()

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    print(
        "[init] "
        f"run_id={run_id} "
        f"cdm_dir={cdm_dir} "
        f"cdm_count={cdm_count} "
        f"variants_per_cdm={variants_per_cdm} "
        f"noise_level={noise_level} "
        f"allow_two_questions={allow_two_questions} "
        f"flatten_like_prod={flatten_like_prod} "
        f"seed={seed}"
    )

    cfg: Dict[str, Any] = {}
    if CFG_PATH.is_file():
        cfg = load_yaml(CFG_PATH) or {}
        print(f"[init] loaded cfg: {CFG_PATH}")
    else:
        print(f"[init] cfg not found: {CFG_PATH} (ok, will use env/cli)")

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

    all_paths = [pathlib.Path(p) for p in sorted(glob.glob(str(cdm_dir / "cdm_*.json")))]
    print(f"[gen] fixtures_found={len(all_paths)}")

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)
    print(f"[gen] actual_cases={len(cdm_paths)}")

    print(
        "[init] "
        f"prompt_id={final_pid} "
        f"prompt_version={final_pver} "
        f"dialogue_gen_model={final_gen_model}"
    )

    synth = DialogueSynthesizer(model=final_gen_model, seed=seed)
    autofill = ScreeningAutofillPromptRunner(prompt_id=final_pid, prompt_version=final_pver)

    usage = {
        "dialogue_generator": _blank_usage(),
        "screening_autofill": _blank_usage(),
    }

    results: List[Dict[str, Any]] = []

    ok_count = 0
    fail_count = 0
    schema_fail_count = 0
    error_count = 0

    total_cases = len(cdm_paths)

    for case_idx, cdm_path in enumerate(cdm_paths, start=1):
        print(f"[run] case {case_idx}/{total_cases} ({cdm_path.name})")

        cdm = load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}

        try:
            dialogues = synth.synthesize(
                cdm=cdm,
                variants=variants_per_cdm,
                noise_level=noise_level,
                allow_two_questions=allow_two_questions,
            )
            _accumulate_usage(usage["dialogue_generator"], synth.last_usage)
        except Exception as e:
            err = repr(e)
            error_count += 1
            fail_count += 1
            print(f"[warn] dialogue synthesis failed: {cdm_path.name}: {err}")
            results.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": vacancy.get("title"),
                    "vacancy_company": vacancy.get("company_name"),
                    "variant_index": None,
                    "dialogue": "",
                    "raw_output": "",
                    "parsed_json": None,
                    "parse_ok": False,
                    "schema_errors": ["dialogue_synthesis_failed"],
                    "error": err,
                }
            )
            continue

        for v_idx, dialogue in enumerate(dialogues, start=1):
            print(f"  [variant {v_idx}/{variants_per_cdm}] running screening_autofill...")

            final_dialogue = _flatten_like_prod(dialogue) if flatten_like_prod else dialogue

            raw_out = ""
            parsed: Any = None
            parse_ok = False
            schema_errors: List[str] = []
            error: Optional[str] = None

            try:
                raw_out = autofill.run_once(final_dialogue)
                _accumulate_usage(usage["screening_autofill"], autofill.last_usage)

                parsed = _safe_json_loads(raw_out)
                schema_errors = _validate_schema(parsed)
                parse_ok = len(schema_errors) == 0
            except Exception as e:
                error = repr(e)

            if error is not None:
                error_count += 1
                fail_count += 1
                print(f"    [fail] error={error}")
            elif not parse_ok:
                fail_count += 1
                schema_fail_count += 1
                print(f"    [fail] schema_errors={schema_errors}")
            else:
                ok_count += 1
                print("    [ok] parse_ok=true")

            results.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": vacancy.get("title"),
                    "vacancy_company": vacancy.get("company_name"),
                    "variant_index": v_idx,
                    "dialogue": final_dialogue,
                    "raw_output": raw_out,
                    "parsed_json": parsed,
                    "parse_ok": parse_ok,
                    "schema_errors": schema_errors,
                    "error": error,
                }
            )

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_dir": str(cdm_dir),
        "cdm_count": cdm_count,
        "variants_per_cdm": variants_per_cdm,
        "noise_level": noise_level,
        "allow_two_questions": allow_two_questions,
        "flatten_like_prod": flatten_like_prod,
        "seed": seed,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "dialogue_gen_model": final_gen_model,
        "token_usage": usage,
        "results_total": len(results),
        "results": results,
    }

    out_path = REPORTS_DIR / f"screening_autofill_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    finished_at = datetime.datetime.now()
    duration_s = (finished_at - started_at).total_seconds()

    print(
        "[summary] "
        f"results_total={len(results)} "
        f"ok={ok_count} "
        f"failed={fail_count} "
        f"schema_failed={schema_fail_count} "
        f"errors={error_count} "
        f"duration_s={duration_s:.1f}"
    )
    print("[done] report saved:", out_path)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run screening_autofill_prompt using dialogues synthesized from CDM fixtures."
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

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()

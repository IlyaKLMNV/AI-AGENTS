# vacancy_keywords_runner.py
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
REPORTS_DIR = ROOT / "tests" / "reports" / "responsibilities_parser"

REPORT_VERBOSITY_VALUES = ("compact", "standard", "full")


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


def _resolve_prompt_from_cfg(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # Optional in tests/tools/model.yaml:
    # responsibilities_parser:
    #   prompt_id: pmpt_...
    #   prompt_version: ...
    #   seed: 1234
    block = cfg.get("responsibilities_parser") or {}
    pid = block.get("prompt_id")
    pver = block.get("prompt_version")
    seed = block.get("seed")
    return (str(pid) if pid else None, str(pver) if pver else None, int(seed) if seed is not None else None)


def _split_list_like(s: Optional[str]) -> List[str]:
    """
    Splits a comma/semicolon/newline-separated skills string into a clean list.
    Keeps original casing in returned values.
    """
    if not s:
        return []
    parts = re.split(r"[,\n;|]+", str(s))
    out: List[str] = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        t = re.sub(r"\s+", " ", t)
        out.append(t)
    return out


def _norm_key(s: str) -> str:
    """
    Normalization for matching. Makes CI/CD and cicd comparable, ignores punctuation.
    """
    t = (s or "").strip().lower()
    t = t.replace("ё", "е")
    # unify some common variants
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip()
    # remove all non-alnum (keep cyrillic/latin digits too, but digits handled separately)
    t = re.sub(r"[^a-z0-9а-я]+", "", t, flags=re.IGNORECASE)
    return t


def _contains_in_text(item: str, text: str) -> bool:
    """
    Checks if item appears in text (case-insensitive) with light normalization.
    """
    if not item or not text:
        return False
    it = item.strip().lower()
    tx = text.lower()
    if it in tx:
        return True

    # Try also normalized matching without punctuation/spaces
    it2 = _norm_key(item)
    tx2 = _norm_key(text)
    if it2 and it2 in tx2:
        return True
    return False


def _parse_json_array_strict(raw: str) -> List[str]:
    """
    Expects a JSON array of strings. Raises ValueError if invalid.
    """
    if not raw:
        raise ValueError("empty output")

    s = raw.strip()
    if not s.startswith("[") or not s.endswith("]"):
        raise ValueError(f"output is not a strict JSON array: {raw!r}")

    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"invalid JSON: {e!r}; raw={raw!r}") from e

    if not isinstance(obj, list):
        raise ValueError(f"output JSON is not a list: {raw!r}")

    out: List[str] = []
    for i, v in enumerate(obj):
        if not isinstance(v, str):
            raise ValueError(f"item[{i}] is not a string: {raw!r}")
        out.append(v.strip())
    return out


def _validate_item_format(item: str) -> Tuple[bool, List[str]]:
    """
    Enforces prompt format rules:
    - 1-3 words (spaces)
    - no digits
    - no commas/semicolons
    - looks like a term (no trailing punctuation)
    """
    errors: List[str] = []
    t = (item or "").strip()
    if not t:
        return False, ["empty item"]

    if re.search(r"\d", t):
        errors.append("contains digits")

    if "," in t or ";" in t:
        errors.append("contains comma/semicolon")

    # Word count by spaces (TTS/STT is one word, Function Calling is two)
    words = [w for w in re.split(r"\s+", t) if w]
    if not (1 <= len(words) <= 3):
        errors.append(f"word_count={len(words)} (expected 1..3)")

    # no full sentences
    if len(t) > 60:
        errors.append("too long (>60 chars)")

    # discourage verbs by simple heuristic (not strict)
    # (kept soft: we do not fail on it, only warn)
    # You can promote to a hard error if needed:
    # if re.search(r"\b(делать|уметь|понимать|знать|настраивать|разрабатывать)\b", t.lower()):
    #     errors.append("looks like a phrase with a verb")

    return len(errors) == 0, errors


def _match_to_expected(pred: str, expected: List[str]) -> Optional[str]:
    """
    Returns matched expected skill if found, else None.
    Matching is done via normalized key equality or substring containment.
    """
    if not pred:
        return None
    pk = _norm_key(pred)
    if not pk:
        return None

    expected_keys = {e: _norm_key(e) for e in expected}
    for e, ek in expected_keys.items():
        if pk == ek and ek:
            return e

    # fallback: sometimes pred is a shorter part of a longer expected item
    for e, ek in expected_keys.items():
        if pk and ek and (pk in ek or ek in pk):
            return e

    return None


class VacancyKeywordsRunner:
    def __init__(self, prompt_id: str, prompt_version: Optional[str]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def extract(self, vacancy_text: str) -> Tuple[List[str], str]:
        resp = self.client.responses.create(
            prompt=self.prompt,
            input=(vacancy_text or "").strip(),
        )
        self.last_usage = getattr(resp, "usage", None)
        raw = (getattr(resp, "output_text", "") or "").strip()
        items = _parse_json_array_strict(raw)
        return items, raw


def _get_raw_vacancy_text(cdm: Dict[str, Any], vacancy: Dict[str, Any]) -> str:
    raw_vacancy = vacancy.get("raw_vacancy") or cdm.get("raw_vacancy") or ""
    text = str(raw_vacancy).strip()
    if not text:
        raise ValueError("raw_vacancy is empty or missing")
    return text


def _find_duplicate_items(items: List[str]) -> List[str]:
    seen: set[str] = set()
    duplicates: List[str] = []
    for item in items:
        key = _norm_key(item)
        if not key:
            continue
        if key in seen:
            duplicates.append(item)
            continue
        seen.add(key)
    return duplicates


def _evaluate_case_contract(
    predicted: List[str],
    vacancy_text: str,
    expected_terms: List[str],
    min_total_matches: int,
    require_all_in_text: bool,
) -> Tuple[bool, List[str], Dict[str, Any], Dict[str, Any]]:
    list_errors: List[str] = []
    if not (1 <= len(predicted) <= 5):
        list_errors.append(f"len={len(predicted)} (expected 1..5)")

    format_errors: Dict[str, List[str]] = {}
    not_in_text: List[str] = []
    matched_expected: List[str] = []
    duplicates = _find_duplicate_items(predicted)

    for item in predicted:
        ok, errs = _validate_item_format(item)
        if not ok:
            format_errors[item] = errs

        if not _contains_in_text(item, vacancy_text):
            not_in_text.append(item)

        expected_match = _match_to_expected(item, expected_terms)
        if expected_match:
            matched_expected.append(expected_match)

    matched_expected_u = list(dict.fromkeys(matched_expected))

    issues: List[str] = []
    if list_errors:
        issues.append("list_constraints_failed")
    if format_errors:
        issues.append("item_format_failed")
    if duplicates:
        issues.append("duplicate_keywords")
    if require_all_in_text and not_in_text:
        issues.append("predicted_not_found_in_text")
    if len(matched_expected_u) < min_total_matches:
        issues.append(f"expected_matches<{min_total_matches}")

    checks = {
        "keywords_count": len(predicted),
        "matched_expected_count": len(matched_expected_u),
        "not_in_text_count": len(not_in_text),
        "duplicate_count": len(duplicates),
    }
    details = {
        "list_errors": list_errors,
        "format_errors": format_errors,
        "not_in_text": not_in_text,
        "duplicates": duplicates,
        "matched_expected": matched_expected_u,
        "expected_terms": expected_terms,
    }

    return len(issues) == 0, issues, checks, details


def _project_contract_case(case: Dict[str, Any], report_verbosity: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cdm_file": case["cdm_file"],
        "vacancy_title": case["vacancy_title"],
        "vacancy_company": case["vacancy_company"],
        "predicted_keywords": case["predicted_keywords"],
        "passed": case["passed"],
        "issues": case["issues"],
        "checks": case["checks"],
    }

    if report_verbosity in {"standard", "full"} and (case["issues"] or report_verbosity == "full"):
        out["details"] = {
            "not_in_text": case["details"]["not_in_text"],
            "duplicates": case["details"]["duplicates"],
            "matched_expected": case["details"]["matched_expected"],
            "list_errors": case["details"]["list_errors"],
            "format_errors": case["details"]["format_errors"],
        }

    if report_verbosity == "full":
        out["raw_extractor_output"] = case["raw_extractor_output"]
        out["raw_vacancy"] = case["raw_vacancy"]
        out["expected_terms"] = {
            "all": case["details"]["expected_terms"],
            "stack": case["details"].get("expected_stack", []),
            "skills": case["details"].get("expected_skills", []),
        }

    return out


def run_vacancy_keywords_dataset(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    cases_count: Optional[int],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    seed: Optional[int],
    min_total_matches: int,
    require_all_in_text: bool,
    report_verbosity: str,
    quiet: bool,
) -> pathlib.Path:
    ensure_dirs()

    if min_total_matches < 0:
        raise ValueError("--min-total-matches must be >= 0")
    if report_verbosity not in REPORT_VERBOSITY_VALUES:
        raise ValueError(f"--report-verbosity must be one of: {', '.join(REPORT_VERBOSITY_VALUES)}")

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    cfg: Dict[str, Any] = {}
    if CFG_PATH.is_file():
        cfg = load_yaml(CFG_PATH) or {}
        _log(quiet, f"[init] loaded cfg: {CFG_PATH}")
    else:
        _log(quiet, f"[init] cfg not found: {CFG_PATH} (ok, will use env/cli)")

    cfg_pid, cfg_pver, cfg_seed = _resolve_prompt_from_cfg(cfg)
    env_pid = os.environ.get("RESPONSIBILITIES_PARSER_PROMPT_ID")
    env_pver = os.environ.get("RESPONSIBILITIES_PARSER_PROMPT_VERSION")

    final_pid = prompt_id or cfg_pid or env_pid
    final_pver = prompt_version or cfg_pver or env_pver
    if not final_pid:
        raise EnvironmentError(
            "No prompt_id found. Provide --prompt-id, or set RESPONSIBILITIES_PARSER_PROMPT_ID, "
            "or add tests/tools/model.yaml -> responsibilities_parser.prompt_id"
        )

    final_seed = seed if seed is not None else cfg_seed
    rng = random.Random(final_seed)

    cdm_paths = load_cdm_files(cdm_dir, cdm_count=cdm_count)
    if cases_count is not None:
        if cases_count <= 0:
            raise ValueError("--cases-count must be > 0")
        # We'll sample from available CDMs
        sample_count = min(cases_count, len(cdm_paths))
        cdm_paths = rng.sample(cdm_paths, k=sample_count)

    _log(
        quiet,
        "[init] "
        f"run_id={run_id} "
        f"cdm_count={cdm_count} "
        f"cases_count_requested={cases_count} "
        f"cases_count_selected={len(cdm_paths)} "
        f"seed={final_seed} "
        f"min_total_matches={min_total_matches} "
        f"require_all_in_text={require_all_in_text} "
        f"report_verbosity={report_verbosity}",
    )
    _log(quiet, f"[init] prompt_id={final_pid} prompt_version={final_pver}")

    runner = VacancyKeywordsRunner(prompt_id=final_pid, prompt_version=final_pver)

    token_usage_total = _blank_usage()
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for cdm_path in cdm_paths:
        cdm = load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}
        v_title = vacancy.get("title")
        v_company = vacancy.get("company_name")

        expected_stack = _split_list_like(vacancy.get("vacancy_stack") or vacancy.get("stack") or "")
        expected_skills = _split_list_like(vacancy.get("vacancy_skills") or "")
        expected_terms = list(dict.fromkeys(expected_stack + expected_skills))

        raw_vacancy = ""
        raw_output = ""
        predicted: List[str] = []
        raw_error: Optional[str] = None

        try:
            raw_vacancy = _get_raw_vacancy_text(cdm=cdm, vacancy=vacancy)
            predicted, raw_output = runner.extract(raw_vacancy)
            _accumulate_usage(token_usage_total, runner.last_usage)

        except Exception as e:
            raw_error = repr(e)

        if raw_error is not None:
            errors.append(
                {
                    "cdm_file": str(cdm_path),
                    "vacancy_title": v_title,
                    "vacancy_company": v_company,
                    "error": raw_error,
                }
            )
            _log(quiet, f"[err] cdm={cdm_path.name} title={v_title} company={v_company} error={raw_error}")
            continue

        passed, issues, checks, details = _evaluate_case_contract(
            predicted=predicted,
            vacancy_text=raw_vacancy,
            expected_terms=expected_terms,
            min_total_matches=min_total_matches,
            require_all_in_text=require_all_in_text,
        )

        case = {
            "cdm_file": str(cdm_path),
            "vacancy_title": v_title,
            "vacancy_company": v_company,
            "raw_vacancy": raw_vacancy,
            "predicted_keywords": predicted,
            "raw_extractor_output": raw_output,
            "passed": passed,
            "issues": issues,
            "checks": checks,
            "details": details,
        }
        if report_verbosity == "full":
            case["details"]["expected_stack"] = expected_stack
            case["details"]["expected_skills"] = expected_skills
        cases.append(case)

        _log(
            quiet,
            "[case] "
            f"cdm={cdm_path.name} "
            f"title={v_title} "
            f"company={v_company} "
            f"passed={passed} "
            f"issues={issues} "
            f"matches={checks['matched_expected_count']} "
            f"not_in_text={checks['not_in_text_count']}",
        )

    total = len(cases)
    passed_n = sum(1 for c in cases if c.get("passed"))
    failed_n = total - passed_n
    pass_rate = round((passed_n / total * 100.0), 2) if total else 0.0
    avg_keywords = round(sum(int(c["checks"]["keywords_count"]) for c in cases) / total, 2) if total else 0.0
    avg_expected_matches = round(sum(int(c["checks"]["matched_expected_count"]) for c in cases) / total, 2) if total else 0.0
    issue_counter = Counter(issue for c in cases for issue in (c.get("issues") or []))

    cases_for_report = [_project_contract_case(case=c, report_verbosity=report_verbosity) for c in cases]

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "cdm_dir": str(cdm_dir),
        "cdm_count": cdm_count,
        "cases_count": len(cdm_paths),
        "cases_count_requested": cases_count,
        "seed": final_seed,
        "report_verbosity": report_verbosity,
        "prompt": {"prompt_id": final_pid, "prompt_version": final_pver},
        "thresholds": {
            "min_total_matches": min_total_matches,
            "require_all_in_text": require_all_in_text,
        },
        "token_usage_total": token_usage_total,
        "summary": {
            "total_cases": total,
            "passed": passed_n,
            "failed": failed_n,
            "pass_rate_pct": pass_rate,
            "avg_keywords_per_case": avg_keywords,
            "avg_expected_matches": avg_expected_matches,
            "errors_count": len(errors),
            "by_issue": dict(issue_counter),
        },
        "cases": cases_for_report,
        "errors": errors,
    }

    out_path = REPORTS_DIR / f"responsibilities_parser_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        quiet,
        "[summary] "
        f"total_cases={total} "
        f"passed={passed_n} "
        f"pass_rate={pass_rate:.2f}% "
        f"avg_expected_matches={avg_expected_matches:.2f} "
        f"errors={len(errors)} "
        f"issues={dict(issue_counter)} "
        f"tokens_total={token_usage_total.get('total_tokens', 0)}",
    )
    _log(quiet, "[done] report saved: " + str(out_path))

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run responsibilities_parser over raw_vacancy fields from CDM fixtures and report prompt-contract checks."
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
        "--cases-count",
        type=int,
        default=None,
        help="If set, sample N CDM fixtures from the available set (after --cdm-count). Default: use all selected.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides cfg seed if provided).",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Override responsibilities_parser prompt id (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override responsibilities_parser prompt version (otherwise from cfg/env).",
    )
    parser.add_argument(
        "--min-total-matches",
        type=int,
        default=2,
        help="Strict threshold only: minimum matched keywords vs (vacancy_stack U vacancy_skills).",
    )
    parser.add_argument(
        "--no-require-all-in-text",
        action="store_true",
        help="Disable the check that every predicted keyword must appear in vacancy text.",
    )
    parser.add_argument(
        "--report-verbosity",
        type=str,
        choices=list(REPORT_VERBOSITY_VALUES),
        default="compact",
        help=(
            "Report detail level: compact (smallest), standard (adds raw extractor output and matches), "
            "full (adds vacancy text and expected terms)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console progress output.",
    )

    args = parser.parse_args()

    run_vacancy_keywords_dataset(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        cases_count=args.cases_count,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        seed=args.seed,
        min_total_matches=int(args.min_total_matches),
        require_all_in_text=not bool(args.no_require_all_in_text),
        report_verbosity=str(args.report_verbosity),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()

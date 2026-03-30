from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

try:
    from app.extractor_agent_runner import PromptCfg, call_openai_step1, load_yaml, safe_json_loads, validate_step1_contract
except Exception:
    from extractor_agent_runner import PromptCfg, call_openai_step1, load_yaml, safe_json_loads, validate_step1_contract  # type: ignore


ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
DEFAULT_CDM_DIR = ROOT / "tests" / "fixtures" / "cdm"


def _load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_cdm_paths(cdm_dir: pathlib.Path, cdm_count: Optional[int]) -> List[pathlib.Path]:
    paths = [pathlib.Path(p) for p in sorted(glob.glob(str(cdm_dir / "cdm_*.json")))]
    if cdm_count is not None:
        paths = paths[: max(0, int(cdm_count))]
    return paths


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_prompt_cfg(
    cfg: Dict[str, Any],
    prompt_id_override: Optional[str],
    prompt_version_override: Optional[str],
    model_override: Optional[str],
) -> PromptCfg:
    block = cfg.get("extractor_agent") if isinstance(cfg.get("extractor_agent"), dict) else {}
    prompt_id = prompt_id_override or block.get("prompt_id") or os.getenv("EXTRACTOR_AGENT_PROMPT_ID")
    prompt_version = prompt_version_override or block.get("prompt_version") or os.getenv("EXTRACTOR_AGENT_PROMPT_VERSION")
    model = model_override or cfg.get("model") or ((cfg.get("openai") or {}).get("model")) or "gpt-4.1-mini"
    return PromptCfg(
        prompt_id=str(prompt_id) if prompt_id else None,
        prompt_version=str(prompt_version) if prompt_version else None,
        model=str(model),
    )


def _write_cdm(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def enrich_cdms(
    cdm_dir: pathlib.Path,
    cdm_count: Optional[int],
    prompt_id: Optional[str],
    prompt_version: Optional[str],
    model: Optional[str],
    timeout_s: int,
    quiet: bool,
) -> Tuple[int, int]:
    _load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set and was not found in .env")

    cfg = load_yaml(CFG_PATH) if CFG_PATH.exists() else {}
    prompt_cfg = _resolve_prompt_cfg(cfg or {}, prompt_id, prompt_version, model)
    if not prompt_cfg.prompt_id:
        raise EnvironmentError("extractor_agent.prompt_id is missing")

    cdm_paths = _load_cdm_paths(cdm_dir, cdm_count)
    if not cdm_paths:
        raise FileNotFoundError(f"No cdm_*.json found in {cdm_dir}")

    updated = 0
    failed = 0

    for cdm_path in cdm_paths:
        cdm = _load_json(cdm_path)
        vacancy = cdm.get("vacancy") or {}
        title = str(vacancy.get("title") or "").strip()
        if not title:
            failed += 1
            if not quiet:
                print(f"[skip] {cdm_path.name}: empty vacancy.title")
            continue

        text, _usage, err = call_openai_step1(
            api_key=api_key,
            prompt_cfg=prompt_cfg,
            user_input=title,
            timeout_s=timeout_s,
        )
        if err or not text:
            failed += 1
            if not quiet:
                print(f"[fail] {cdm_path.name}: extractor call failed: {err}")
            continue

        parsed, parse_err = safe_json_loads(text)
        if parse_err:
            failed += 1
            if not quiet:
                print(f"[fail] {cdm_path.name}: invalid JSON: {parse_err}")
            continue

        ok, errors, _warnings = validate_step1_contract(parsed, title)
        if not ok or not isinstance(parsed, dict):
            failed += 1
            if not quiet:
                print(f"[fail] {cdm_path.name}: invalid extractor contract: {errors}")
            continue

        vacancy["extractor_entities"] = parsed
        cdm["vacancy"] = vacancy
        _write_cdm(cdm_path, cdm)
        updated += 1
        if not quiet:
            print(f"[ok] {cdm_path.name}: title={title}")

    return updated, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill vacancy.extractor_entities in CDM files using extractor_agent prompt.")
    parser.add_argument("--cdm-dir", type=str, default=str(DEFAULT_CDM_DIR))
    parser.add_argument("--cdm-count", type=int, default=None)
    parser.add_argument("--prompt-id", type=str, default=None)
    parser.add_argument("--prompt-version", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started_at = datetime.datetime.now().isoformat()
    updated, failed = enrich_cdms(
        cdm_dir=pathlib.Path(args.cdm_dir),
        cdm_count=args.cdm_count,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        model=args.model,
        timeout_s=int(args.timeout_s),
        quiet=bool(args.quiet),
    )
    if not args.quiet:
        print(f"[done] started_at={started_at} updated={updated} failed={failed}")


if __name__ == "__main__":
    main()

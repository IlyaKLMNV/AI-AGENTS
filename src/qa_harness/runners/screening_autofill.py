"""Раннер screening_autofill: диалог → форма скрининга, оценка контракт + golden-ожидания + анти-утечка.

Что тестируем — промпт `screening_autofill`: на вход диалог рекрутер/кандидат (расплющенный в одну строку,
как в проде), на выход JSON-форма {preferred_location, min_salary, max_salary, work_format, additional_info}.
Бэкенда НЕТ (только LLM). Кейс:
- schema (форма): ключи/типы/enum work_format/digits зарплат/форма additional_info;
- expect (golden): ожидаемые поля совпали (work_format точным значением; зарплата/локация — `<nonempty>`);
- no_leak: в additional_info нет запрещённых тем (salary/location/work_format) и меток спикера;
- additional_info_nonempty (если кейс требует): доп. вопрос был — список не пуст.

Итог (passed) = schema & expect & no_leak & (ai_nonempty?). quality ≠ infra: сетевой сбой → errors;
невалидный JSON-вывод → schema-фейл (качество).

Режимы: golden (дефолт, диалоги из golden.yaml) · `--generate` (LLM генерит диалог с ИЗВЕСТНЫМ work_format,
кандидат явно называет формат/город/зарплату → expect строится детерминированно; work_format'ы × `--variants`)
· `--offline` (replay offline_output).

  python -m qa_harness.runners.screening_autofill --offline     # replay, без сети
  python -m qa_harness.runners.screening_autofill               # онлайн golden (OPENAI_API_KEY)
  python -m qa_harness.runners.screening_autofill --generate --variants 2 --temperature 0.9
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qa_harness.core import (
    accumulate_usage,
    add_prompt_source_args,
    blank_usage,
    load_cfg,
    make_prompt_client,
    prompt_under_test_meta,
    resolve_prompt,
    resolve_source,
    run_cases,
    usage_total,
)
from qa_harness.core.reporting import CaseRecord, ReportBuilder, write_reports
from qa_harness.domain.generators import (
    AutofillDialogueGenerator,
    AutofillSpec,
    GenerationPolicy,
    WORK_FORMATS,
    generate_valid,
)
from qa_harness.domain.screening_autofill import (
    GoldenCase,
    additional_info_leaks,
    check_expect,
    load_golden,
    parse_form,
    validate_schema,
)
from qa_harness.domain.screening_autofill.semantic import NONEMPTY

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "screening_autofill" / "golden.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_autofill"
AUTOFILL_INSTRUCTION = "Fill the screening form based on the dialogue below."
DEFAULT_GEN_MODEL = "gpt-4.1-mini"
# Контекст вакансии/кандидата для генерации диалогов (--generate); golden-кейсы его не используют.
DEFAULT_CDM: Dict[str, Any] = {
    "vacancy": {"title": "Python Backend Developer", "company_name": "ExampleSoft",
                "responsibilities": "Поддержка и развитие микросервисов, интеграции с продуктами."},
    "candidate": {"recruiter_name": "Анна", "candidate_name": "Кандидат"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="screening_autofill QA runner (schema + golden expect + no-leak).")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Курируемые golden-диалоги с ожиданиями.")
    p.add_argument("--offline", action="store_true", help="Replay offline_output из golden (без сети).")
    # --- режим вариативной генерации (батч-диалог с известным work_format) ---
    p.add_argument("--generate", action="store_true",
                   help="Вариативный режим: диалоги генерит LLM с известным work_format (а не из golden).")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help=f"Модель генератора диалога (по умолч. {DEFAULT_GEN_MODEL}).")
    p.add_argument("--gen-seed", type=int, default=None, help="Seed (резерв; вариативность даёт temperature).")
    p.add_argument("--variants", type=int, default=2, help="Диалогов на каждый work_format (--generate).")
    p.add_argument("--temperature", type=float, default=None, help="Temperature генератора диалога (--generate).")
    p.add_argument("--gen-retries", type=int, default=2, help="Повторов генерации диалога при провале валидации.")
    p.add_argument("--formats", default=None, help="work_format'ы через запятую (по умолч. hybrid,remote,office).")
    p.add_argument("--no-flatten", action="store_true", help="НЕ расплющивать диалог в одну строку (по умолчанию плющим, как прод).")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt-version", default=None)
    add_prompt_source_args(p)
    p.add_argument("--workers", type=int, default=6, help="Параллельных воркеров (I/O-bound LLM-вызовы).")
    p.add_argument("--step1-timeout", type=int, default=60, help="Таймаут вызова промпта, сек.")
    p.add_argument("--checkpoint-every", type=int, default=20, help="Перезапись отчёта каждые N кейсов (0=только в конце).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _flatten(dialogue: str) -> str:
    return re.sub(r"\s+", " ", (dialogue or "").replace("\r\n", "\n").replace("\r", "\n")).strip()


def _run_autofill(dialogue: str, client: Any, flatten: bool, res: Dict[str, Any]) -> None:
    """Прогнать промпт autofill по диалогу и положить raw/usage/parsed (или call_error/parse_error) в res."""
    d = _flatten(dialogue) if flatten else dialogue
    try:
        raw, usage = client.run(f"{AUTOFILL_INSTRUCTION}\n\n{d}")
        res["raw"], res["usage"] = raw, usage
    except Exception as e:  # noqa: BLE001 — сетевой/HTTP сбой = инфра
        res["call_error"] = f"autofill:{type(e).__name__}:{e}"
        return
    try:
        res["parsed"] = parse_form(raw)
    except ValueError as e:
        res["parse_error"] = f"invalid_json_output:{e}"


def _process(case: GoldenCase, client: Any, offline: bool, flatten: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"case": case, "parsed": None, "raw": None,
                           "call_error": None, "parse_error": None, "usage": None}
    if offline:
        if case.offline_output is None:
            res["call_error"] = "no_offline_output"
        else:
            res["parsed"] = case.offline_output
        return res
    _run_autofill(case.dialogue, client, flatten, res)
    return res


def _process_generate(item: Any, *, client: Any, gen_client: Any, gen_model: str,
                      gen_policy: GenerationPolicy, flatten: bool) -> Dict[str, Any]:
    """Один (work_format, вариант): генерим диалог с известным форматом → прогоняем промпт autofill."""
    work_format, fmt_index, variant = item
    res: Dict[str, Any] = {"case": None, "parsed": None, "raw": None, "call_error": None,
                           "parse_error": None, "usage": None, "mode": "generate",
                           "work_format": work_format, "variant": variant, "gen_source": None, "gen_usage": None}
    gen = AutofillDialogueGenerator(gen_client)

    def produce(_attempt):
        return gen.generate(AutofillSpec(DEFAULT_CDM, work_format, noise_level=variant % 3)), None

    gr = generate_valid(produce, policy=gen_policy)  # валидация — внутри parse (бросает на невалидном)
    res["gen_source"] = gr.source
    res["gen_usage"] = dict(gen.usage)
    if not gr.ok:
        res["call_error"] = f"dialogue_gen_failed:{'; '.join(gr.errors[-2:]) or 'unknown'}"
        return res
    dialogue = str(gr.item)
    expect = {"work_format": work_format, "preferred_location": NONEMPTY, "min_salary": NONEMPTY}
    res["case"] = GoldenCase(
        name=f"{work_format}", dialogue=dialogue, expect=expect,
        forbid_in_additional_info=["salary", "location", "work_format"],
        expect_additional_info_nonempty=False,
    )
    _run_autofill(dialogue, client, flatten, res)
    return res


def run(args: argparse.Namespace) -> Dict[str, Path]:
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")

    cfg = load_cfg(args.cfg)
    prompt = resolve_prompt(cfg, RUNNER, cli_id=args.prompt_id, cli_version=args.prompt_version)
    source = resolve_source(args.prompt_source)

    if args.generate and args.offline:
        raise ValueError("--generate несовместим с --offline (генерация требует сети).")

    flatten = not args.no_flatten
    client = None
    if not args.offline:
        from qa_harness.core.llm_client import get_client

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (промпт requires it)")
        client = make_prompt_client(prompt, source=source, local_version=args.local_prompt_version,
                                    prompts_path=args.prompts_path, client=get_client(timeout=args.step1_timeout))

    gen_setup: Dict[str, Any] = {}
    if args.generate:
        from qa_harness.core.llm_client import ModelClient

        formats = [f.strip() for f in (args.formats or ",".join(WORK_FORMATS)).split(",") if f.strip()]
        bad = [f for f in formats if f not in WORK_FORMATS]
        if bad:
            raise ValueError(f"неизвестные work_format: {bad}; допустимы {list(WORK_FORMATS)}")
        gen_setup = dict(
            gen_client=ModelClient(args.gen_model, timeout=args.step1_timeout, temperature=args.temperature),
            gen_model=args.gen_model,
            gen_policy=GenerationPolicy(max_retries=args.gen_retries, temperature=args.temperature),
            flatten=flatten,
        )
        work_items: List[Any] = [(fmt, fi, v) for fi, fmt in enumerate(formats)
                                 for v in range(max(1, args.variants))]
        models = {"dialogue_generator": args.gen_model, "evaluator": None}
        run_args = {"mode": "generate", "formats": formats, "variants": args.variants,
                    "gen_model": args.gen_model, "temperature": args.temperature,
                    "workers": args.workers, "flatten": flatten}
    else:
        work_items = list(load_golden(args.golden))
        models = {"generator": None, "evaluator": None}
        run_args = {"mode": "golden", "offline": bool(args.offline), "golden": len(work_items),
                    "workers": args.workers, "flatten": flatten}

    rb = ReportBuilder(
        runner=RUNNER,
        prompt_under_test=prompt_under_test_meta(prompt, source, args.local_prompt_version),
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        models=models,
        seed=prompt.seed,
        args=run_args,
    )

    usage_bucket = blank_usage()
    gen_usage_bucket = blank_usage()
    m, reasons, gen_sources = Counter(), Counter(), Counter()

    def _flush(interrupted: bool = False):
        rb.set_token_usage(usage_total(usage_bucket))
        extra: Dict[str, Any] = {"autofill": dict(m), "reasons": dict(reasons)}
        if args.generate:
            extra["generation"] = {"usage": usage_total(gen_usage_bucket), "sources": dict(gen_sources)}
        if interrupted:
            extra["interrupted"] = True
        finished = datetime.datetime.now()
        md, cd = rb.finalize(extra, finished_at=finished.isoformat(timespec="seconds"),
                             duration_s=round((finished - started).total_seconds(), 3))
        return write_reports(args.out_dir, RUNNER, run_id, md, cd)

    def _fold(res: Dict[str, Any]) -> None:
        is_gen = res.get("mode") == "generate"
        accumulate_usage(usage_bucket, res["usage"])
        accumulate_usage(usage_bucket, res.get("gen_usage"))
        accumulate_usage(gen_usage_bucket, res.get("gen_usage"))
        if res.get("gen_source"):
            gen_sources[res["gen_source"]] += 1
        if is_gen:
            label = f"{res['work_format']}/v{res['variant']}"
            cid = f"autofill:{res['work_format']}:v{res['variant']}"
        else:
            label = res["case"].name
            cid = f"golden:{res['case'].name}:v1"

        if res["call_error"]:
            rb.add_error(cid, res["call_error"])
            reasons[res["call_error"].split(":")[0]] += 1
            m["call_error"] += 1
            if not args.quiet:
                print(f"  [ERR ] {label}: {res['call_error']}")
            return
        case: GoldenCase = res["case"]

        m["total"] += 1
        rc: List[str] = []
        if res["parse_error"]:
            parsed = {}
            schema_ok, schema_errors = False, ["invalid_json_output"]
            expect_ok, expect_diffs = False, ["no_output"]
            leaks, ai_ok = [], False
            reasons["invalid_json_output"] += 1
            rc = ["invalid_json_output"]
        else:
            parsed = res["parsed"]
            schema_errors = validate_schema(parsed)
            schema_ok = not schema_errors
            expect_ok, expect_diffs = check_expect(parsed, case.expect)
            leaks = additional_info_leaks(parsed, case.forbid_in_additional_info)
            ai = parsed.get("additional_info") if isinstance(parsed, dict) else None
            ai_ok = (isinstance(ai, list) and len(ai) > 0) if case.expect_additional_info_nonempty else True
            if schema_ok:
                m["schema_pass"] += 1
            else:
                reasons["schema_fail"] += 1
                rc += [f"schema:{e}" for e in schema_errors[:6]]
            if expect_ok:
                m["expect_pass"] += 1
            else:
                reasons["expect_fail"] += 1
                rc += [f"expect:{d}" for d in expect_diffs[:6]]
            if not leaks:
                m["no_leak_pass"] += 1
            else:
                reasons["leak"] += 1
                m["leak_total"] += len(leaks)
                rc += [f"leak:{x}" for x in leaks[:6]]
            if not ai_ok:
                reasons["additional_info_empty"] += 1
                rc.append("additional_info_empty")

        quality_passed = bool(schema_ok) and bool(expect_ok) and (not leaks) and bool(ai_ok)
        checks = [
            {"rule": "schema", "passed": bool(schema_ok), "detail": ",".join(schema_errors[:6])},
            {"rule": "expect", "passed": bool(expect_ok), "detail": ",".join(expect_diffs[:6])},
            {"rule": "no_leak", "passed": not leaks, "detail": ",".join(leaks[:6])},
        ]
        if case.expect_additional_info_nonempty:
            checks.append({"rule": "additional_info_nonempty", "passed": bool(ai_ok), "detail": ""})

        inputs: Dict[str, Any] = {"criterion": "schema(форма) + expect(golden) + no_leak(additional_info)",
                                  "dialogue": case.dialogue, "expect": case.expect,
                                  "forbid_in_additional_info": case.forbid_in_additional_info}
        if is_gen:
            inputs["work_format_target"] = res["work_format"]
            inputs["variant"] = res["variant"]
            inputs["gen_source"] = res.get("gen_source")
        rb.add_case(CaseRecord(
            case_id=cid, source="synthetic" if is_gen else "golden", passed=quality_passed,
            inputs=inputs,
            output={"raw": res["raw"], "parsed": parsed},
            verdict={"evaluator": "screening_autofill_quality", "passed": quality_passed, "reason_codes": rc},
            checks=checks,
        ))
        if not args.quiet:
            wf = parsed.get("work_format") if isinstance(parsed, dict) else None
            print(f"  [{'ok ' if quality_passed else 'MISS'}] {label} "
                  f"schema={int(bool(schema_ok))} expect={expect_ok} leaks={len(leaks)} wf={wf!r}")

    total = len(work_items)

    if args.generate:
        def _work(it):
            return _process_generate(it, client=client, **gen_setup)
    else:
        def _work(c):
            return _process(c, client, args.offline, flatten)

    def _on_interrupt() -> None:
        if not args.quiet:
            print("\n[interrupted] сохраняю частичный отчёт...")

    outcome = run_cases(
        work_items,
        work=_work,
        fold=_fold,
        max_workers=max(1, args.workers),
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_flush,
        on_interrupt=_on_interrupt,
    )

    metrics_path, cases_path = _flush(interrupted=outcome.interrupted)
    if not args.quiet:
        import json as _json
        s = _json.loads(Path(metrics_path).read_text(encoding="utf-8"))["summary"]
        tag = "partial" if outcome.interrupted else "summary"
        mode = "offline" if args.offline else "online"
        print(f"[{tag}] {mode} quality_cases={s['total']} passed={s['passed']} failed={s['failed']} "
              f"errors(infra)={s['errors']} done={outcome.done}/{total}")
        print(f"[done] metrics -> {metrics_path}")
        print(f"[done] cases   -> {cases_path}")
    return {"metrics": metrics_path, "cases": cases_path}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

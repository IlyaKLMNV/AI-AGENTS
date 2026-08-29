"""Переигрывание записанных трасс через новое ядро политики. Ноль вызовов LLM, ноль токенов.

    python -m qa_harness.runners.policy_replay
    python -m qa_harness.runners.policy_replay --reports "tests/reports_v2/screening_split/*2026082*.cases.json"

Зачем: перестройка движка (docs/screening_split/rearchitecture.html) выкатывается ЖЁСТКОЙ ПОДМЕНОЙ,
без по-диалогового переключения и, значит, без отката (решение Р7). Сверка ядра на уже записанных
ходах — единственная страховка, которую можно получить до релиза, и она бесплатна.

ЧТО ЭТО ПРОВЕРЯЕТ: арифметику и порядок КОДА — зарплатный блок, счётчики, пороги, лимиты переспросов,
монотонность состояния, определение завершения.
ЧТО НЕ ПРОВЕРЯЕТ: качество наблюдения и правила R3a/R3b — старый `Decision` отдаёт ровно один
выбранный сигнал, а не все увиденные, поэтому восстановить вход для них нечем (см. `policy.adapter`).

Расхождения делятся на ЗАМЫСЕЛ и НАСТОЯЩИЕ. Замысел — там, где новое ядро обязано вести себя иначе:
фокус вопроса теперь выбирает код, а не модель. Настоящее расхождение — это когда ход завершился/не
завершился не так, ушёл не тот скрипт или разъехалось состояние; каждое такое требует разбора.
"""

import argparse
import glob
import json
import os
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from qa_harness.domain.screening_split import state as state_model
from qa_harness.domain.screening_split.policy import DecideContext, decide
from qa_harness.domain.screening_split.policy.guards import GuardSpec, apply_guards
from qa_harness.domain.screening_split.policy.adapter import (
    decision_to_observation,
    expected_outcome,
    is_replayable,
)

DEFAULT_REPORTS = "tests/reports_v2/screening_split/*.cases.json"
DEFAULT_BAND_MIN = 200000
DEFAULT_BAND_MAX = 280000

# Категории сравнения хода.
MATCH = "совпало"
FOCUS_SHIFT = "фокус выбрал код"        # замысел: приоритет теперь ведёт код, а не самоотчёт модели
ANSWER_ONLY = "модель не спрашивала"    # замысел: старый asking=null, код задаёт вопрос фокуса
RETIRED_KEY = "ключ снят в новой схеме"  # замысел: STOP_ABROAD и прочие рудименты реестра
# Не «замысел» и не баг: ход, где модель ТЯНУЛА диалог при полностью закрытой повестке. Её же промпт
# это прямо запрещает (`system.md:346`: «ЗАПРЕЩЕНО возвращать ask с подводкой вместо финиша»), но
# адхеренс не стопроцентный. Код финиширует по состоянию, поэтому расходится — в сторону правила.
# Вынесено отдельной строкой, а не спрятано в «совпало»: это изменение наблюдаемого поведения.
STRICTER = "код завершил, модель тянула"
KIND_MISMATCH = "скрипт vs вопрос"      # НАСТОЯЩЕЕ
SCRIPT_MISMATCH = "другой скрипт"       # НАСТОЯЩЕЕ
END_MISMATCH = "разошлось завершение"   # НАСТОЯЩЕЕ
STATE_MISMATCH = "разошлось состояние"  # НАСТОЯЩЕЕ

REAL = (KIND_MISMATCH, SCRIPT_MISMATCH, END_MISMATCH, STATE_MISMATCH)

# Ключи, которых в новой схеме нет намеренно: их отсутствие — не расхождение, а решение.
RETIRED_KEYS = frozenset({"STOP_ABROAD"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Переигрывание трасс screening_split через ядро policy (офлайн).")
    p.add_argument("--reports", default=DEFAULT_REPORTS, help="glob по *.cases.json")
    p.add_argument("--band-min", type=int, default=DEFAULT_BAND_MIN)
    p.add_argument("--band-max", type=int, default=DEFAULT_BAND_MAX)
    p.add_argument("--limit", type=int, default=0, help="ограничить число файлов (0 — все)")
    p.add_argument("--show", type=int, default=12, help="сколько настоящих расхождений печатать")
    p.add_argument("--out", default="", help="куда сложить json-отчёт (пусто — не писать)")
    p.add_argument("--guard-scan", action="store_true",
                   help="теневой прогон гардов по записанным сообщениям вместо сверки ядра")
    p.add_argument("--vacancy-url", default="https://example.com/vacancies/python-backend")
    p.add_argument("--contract-check", default="",
                   help="путь к config.yaml промпта-наблюдателя: сверить его схему с контрактом кода")
    return p


def contract_check(config_path: str) -> int:
    """Сверка JSON-схемы промпта с контрактом `policy.observation`.

    Промпт живёт в ДРУГОМ репозитории и релизится отдельно, поэтому словари могут разъехаться молча:
    модель вернёт сигнал, которого код не знает, и он будет тихо отброшен как «неизвестный».
    Проверка дешёвая и ловит расхождение до прогона.
    """
    import yaml
    from qa_harness.domain.screening_split.policy.observation import (
        ALL_SIGNALS, FOCUS_ANSWERED, MAX_SIGNALS)

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    schema = cfg["text_format"]["schema"]
    props = schema["properties"]

    problems: List[str] = []

    in_schema = set(props["signals"]["items"]["properties"]["code"]["enum"])
    if in_schema != set(ALL_SIGNALS):
        only_schema = sorted(in_schema - set(ALL_SIGNALS))
        only_code = sorted(set(ALL_SIGNALS) - in_schema)
        if only_schema:
            problems.append(f"сигналы есть в промпте, но код их не знает: {only_schema}")
        if only_code:
            problems.append(f"сигналы есть в коде, но промпт их не вернёт: {only_code}")

    in_schema_focus = set(props["focus_answered"]["enum"])
    if in_schema_focus != set(FOCUS_ANSWERED):
        problems.append(f"focus_answered разошёлся: промпт {sorted(in_schema_focus)}, "
                        f"код {sorted(FOCUS_ANSWERED)}")

    forbidden = {"next_action", "script_key", "instruction", "asking", "event"}
    leaked = sorted(forbidden & set(props))
    if leaked:
        problems.append(f"в схеме остались поля решения: {leaked}")

    if sorted(schema.get("required", [])) != sorted(props):
        problems.append("required не совпадает со списком полей (strict-режим требует полного списка)")

    print(f"схема: {config_path}")
    print(f"   сигналов: {len(in_schema)} · лимит в коде: {MAX_SIGNALS}")
    if problems:
        for item in problems:
            print(f"   !! {item}")
        return 1
    print("   контракт промпта и кода совпадает")
    return 0


# ── теневой прогон гардов ────────────────────────────────────────────────────

_TRACE_RE = re.compile(r"⟨trace:[^⟩]*⟩\s*$")


def _strip_trace(text: str) -> str:
    """Раннер дописывает к сообщению служебную метку трассы — она не часть текста кандидату."""
    return _TRACE_RE.sub("", text).strip()

def _vacancy_urls() -> Dict[int, str]:
    """index сценария → каноническая ссылка его вакансии.

    Без этого замер подставляет всем сценариям ДЕФОЛТНУЮ ссылку и «подменяет» правильные адреса
    сценариев со своей вакансией (64/66 — `hh.ru/vacancy/133203750`), завышая число срабатываний.
    """
    import yaml
    path = pathlib.Path("tests/fixtures/generation/screening_scenarios/scenario_vacancies.yaml")
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: Dict[int, str] = {}
    for item in doc.get("scenarios") or []:
        url = ((item.get("company_info") or {}).get("vacancy_url") or "").strip()
        if item.get("index") is not None:
            out[int(item["index"])] = url
    return out


def _scenario_index(case_id: str) -> Optional[int]:
    match = re.match(r"scenario:(\d+):", case_id or "")
    return int(match.group(1)) if match else None


def guard_scan(files: List[str], args: Any) -> int:
    """Сколько сообщений гарды тронули бы и какие именно.

    Смысл — измерить долю ложных вырезов ДО того, как гарды получат право резать в проде. Косметика
    (G0/G2/G5) считается отдельно: у неё риска нет по построению, и включать её можно сразу.
    """
    per_scenario = _vacancy_urls()
    trips: Counter = Counter()
    touched = cosmetic_only = total = 0
    samples: List[str] = []

    for path in files:
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for case in payload.get("cases") or []:
            index = _scenario_index(case.get("case_id") or "")
            url = per_scenario.get(index, args.vacancy_url) if index is not None else args.vacancy_url
            for item in case.get("transcript") or []:
                if item.get("role") == "candidate":
                    continue
                text = _strip_trace(item.get("text") or "")
                if not text.strip():
                    continue
                total += 1
                spec = GuardSpec(allow_urls=(url,) if url else ())
                result = apply_guards(text, spec)
                if not result.trips:
                    continue
                touched += 1
                codes = {t.split(".", 1)[0].replace("[тень] ", "") for t in result.trips}
                for trip in result.trips:
                    trips[trip.split(" (", 1)[0]] += 1
                if codes <= {"G0", "G2", "G5"}:
                    cosmetic_only += 1
                elif len(samples) < 8:
                    samples.append({"trips": "; ".join(result.trips),
                                    "before": text[:110], "after": result.text[:110]})

    print("")
    print(f"сообщений просмотрено: {total}")
    print("-" * 72)
    for code, count in sorted(trips.items()):
        print(f"   {code:<28} {count:>5}  {count / total * 100 if total else 0:5.1f}%")
    print("-" * 72)
    print(f"тронуто всего: {touched} ({touched / total * 100 if total else 0:.1f}%), "
          f"из них только косметика: {cosmetic_only}")
    if samples:
        print("")
        print("примеры защитных срабатываний:")
        for sample in samples:
            print(f"   {sample['trips']}")
            print(f"      было:  {sample['before']}")
            print(f"      стало: {sample['after']}")
    return 0


# ── реконструкция входа хода ─────────────────────────────────────────────────

def _init_state_from(first_state: Dict[str, Any]) -> Dict[str, Any]:
    """Стартовое состояние по первому записанному снимку.

    Трасса хранит ПРОЕКЦИЮ состояния (`conversation.py:82-90`): без служебных полей и с вопросами в
    виде `{key: status}`. Поэтому переигрываем не «с середины», а всю беседу с нуля, ведя полное
    состояние у себя, и сравниваем ту же проекцию.
    """
    fmt = (first_state or {}).get("format_check")
    work_format = "remote" if fmt == "n/a" else "office"
    keys = list((first_state or {}).get("questions", {}).keys())
    questions_text = "\n".join(f"Вопрос {i}" for i in range(1, len(keys) + 1))
    return state_model.init_state(work_format, questions_text)


def _project(state: Dict[str, Any]) -> Dict[str, Any]:
    """Та же проекция, что пишет раннер, — иначе сравнивать не с чем."""
    return {
        "salary": state.get("salary"),
        "format_check": state.get("format_check"),
        "city": state.get("candidate_city"),
        "questions": {q["key"]: q["status"] for q in state.get("questions", [])},
        "counters": dict(state.get("counters", {})),
    }


def _pairs(transcript: List[Dict[str, Any]]) -> List[tuple]:
    """(реплика кандидата, ход ассистента) по раундам."""
    by_round: Dict[Any, Dict[str, Any]] = {}
    for item in transcript or []:
        slot = by_round.setdefault(item.get("round"), {})
        slot[item.get("role")] = item
    out = []
    for rnd in sorted(by_round, key=lambda x: (x is None, x)):
        slot = by_round[rnd]
        if "assistant" in slot:
            out.append(((slot.get("candidate") or {}).get("text", ""), slot["assistant"]))
    return out


# ── сравнение ────────────────────────────────────────────────────────────────

def _compare(plan: Any, assistant: Dict[str, Any], decision: Dict[str, Any]) -> tuple:
    """(категория, пояснение). Категория из REAL требует разбора, остальные — ожидаемы."""
    exp_kind, exp_key = expected_outcome(decision)
    got_kind = plan.kind

    if exp_kind == "script" and exp_key in RETIRED_KEYS:
        return RETIRED_KEY, f"{exp_key} → {plan.reason_code}"

    if exp_kind == "script" and got_kind != "script":
        return KIND_MISMATCH, f"было script:{exp_key}, стало {got_kind}:{plan.reason_code}"
    if exp_kind == "ask" and got_kind == "script":
        if plan.rule == "R9.agenda_complete":
            return STRICTER, "повестка закрыта, модель вернула ask вместо FINISH"
        return KIND_MISMATCH, f"было ask:{exp_key}, стало script:{plan.reason_code}"

    if exp_kind == "script":
        if plan.reason_code != exp_key:
            return SCRIPT_MISMATCH, f"было {exp_key}, стало {plan.reason_code}"
        expected_end = bool(assistant.get("ended"))
        if plan.end != expected_end:
            return END_MISMATCH, f"было end={expected_end}, стало end={plan.end}"
        return MATCH, ""

    # ask
    if exp_key is None:
        return (MATCH, "") if plan.focus is None else (ANSWER_ONLY, f"код спрашивает {plan.focus}")
    if plan.focus != exp_key:
        return FOCUS_SHIFT, f"модель спрашивала {exp_key}, код — {plan.focus}"
    return MATCH, ""


def _state_diff(mine: Dict[str, Any], recorded: Optional[Dict[str, Any]]) -> str:
    if not recorded:
        return ""
    diffs = []
    for key in ("salary", "format_check", "city"):
        if mine.get(key) != recorded.get(key):
            diffs.append(f"{key}: {recorded.get(key)!r} → {mine.get(key)!r}")
    if mine.get("questions") != recorded.get("questions"):
        diffs.append(f"questions: {recorded.get('questions')} → {mine.get('questions')}")
    if mine.get("counters") != recorded.get("counters"):
        for k in sorted(set(mine.get("counters", {})) | set(recorded.get("counters", {}))):
            a, b = recorded.get("counters", {}).get(k, 0), mine.get("counters", {}).get(k, 0)
            if a != b:
                diffs.append(f"counters.{k}: {a} → {b}")
    return "; ".join(diffs)


# ── прогон ───────────────────────────────────────────────────────────────────

def replay_case(case: Dict[str, Any], ctx: DecideContext) -> Dict[str, Any]:
    pairs = _pairs(case.get("transcript") or [])
    if not pairs:
        return {"turns": 0, "results": [], "stopped": "нет ходов"}

    first_state = (pairs[0][1] or {}).get("state") or {}
    state = _init_state_from(first_state)

    results: List[Dict[str, Any]] = []
    stopped = ""
    for index, (message, assistant) in enumerate(pairs, start=1):
        decision = assistant.get("decision") or {}
        ok, why = is_replayable(decision)
        if not ok:
            stopped = f"ход {index}: {why}"
            break

        obs = decision_to_observation(decision, message)
        plan = decide(state, obs, message, ctx)

        # Самофильтрация корпуса: если зарплатный разбор ЭТОГО хода не сходится с записанным, трасса
        # снята другой версией `salary.py` (абсолютный потолок правдоподобия сняли 27.08). Сравнивать
        # дальше нечего — состояние с этого хода разъедется, и все последующие расхождения будут
        # следствием версии, а не ядра.
        recorded_salary = assistant.get("salary") or {}
        mine_salary = plan.audit.get("salary") or {}
        if recorded_salary and (recorded_salary.get("status") != mine_salary.get("status")
                                or recorded_salary.get("verdict") != mine_salary.get("verdict")):
            stopped = (f"ход {index}: другая версия salary.py "
                       f"({recorded_salary.get('status')}/{recorded_salary.get('verdict')} → "
                       f"{mine_salary.get('status')}/{mine_salary.get('verdict')})")
            break

        category, detail = _compare(plan, assistant, decision)

        mine = _project(plan.state_next)
        state_note = _state_diff(mine, assistant.get("state"))
        if category == MATCH and state_note:
            category, detail = STATE_MISMATCH, state_note

        results.append({
            "turn": index,
            "category": category,
            "detail": detail,
            "state_diff": state_note,
            "rule": plan.rule,
            "reason_code": plan.reason_code,
            "was": f"{decision.get('next_action')}:{decision.get('script_key') or decision.get('asking')}",
        })
        state = plan.state_next
        if plan.end:
            break

    return {"turns": len(pairs), "results": results, "stopped": stopped}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    files = sorted(glob.glob(args.reports))
    if args.limit:
        files = files[-args.limit:]
    if not files:
        print(f"нет файлов по маске {args.reports}")
        return 2

    # Гео-ограничение включено: трассы, где модель вернула KO_GEO, снимались на вакансии с явным
    # ограничением, а самой вакансии в отчёте нет. Допущение переигрывания, не свойство ядра.
    if args.contract_check:
        return contract_check(args.contract_check)

    if args.guard_scan:
        return guard_scan(files, args)

    ctx = DecideContext(band_min=args.band_min, band_max=args.band_max,
                        work_format="remote", location="Москва", has_geo_restriction=True)

    tally: Counter = Counter()
    stopped: Counter = Counter()
    real_examples: List[Dict[str, Any]] = []
    cases_total = replayed_turns = 0

    for path in files:
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"пропущен {os.path.basename(path)}: {exc}")
            continue
        for case in payload.get("cases") or []:
            cases_total += 1
            outcome = replay_case(case, ctx)
            if outcome["stopped"]:
                stopped[outcome["stopped"].split(":", 1)[1].strip()] += 1
            for item in outcome["results"]:
                replayed_turns += 1
                tally[item["category"]] += 1
                if item["category"] in REAL and len(real_examples) < 500:
                    real_examples.append({**item, "case": case.get("case_id"),
                                          "file": os.path.basename(path)})

    real_total = sum(tally[c] for c in REAL)
    print(f"\nфайлов: {len(files)} · кейсов: {cases_total} · переиграно ходов: {replayed_turns}")
    print("-" * 72)
    for category, count in tally.most_common():
        mark = "!!" if category in REAL else "  "
        share = (count / replayed_turns * 100) if replayed_turns else 0
        print(f"{mark} {category:<26} {count:>5}  {share:5.1f}%")
    print("-" * 72)
    print(f"настоящих расхождений: {real_total} из {replayed_turns}"
          f" ({(real_total / replayed_turns * 100) if replayed_turns else 0:.1f}%)")

    if stopped:
        print("\nкейсы, прерванные (ход не переигрывается — решение подменил код):")
        for reason, count in stopped.most_common(8):
            print(f"   {reason:<48} {count:>4}")

    if real_examples and args.show:
        print(f"\nпримеры настоящих расхождений (первые {min(args.show, len(real_examples))}):")
        for item in real_examples[:args.show]:
            print(f"   [{item['category']}] {item['file']} · кейс {item['case']} · ход {item['turn']}")
            print(f"      было {item['was']} → стало {item['reason_code']} (правило {item['rule']})")
            if item["detail"]:
                print(f"      {item['detail']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"files": len(files), "cases": cases_total, "turns": replayed_turns,
                       "tally": dict(tally), "real_total": real_total,
                       "examples": real_examples[:200]},
                      fh, ensure_ascii=False, indent=2)
        print(f"\nотчёт: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

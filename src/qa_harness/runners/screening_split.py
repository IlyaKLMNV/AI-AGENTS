"""Раннер screening_split: тест НОВОГО раздельного скрининга (Аналитик + Интервьюер).

Split = два промпта из пакета `prompts` (`screening_analyzer` — «мозг», строгий JSON
Decision; `screening_interviewer` — «рот», одно сообщение) + КОД-оркестратор (состояние,
счётчики/пороги, фиксированные скрипты), портированный из tgApi 1:1
(qa_harness.domain.screening_split). Тестируется как в проде: тела/схема — из пакета
`prompts` (local-источник), арифметика состояний — в коде.

СЦЕНАРИИ — отдельный CSV (`tests/fixtures/screening_split/scenarios.csv`, копия golden
монолита + новый зарплатный кейс). Легаси-раннер screening_scenarios и его CSV не трогаем.

Режимы (по мере готовности этапов):
- `--offline` — плумбинг: грузим сценарии, извлекаем реплики кандидата, прогоняем чистый
  порт домена (render_script/init_state/apply_updates) — без сети и без пакета `prompts`;
- golden / `--generate` — живой прогон движка + судья (появятся на следующих этапах).

  python -m qa_harness.runners.screening_split --offline
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, List

from qa_harness.core import add_prompt_source_args
from qa_harness.domain import screening_split as sp
from qa_harness.domain.screening_scenarios import (
    Scenario,
    extract_candidate_examples,
    load_scenarios,
    parse_scenario_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_CSV = FIXTURES / "screening_split" / "scenarios.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "reports_v2"
RUNNER = "screening_split"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="screening_split QA runner (Аналитик + Интервьюер; local prompts).")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV сценариев split (по умолч. отдельный от легаси).")
    p.add_argument("--sample", type=int, default=5, help="Случайная выборка N сценариев (0 = все).")
    p.add_argument("--scenario-indices", default=None, help="Точечные номера строк CSV, напр. 1,7,65 (override --sample).")
    p.add_argument("--max-examples", type=int, default=4, help="Сколько реплик кандидата брать из примеров на сценарий.")
    p.add_argument("--offline", action="store_true", help="Плумбинг: сценарии + реплики + санити чистого домена, без сети.")
    p.add_argument("--seed", type=int, default=None, help="Seed выборки сценариев.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--quiet", action="store_true")
    add_prompt_source_args(p)  # --prompt-source/--local-prompt-version/--prompts-path (для онлайна, этап 2+)
    return p


def _select(scenarios: List[Scenario], indices_raw: str | None, sample: int, seed: Any) -> List[Scenario]:
    if indices_raw:  # точечный выбор — как просили, без фильтра
        wanted = parse_scenario_indices(indices_raw)
        by_idx = {s.index: s for s in scenarios}
        return [by_idx[i] for i in wanted if i in by_idx]
    if sample and sample > 0:
        return random.Random(seed).sample(scenarios, min(sample, len(scenarios)))
    return scenarios


def _domain_sanity() -> List[str]:
    """Быстрый прогон чистого порта домена (без сети): доказывает, что оркестрационные
    примитивы перенесены рабочими. Не оценка промпта — санити инфраструктуры."""
    out: List[str] = []
    ko = sp.render_script("KO_FORMAT_OFFICE", city="Москва")
    out.append(f"render KO_FORMAT_OFFICE(city=Москва): city_grammar={'в городе Москва' in (ko or '')} · terminal={sp.is_terminal('KO_FORMAT_OFFICE')}")
    st_office = sp.init_state("office", "1. Опыт с Python?\n2. SQL?")
    st_remote = sp.init_state("remote", "")
    out.append(f"init_state office format_check={st_office['format_check']} questions={[q['key'] for q in st_office['questions']]} · remote format_check={st_remote['format_check']}")
    st2 = sp.apply_updates(st_office, [{"key": "salary", "value": "closed"}, {"key": "candidate_city", "value": "Казань"}])
    st3 = sp.apply_updates(st2, [], event="gibberish")
    out.append(f"apply_updates salary={st2['salary']} city={st2['candidate_city']} gibberish_counter={st3['counters']['gibberish']}")
    dec, err = sp.parse_and_validate('{"next_action":"ask","script_key":null,"instruction":"Спроси зарплату","updates":[],"event":null,"asking":"salary"}')
    out.append(f"Decision-валидатор: valid={dec is not None} err={err or '—'}")
    return out


def _run_offline(args: argparse.Namespace, scenarios: List[Scenario]) -> None:
    with_ex = turns = 0
    for s in scenarios:
        ct = extract_candidate_examples(s.examples_raw, args.max_examples)
        mark = f"реплик-кандидата: {len(ct)}" if ct else "нет реплик кандидата"
        if ct:
            with_ex += 1
            turns += len(ct)
        if not args.quiet:
            print(f"  {s.index:>2} {s.name[:60]:<60} {mark}")
    print(f"\n[offline] сценариев={len(scenarios)} с_примерами={with_ex} реплик_всего={turns} (плумбинг, без судьи)")
    print("[offline] санити чистого домена (порт tgApi):")
    for line in _domain_sanity():
        print(f"  · {line}")
    print("[offline] сеть и пакет prompts не дёргались.")


def run(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.csv)
    selected = _select(scenarios, args.scenario_indices, args.sample, args.seed)
    print(f"Сценариев в CSV: {len(scenarios)} · выбрано: {len(selected)} · CSV: {args.csv}")

    if args.offline:
        _run_offline(args, selected)
        return

    raise NotImplementedError(
        "Онлайн-режим (golden/--generate) появится на Этапе 2 (движок + судья). "
        "Сейчас доступен только --offline."
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

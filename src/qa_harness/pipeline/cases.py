"""Источники тест-кейсов для extractor/one_line: real (из файлов), suite (встроенные),
synthetic (деградированные запросы). Перенос из extractor_agent_runner.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from qa_harness.core.jsonio import safe_json_loads

SUITE_CASES: List[Dict[str, str]] = [
    {"name": "suite_0001", "input": "Python разработчик"},
    {"name": "suite_0002", "input": "React developer"},
    {"name": "suite_0003", "input": "ключевые слова: Иван Петров"},
    {"name": "suite_0004", "input": "React + TypeScript + Next.js"},
    {"name": "suite_0005", "input": "Django или Flask"},
    {"name": "suite_0006", "input": "Python developer, но не Django"},
    {"name": "suite_0007", "input": "SRE / DevOps"},
    {"name": "suite_0008", "input": "C++ developer"},
    {"name": "suite_0009", "input": "опыт с g++ и clang"},
    {"name": "suite_0010", "input": "React, русский обязателен, английский желателен"},
    {"name": "suite_0011", "input": "React, английский B2, русский не нужен"},
    {"name": "suite_0012", "input": "Backend разработчик Москва офис"},
    {"name": "suite_0013", "input": "Python разработчик опыт от 5 лет"},
    {"name": "suite_0014", "input": "DevOps инженер опыт 3-5 лет"},
    {"name": "suite_0015", "input": "контакт: ivan.petrov@gmail.com, +7 999 123-45-67"},
    {"name": "suite_0016", "input": "Платов Анатолий backend"},
    {"name": "suite_0017", "input": "DevOps из финтеха"},
    {"name": "suite_0018", "input": "SRE, последнее место работы: Яндекс"},
]


@dataclass
class Case:
    name: str
    input: str
    source: str  # real | suite | syn


def guess_source(name: str) -> str:
    if name.startswith("suite_"):
        return "suite"
    if name.startswith("syn_"):
        return "syn"
    return "real"


def _input_from_mapping(obj: Dict[str, Any]) -> Optional[str]:
    raw = obj.get("input") or obj.get("query") or obj.get("text") or obj.get("user_phrase")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _case_from_mapping(obj: Dict[str, Any], stem: str, idx: Optional[int] = None) -> Optional[Case]:
    raw = _input_from_mapping(obj)
    if not raw:
        return None
    name_raw = obj.get("name")
    if isinstance(name_raw, str) and name_raw.strip():
        name = name_raw.strip()
    else:
        name = f"{stem}_{idx:04d}" if idx is not None else stem
    return Case(name=name, input=raw, source=guess_source(name))


def _cases_from_data(data: Any, stem: str) -> List[Case]:
    out: List[Case] = []
    if isinstance(data, str):
        if data.strip():
            out.append(Case(name=stem, input=data.strip(), source=guess_source(stem)))
        return out
    if isinstance(data, dict):
        nested = data.get("cases")
        if isinstance(nested, list):
            data = nested
        else:
            c = _case_from_mapping(data, stem=stem)
            return [c] if c else []
    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, str) and item.strip():
                name = f"{stem}_{i:04d}"
                out.append(Case(name=name, input=item.strip(), source=guess_source(name)))
            elif isinstance(item, dict):
                c = _case_from_mapping(item, stem=stem, idx=i)
                if c:
                    out.append(c)
    return out


def load_cases_from_dir(cases_dir: Path) -> List[Case]:
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        raise FileNotFoundError(f"cases_dir not found: {cases_dir}")
    cases: List[Case] = []
    for p in sorted(x for x in cases_dir.rglob("*") if x.is_file() and x.name[0] != "."):
        text = p.read_text(encoding="utf-8-sig")
        if p.suffix.lower() == ".json":
            obj, err = safe_json_loads(text)
            cases.extend(_cases_from_data(text if err else obj, p.stem))
        elif p.suffix.lower() in (".yml", ".yaml"):
            cases.extend(_cases_from_data(yaml.safe_load(text), p.stem))
        elif text.strip():
            cases.append(Case(name=p.stem, input=text, source=guess_source(p.stem)))
    uniq: Dict[str, Case] = {c.name: c for c in cases}
    return list(uniq.values())


def build_suite_cases(limit: int, seed: int) -> List[Case]:
    rnd = random.Random(seed)
    items = [dict(x) for x in SUITE_CASES]
    rnd.shuffle(items)
    if limit > 0:
        items = items[:limit]
    return [Case(name=it["name"], input=it["input"], source="suite") for it in items]


def build_synthetic_cases(base_cases: List[Case], n: int, seed: int) -> List[Case]:
    rnd = random.Random(seed)
    if not base_cases or n <= 0:
        return []
    out: List[Case] = []
    for i in range(1, n + 1):
        tokens = rnd.choice(base_cases).input.strip().split()
        if len(tokens) > 3:
            drop_p = rnd.uniform(0.15, 0.35)
            kept = [t for t in tokens if rnd.random() > drop_p]
            if len(kept) >= 2:
                tokens = kept
        if len(tokens) > 6 and rnd.random() < 0.35:
            tokens = tokens[: rnd.randint(2, max(3, len(tokens) // 2))]
        if tokens and rnd.random() < 0.25:
            j = rnd.randrange(len(tokens))
            t = tokens[j]
            if len(t) >= 5:
                k = rnd.randint(1, len(t) - 2)
                tokens[j] = t[:k] + t[k + 1] + t[k] + t[k + 2:]
        out.append(Case(name=f"syn_{i:04d}", input=" ".join(tokens).strip(), source="syn"))
    return out


def parse_mix_ratios(s: str) -> Dict[str, int]:
    ratios = {"real": 1, "suite": 0, "syn": 0}
    for p in (s or "").split(","):
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip()
            if k in ratios:
                try:
                    ratios[k] = max(0, int(v.strip()))
                except ValueError:
                    pass
    return ratios


def _sample(cases: List[Case], n: int, rnd: random.Random) -> List[Case]:
    if n <= 0 or not cases:
        return []
    if n <= len(cases):
        return rnd.sample(cases, n)
    out = [cases[i % len(cases)] for i in range(n)]
    rnd.shuffle(out)
    return out


def mix_cases(
    real_cases: List[Case],
    suite_cases: List[Case],
    syn_cases: List[Case],
    per_unit_count: int,
    ratios: Dict[str, int],
    seed: int,
    total_limit: Optional[int] = None,
) -> Tuple[List[Case], Dict[str, int]]:
    rnd = random.Random(seed)
    picked: List[Case] = []
    picked += _sample(real_cases, per_unit_count * int(ratios.get("real", 0)), rnd)
    picked += _sample(suite_cases, per_unit_count * int(ratios.get("suite", 0)), rnd)
    picked += _sample(syn_cases, per_unit_count * int(ratios.get("syn", 0)), rnd)
    rnd.shuffle(picked)
    if total_limit and total_limit > 0:
        picked = picked[:total_limit]
    counts = {"real": 0, "suite": 0, "syn": 0}
    for c in picked:
        counts[c.source] = counts.get(c.source, 0) + 1
    counts["total"] = len(picked)
    return picked, counts


def parse_steps(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [1, 2, 3]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        try:
            v = int(part)
            if v in (1, 2, 3) and v not in out:
                out.append(v)
        except ValueError:
            pass
    return out or [1, 2, 3]

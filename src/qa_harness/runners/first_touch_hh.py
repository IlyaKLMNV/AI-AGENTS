"""Раннер first_touch_hh — вариант первого касания для HeadHunter.

Тонкая обёртка над `first_touch`: тот же конвейер (генерация → LLM-судья фактов + эвристики), но
промпт-компонент `first_touch_hh` и своя golden-фикстура. Доп. правило HH — НЕ упоминать источник
(сообщение шлётся на hh) — задаётся `forbid_in_message` в golden. Судья/эвристики/контракт — общие.

  python -m qa_harness.runners.first_touch_hh --offline
  python -m qa_harness.runners.first_touch_hh
"""

from __future__ import annotations

from pathlib import Path

from qa_harness.runners.first_touch import build_parser, run

REPO_ROOT = Path(__file__).resolve().parents[3]
HH_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "first_touch_hh" / "golden.yaml"


def main() -> None:
    run(build_parser(default_component="first_touch_hh", default_golden=HH_GOLDEN).parse_args())


if __name__ == "__main__":
    main()

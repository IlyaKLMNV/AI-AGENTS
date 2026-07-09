"""qa_harness.core — переиспользуемые инфраструктурные примитивы.

В этом подпакете живёт ТОЛЬКО инфраструктура (учёт токенов, JSON-IO, конфиг,
LLM-клиент, отчёты, метрики, CLI). Доменная логика рекрутинга — в qa_harness.domain.
См. docs/REFACTOR_PLAN.md §2 (граница core ↔ domain).
"""

from .usage import accumulate_usage, blank_usage, extract_usage_numbers, usage_total
from .jsonio import expect_json_object, extract_json_substring, safe_json_loads
from .config import PromptCfg, component_cfg, default_cfg_path, load_cfg, resolve_prompt
from .prompt_source import (
    LOCAL,
    STORED,
    add_prompt_source_args,
    ensure_prompts_importable,
    load_local_spec,
    make_prompt_client,
    prompt_under_test_meta,
    resolve_source,
)
from .run_loop import LoopOutcome, run_cases

__all__ = [
    # usage
    "blank_usage",
    "extract_usage_numbers",
    "accumulate_usage",
    "usage_total",
    # jsonio
    "safe_json_loads",
    "extract_json_substring",
    "expect_json_object",
    # config
    "PromptCfg",
    "load_cfg",
    "component_cfg",
    "resolve_prompt",
    "default_cfg_path",
    # prompt_source (переключатель stored ↔ local)
    "STORED",
    "LOCAL",
    "resolve_source",
    "add_prompt_source_args",
    "make_prompt_client",
    "load_local_spec",
    "ensure_prompts_importable",
    "prompt_under_test_meta",
    # run_loop
    "run_cases",
    "LoopOutcome",
]

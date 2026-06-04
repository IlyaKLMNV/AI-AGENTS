"""Загрузка реестра промптов и резолв prompt_id/version.

P0-3 (docs/REFACTOR_PLAN.md): НЕ заводим вторую копию model.yaml. Этот модуль читает
существующий `tests/tools/model.yaml` напрямую (или путь из env QA_HARNESS_CFG),
поэтому дрейф двух копий невозможен. Перенос в config/ — только на cutover.

Резолв промпта: CLI > env (<PREFIX>_PROMPT_ID/_VERSION) > model.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ENV_CFG = "QA_HARNESS_CFG"


def default_cfg_path() -> Path:
    """Путь к model.yaml: env QA_HARNESS_CFG, иначе tests/tools/model.yaml в корне репо."""
    env = os.environ.get(_ENV_CFG)
    if env:
        return Path(env)
    # src/qa_harness/core/config.py -> parents[3] == корень репозитория
    return Path(__file__).resolve().parents[3] / "tests" / "tools" / "model.yaml"


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_cfg(path: Optional[Path] = None) -> Dict[str, Any]:
    """Прочитать model.yaml (кэшируется по пути)."""
    return _load_cached(str(path or default_cfg_path()))


def component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Секция компонента из реестра (или пустой dict)."""
    block = cfg.get(name)
    return block if isinstance(block, dict) else {}


@dataclass(frozen=True)
class PromptCfg:
    """Резолвнутая конфигурация stored-промпта."""

    component: str
    prompt_id: str
    prompt_version: Optional[str] = None
    seed: Optional[int] = None


def resolve_prompt(
    cfg: Dict[str, Any],
    component: str,
    *,
    env_prefix: Optional[str] = None,
    cli_id: Optional[str] = None,
    cli_version: Optional[str] = None,
) -> PromptCfg:
    """Резолв prompt_id/version с приоритетом CLI > env > model.yaml.

    env_prefix по умолчанию = component.upper() (например, "message_classifier" ->
    переменные MESSAGE_CLASSIFIER_PROMPT_ID / MESSAGE_CLASSIFIER_PROMPT_VERSION).
    Бросает ValueError, если prompt_id не найден ни в одном источнике.
    """
    prefix = (env_prefix or component).upper()
    block = component_cfg(cfg, component)

    pid = cli_id or os.environ.get(f"{prefix}_PROMPT_ID") or block.get("prompt_id")
    if not pid:
        raise ValueError(
            f"prompt_id for '{component}' not found "
            f"(checked CLI, env {prefix}_PROMPT_ID, model.yaml[{component}])"
        )

    pver = cli_version or os.environ.get(f"{prefix}_PROMPT_VERSION") or block.get("prompt_version")
    seed = block.get("seed")

    return PromptCfg(
        component=component,
        prompt_id=str(pid),
        prompt_version=str(pver) if pver is not None else None,
        seed=int(seed) if seed is not None else None,
    )

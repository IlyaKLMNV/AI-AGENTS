"""Переключатель источника промпта-под-тестом: stored ↔ local.

Два источника тела/параметров промпта:
- **stored** — platform.openai.com (`prompt={id, version}`), как исторически. ДЕФОЛТ (обратная
  совместимость: без флага ничего не меняется).
- **local** — пакет `prompts` (репозиторий podbor/prompts): тело в `system.md`, параметры в
  `config.yaml`, вызов обычным `responses.create(model=..., input=messages, ...)`. Единый источник
  правды для прода и тестов — «тестируется ровно то, что в проде».

Ключ выбора (простой, по одному значению):
    --prompt-source {stored,local}     ·  или env QA_HARNESS_PROMPT_SOURCE

Версия в local-режиме (можно тестировать не-дефолтную версию, хотя задефолчена другая через
pointer.yaml active в самом пакете):
    --local-prompt-version vN          ·  приоритет: этот флаг > model.yaml[local_version] >
                                          env <COMPONENT>_PROMPT_VERSION > pointer.yaml active.

Где берётся пакет `prompts`:
    - **как в проде** — установленный пакет (релиз из GHCR-образа ghcr.io/podbor/prompts,
      см. docs/LOCAL_PROMPTS.md). Это дефолт: тестируется ровно тот релиз, что поедет в прод.
    - **для дева** — исходники через явный --prompts-path PATH или env PROMPTS_REPO_PATH.
      Неявного авто-подхвата соседнего ../prompts НЕТ: он маскировал бы отсутствие установленного
      релиза (тестировали бы незакоммиченную локальную копию вместо релиза).

Импорт `prompts` — ленивый: stored-режим и офлайн его не требуют.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

STORED = "stored"
LOCAL = "local"
SOURCES = (STORED, LOCAL)

_ENV_SOURCE = "QA_HARNESS_PROMPT_SOURCE"
_ENV_PATH = "PROMPTS_REPO_PATH"

_prompts_ready = False


def resolve_source(cli_value: Optional[str]) -> str:
    """Итоговый источник: CLI > env QA_HARNESS_PROMPT_SOURCE > 'stored'."""
    val = (cli_value or os.environ.get(_ENV_SOURCE) or STORED).strip().lower()
    if val not in SOURCES:
        raise ValueError(f"неизвестный prompt-source: {val!r} (допустимо: {', '.join(SOURCES)})")
    return val


def add_prompt_source_args(parser: Any) -> None:
    """Зарегистрировать общие флаги переключателя источника (единообразно во всех раннерах)."""
    g = parser.add_argument_group("prompt source (stored ↔ local package `prompts`)")
    g.add_argument("--prompt-source", choices=list(SOURCES), default=None,
                   help="Источник промпта-под-тестом: stored (platform.openai.com, дефолт) "
                        "или local (пакет prompts). Также env QA_HARNESS_PROMPT_SOURCE.")
    g.add_argument("--local-prompt-version", default=None, metavar="vN",
                   help="Пин версии в пакете prompts (напр. v51). По умолчанию — pointer.yaml active. "
                        "Действует только с --prompt-source local.")
    g.add_argument("--prompts-path", default=None, metavar="PATH",
                   help="ДЕВ-опция: путь к исходникам репозитория prompts вместо установленного релиза. "
                        "Также env PROMPTS_REPO_PATH. По умолчанию используется установленный пакет (как в проде).")


def _candidate_paths(explicit: Optional[str]) -> Iterator[Path]:
    # ТОЛЬКО явные источники: --prompts-path и env PROMPTS_REPO_PATH. Неявного соседнего ../prompts
    # НЕТ намеренно — иначе харнесс молча тестировал бы локальную копию вместо установленного релиза.
    if explicit:
        yield Path(explicit)
    env = os.environ.get(_ENV_PATH)
    if env:
        yield Path(env)


def ensure_prompts_importable(prompts_path: Optional[str] = None) -> None:
    """Гарантировать, что `import prompts` сработает.

    Приоритет — установленный пакет (как в проде: релиз из GHCR-образа). Явный путь к исходникам
    (--prompts-path / env PROMPTS_REPO_PATH) — только дев-обходной путь. Идемпотентно.
    """
    global _prompts_ready
    if _prompts_ready:
        return
    # дев-override исходниками имеет приоритет, только если задан явно
    for p in _candidate_paths(prompts_path):
        try:
            if (p / "prompts" / "__init__.py").is_file():
                sys.path.insert(0, str(p))
                _prompts_ready = True
                print(f"[prompt-source] ВНИМАНИЕ: local берётся из ДЕВ-исходников {p} "
                      f"(не установленный релиз из GHCR).", file=sys.stderr)
                return
        except OSError:
            continue
    try:
        import prompts  # noqa: F401  (установленный релиз — основной путь)
        _prompts_ready = True
        return
    except ImportError:
        pass
    raise ImportError(
        "Пакет `prompts` не найден. Как в проде — установи релиз из GHCR-образа "
        "ghcr.io/podbor/prompts:latest (см. docs/LOCAL_PROMPTS.md, раздел «Установка релиза из GHCR»). "
        "Для локального дева можно указать исходники через --prompts-path / env PROMPTS_REPO_PATH."
    )


def load_local_spec(component: str, version: Optional[str] = None, *,
                    prompts_path: Optional[str] = None) -> Any:
    """Вернуть PromptSpec из пакета `prompts` (боевая версия из pointer.yaml, если version=None)."""
    ensure_prompts_importable(prompts_path)
    from prompts import registry
    return registry.get(component, version)


def make_prompt_client(
    prompt_cfg: Any,
    *,
    source: str,
    local_version: Optional[str] = None,
    prompts_path: Optional[str] = None,
    client: Any = None,
    text_format: Optional[dict] = None,
) -> Any:
    """Фабрика клиента промпта-под-тестом по выбранному источнику.

    Возвращает объект с общим контрактом .run(input_text) -> (text, usage): StoredPromptClient
    (source=stored) либо LocalPromptClient (source=local). local_version переопределяет версию из
    model.yaml (иначе pointer.yaml active).
    """
    from qa_harness.core.llm_client import LocalPromptClient, StoredPromptClient

    if source == LOCAL:
        ensure_prompts_importable(prompts_path)
        version = local_version or getattr(prompt_cfg, "local_version", None)
        return LocalPromptClient(prompt_cfg.local_component, version, client=client, text_format=text_format)
    return StoredPromptClient(prompt_cfg.prompt_id, prompt_cfg.prompt_version,
                              client=client, text_format=text_format)


def prompt_under_test_meta(prompt_cfg: Any, source: str, local_version: Optional[str] = None, *,
                          prompts_path: Optional[str] = None) -> dict:
    """Метаданные промпта-под-тестом для отчёта (что реально тестировалось и откуда).

    Для local резолвит фактическую версию и модель из пакета `prompts` (а не платформенный
    номер из model.yaml): в отчёт попадает `local_version` вида `v13` и `model` из config.yaml —
    то, что реально исполнялось. Резолв безопасен: при недоступности пакета — мягкий фолбэк.
    """
    meta = {
        "component": prompt_cfg.component,
        "source": source,
        "prompt_id": prompt_cfg.prompt_id,
        "prompt_version": prompt_cfg.prompt_version,
    }
    if source == LOCAL:
        meta["local_component"] = prompt_cfg.local_component
        version = local_version or getattr(prompt_cfg, "local_version", None)
        try:
            spec = load_local_spec(prompt_cfg.local_component, version, prompts_path=prompts_path)
            meta["local_version"] = spec.version  # разрезолвленная (напр. v13), а не "active"
            meta["model"] = spec.model             # реальная LLM-модель из config.yaml пакета
        except Exception:  # noqa: BLE001  пакет недоступен (напр. offline) — не роняем отчёт
            meta["local_version"] = version or "active"
    return meta

"""Ядро split-скрининга (TG-канал) для QA-стенда ai-agents.

Split = два промпта из пакета `prompts` (`screening_analyzer` — «мозг», возвращает `Observation`;
`screening_interviewer` — «рот», одно сообщение) + КОД-оркестратор: чистое ядро `decide()` держит
состояние, считает счётчики/пороги и выбирает причину хода, гарды правят исходящую строку.

Оркестратор живёт в `policy/` — этот же пакет переносится в `tgApi` и `eggplant-api`. Рядом лежат
части, которые в продуктовые репозитории НЕ едут: `selfcheck/` (офлайн-гейт), `checks.py`,
`interviewer_judge.py`, `candidate_script.py`, in-memory стор и QA-наблюдаемость движка.

Домен НЕ импортирует `app`/`adapters` (контракт qa_harness ⊥ app) и не тянет `openai`/`prompts` на
уровне модуля — клиенты приходят снаружи, поэтому пакет импортируется офлайн.
"""

from .candidate_script import build_scripted_turns, load_candidate_inputs, resolve_convey, salary_directive
from .checks import (
    CheckResult,
    LeakResult,
    evaluate_analyzer,
    evaluate_dialogue,
    injection_scan,
    leak_scan,
    load_checks,
)
from .context import build_context, build_interviewer_seed, candidate_source, salary_display
from .conversation import SplitConversation, TurnResult
from .errors import AssistantError
from .interviewer import PolicyInterviewer
from .interviewer_judge import InterviewerJudge, InterviewerVerdict
from .salary import (
    ABSENT,
    ACTIONABLE,
    UNUSABLE,
    claim_status,
    compare_with_band,
    normalize,
    read_claim,
)
from .salary_rules import SALARY_RULES_VERSION
from .state import apply_updates, init_state, is_complete, progress_signature
from .store import InMemoryStateStore

__all__ = [
    # оркестратор
    "InMemoryStateStore",
    "SplitConversation",
    "TurnResult",
    # роли
    "PolicyInterviewer",
    "AssistantError",
    # детерминированные проверки (слой A/B)
    "evaluate_analyzer",
    "evaluate_dialogue",
    "injection_scan",
    "leak_scan",
    "load_checks",
    "CheckResult",
    "LeakResult",
    # скриптовые входы кандидата (C1) + директивы генератору (Фаза 2)
    "load_candidate_inputs",
    "build_scripted_turns",
    "salary_directive",
    "resolve_convey",
    # судья Интервьюера (слой B, семантика)
    "InterviewerJudge",
    "InterviewerVerdict",
    # зарплата: распознавание за Наблюдателем (salary_claim), пересчёт и вердикт за кодом
    "read_claim",
    "claim_status",
    "normalize",
    "compare_with_band",
    "ACTIONABLE",
    "UNUSABLE",
    "ABSENT",
    "SALARY_RULES_VERSION",
    # чистые примитивы (state/context)
    "init_state",
    "apply_updates",
    "is_complete",
    "progress_signature",
    "build_context",
    "build_interviewer_seed",
    "candidate_source",
    "salary_display",
]

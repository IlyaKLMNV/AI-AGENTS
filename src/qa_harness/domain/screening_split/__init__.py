"""Порт прод-движка split-скрининга (tgApi) для QA-стенда ai-agents.

Split = два промпта из пакета `prompts` (`screening_analyzer` — «мозг», строгий JSON
Decision; `screening_interviewer` — «рот», одно сообщение) + КОД-оркестратор, который
держит состояние, считает счётчики/пороги и рендерит фиксированные скрипты. Всё это
перенесено 1:1 из tgApi (HEAD e733095), чтобы тест гонял ровно прод-логику; единственные
адаптации — inject-зависимости вместо app-импортов, dict вместо ScreeningVacancyDTO,
in-memory стор вместо Mongo и QA-наблюдаемость (last_decision/last_state/last_usage).

Домен НЕ импортирует `app`/`adapters` (контракт qa_harness ⊥ app) и не тянет `openai`/
`prompts` на уровне модуля — клиенты приходят снаружи, поэтому пакет импортируется офлайн.
"""

from .analyzer import ScreeningAnalyzer
from .candidate_script import build_scripted_turns, load_candidate_inputs, resolve_convey, salary_directive
from .checks import CheckResult, LeakResult, evaluate_analyzer, injection_scan, leak_scan, load_checks
from .context import build_context, build_interviewer_seed, candidate_source, salary_display
from .conversation import SplitConversation, TurnResult
from .decision import REQUIRED_FIELDS, parse_and_validate
from .engine import ConversationResult, ScreeningSplitEngine
from .errors import AssistantError
from .interviewer import PolicyInterviewer, ScreeningInterviewer
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
from .scripts import is_known, is_terminal, render_script
from .state import apply_updates, init_state, is_complete, progress_signature
from .store import InMemoryStateStore

__all__ = [
    # оркестратор
    "ScreeningSplitEngine",
    "ConversationResult",
    "InMemoryStateStore",
    "SplitConversation",
    "TurnResult",
    # роли
    "ScreeningAnalyzer",
    "PolicyInterviewer",
    "ScreeningInterviewer",
    # контракт Decision
    "parse_and_validate",
    "REQUIRED_FIELDS",
    "AssistantError",
    # детерминированные проверки (слой A/B)
    "evaluate_analyzer",
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
    # зарплата: распознавание за Аналитиком (salary_claim), пересчёт и вердикт за кодом
    "read_claim",
    "claim_status",
    "normalize",
    "compare_with_band",
    "ACTIONABLE",
    "UNUSABLE",
    "ABSENT",
    "SALARY_RULES_VERSION",
    # чистые примитивы (state/scripts/context)
    "init_state",
    "apply_updates",
    "is_complete",
    "progress_signature",
    "render_script",
    "is_terminal",
    "is_known",
    "build_context",
    "build_interviewer_seed",
    "candidate_source",
    "salary_display",
]

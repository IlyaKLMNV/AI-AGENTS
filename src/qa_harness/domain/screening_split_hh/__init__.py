"""HH-канал split-скрининга для QA-стенда (параллельно `screening_split` = TG).

Оркестратор hh-канала (потребитель — eggplant-api, где боевой split-движок ещё не реализован).
Эталон механики — `prompts/docs/EGGPLANT_SPLIT_TASK.md` (§3 STATE, §5 реестр скриптов, §9 поток хода)
+ `docs/SPLIT_TG_VS_HH.md` (построчная дельта промптов). Идентичные с TG куски импортируются из
`screening_split` (без дрейфа); переопределяется только hh-дельта (scripts/state/context/decision/engine).

Домен НЕ импортирует `app`/`adapters` (контракт qa_harness ⊥ app).
"""

# переиспользуемые (канало-независимые) части TG-модуля
from qa_harness.domain.screening_split.candidate_script import (
    build_scripted_turns,
    load_candidate_inputs,
    resolve_convey,
    salary_directive,
)
from qa_harness.domain.screening_split.conversation import TurnResult
from qa_harness.domain.screening_split.engine import ConversationResult
from qa_harness.domain.screening_split.errors import AssistantError
from qa_harness.domain.screening_split.interviewer import ScreeningInterviewer
from qa_harness.domain.screening_split.interviewer_judge import InterviewerJudge, InterviewerVerdict
from qa_harness.domain.screening_split.store import InMemoryStateStore

from .analyzer import ScreeningAnalyzer
from .checks import CheckResult, LeakResult, evaluate_analyzer, leak_scan, load_checks
from .context import allowed_formats_of, build_context, build_interviewer_seed, salary_display
from .conversation import SplitConversation
from .decision import REQUIRED_FIELDS, parse_and_validate
from .engine import NO_PROGRESS_CAP, ScreeningSplitEngine
from .scripts import is_known, is_terminal, render_script
from .state import (
    COUNTER_KEYS,
    apply_updates,
    init_state,
    is_complete,
    normalize_work_formats,
    progress_signature,
)

__all__ = [
    # оркестратор
    "ScreeningSplitEngine",
    "ConversationResult",
    "InMemoryStateStore",
    "SplitConversation",
    "TurnResult",
    "NO_PROGRESS_CAP",
    # роли
    "ScreeningAnalyzer",
    "ScreeningInterviewer",
    # контракт Decision (hh)
    "parse_and_validate",
    "REQUIRED_FIELDS",
    "AssistantError",
    # детерминированные проверки (слой A/B) — hh
    "evaluate_analyzer",
    "leak_scan",
    "load_checks",
    "CheckResult",
    "LeakResult",
    # скриптовые входы кандидата + директивы генератору (канало-независимо)
    "load_candidate_inputs",
    "build_scripted_turns",
    "salary_directive",
    "resolve_convey",
    # судья Интервьюера (слой B, семантика; канало-независим)
    "InterviewerJudge",
    "InterviewerVerdict",
    # чистые примитивы (state/scripts/context) — hh-дельта
    "init_state",
    "apply_updates",
    "is_complete",
    "progress_signature",
    "normalize_work_formats",
    "COUNTER_KEYS",
    "render_script",
    "is_terminal",
    "is_known",
    "build_context",
    "build_interviewer_seed",
    "salary_display",
    "allowed_formats_of",
]

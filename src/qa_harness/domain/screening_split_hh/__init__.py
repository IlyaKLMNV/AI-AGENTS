"""HH-канал split-скрининга для QA-стенда (параллельно `screening_split` = TG).

Оркестратор hh-канала — `policy/` (потребитель порта — eggplant-api). Идентичные с TG куски
импортируются из `screening_split` (без дрейфа); переопределяется только hh-дельта: мультиформат,
разъездной формат, свой реестр причин, отсутствие источника контакта.

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
from qa_harness.domain.screening_split.errors import AssistantError
from qa_harness.domain.screening_split.interviewer import PolicyInterviewer
from qa_harness.domain.screening_split.interviewer_judge import InterviewerJudge, InterviewerVerdict
from qa_harness.domain.screening_split.store import InMemoryStateStore

from .checks import (
    CheckResult,
    LeakResult,
    evaluate_analyzer,
    evaluate_dialogue,
    injection_scan,
    leak_scan,
    load_checks,
)
from .context import allowed_formats_of, build_context, build_interviewer_seed, salary_display
from .conversation import SplitConversation
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
    "InMemoryStateStore",
    "SplitConversation",
    "TurnResult",
    # роли
    "PolicyInterviewer",
    "AssistantError",
    # детерминированные проверки (слой A/B) — hh
    "evaluate_analyzer",
    "evaluate_dialogue",
    "injection_scan",
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
    # чистые примитивы (state/context) — hh-дельта
    "init_state",
    "apply_updates",
    "is_complete",
    "progress_signature",
    "normalize_work_formats",
    "COUNTER_KEYS",
    "build_context",
    "build_interviewer_seed",
    "salary_display",
    "allowed_formats_of",
]

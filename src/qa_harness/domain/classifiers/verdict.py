"""Классификатор итогового вердикта диалога (passed/failed/deadlock).

- StoredPromptVerdictClassifier — онлайн, stored-промпт verdict_classifier (тестируемый
  компонент). Поведение перенесено из старого VerdictClassifierRunner.
- HeuristicVerdictClassifier — офлайн, детерминированный (regex-маркеры), без сети.
  Заглушка для CI/демо, не тестируемый компонент.

extract_label обобщён на любые метки, поэтому verdict переиспользует судью LabelJudge(VERDICTS).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ..judge.label_judge import extract_label
from ..text.message_markers import has_any_pattern

VERDICTS = ("passed", "failed", "deadlock")

ClassifyResult = Tuple[Optional[str], str, Any]  # (verdict, raw_output, usage)


class StoredPromptVerdictClassifier:
    """Онлайн-классификатор диалога через stored-промпт verdict_classifier."""

    def __init__(self, client: Any, labels: Sequence[str] = VERDICTS) -> None:
        self._client = client
        self._labels = tuple(labels)

    def classify(self, dialogue: str) -> ClassifyResult:
        raw, usage = self._client.run(dialogue.strip())
        verdict = extract_label(raw, self._labels)
        if verdict not in self._labels:
            raise ValueError(f"verdict_classifier returned invalid output: {raw!r}")
        return verdict, raw, usage


# --- офлайн-эвристика (детерминированная, без сети; стаб для CI/демо) -------------

_DEADLOCK_MARKERS = (
    r"не\s+тот\s+человек", r"ошиблись\s+(номером|контактом)", r"\bэто\s+не\s+я\b",
    r"не\s+по\s+адресу", r"легитим", r"источник\s+контакта", r"корпоративн",
    r"неразборчив", r"не\s+разобрать",
)
_FAILED_MARKERS = (
    r"не\s+подходит", r"отказыва", r"\bотказ", r"не\s+интересно", r"не\s+актуальн",
    r"не\s+готов", r"выше\s+бюджета", r"не\s+рассматрива", r"нет\s+(нужного\s+)?опыта",
    r"уже\s+(принял|трудоустроен)", r"вы\s+бот", r"вы\s+ии",
)


class HeuristicVerdictClassifier:
    """Офлайн-эвристика: deadlock-маркеры -> deadlock, иначе failed-маркеры -> failed, иначе passed."""

    def classify(self, dialogue: str) -> ClassifyResult:
        # смотрим только реплики кандидата + общий текст
        text = dialogue or ""
        if has_any_pattern(text, _DEADLOCK_MARKERS):
            verdict = "deadlock"
        elif has_any_pattern(text, _FAILED_MARKERS):
            verdict = "failed"
        else:
            verdict = "passed"
        return verdict, verdict, None

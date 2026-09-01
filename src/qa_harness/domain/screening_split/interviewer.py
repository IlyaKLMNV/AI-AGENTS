"""Интервьюер split-скрининга — «рот». Формулирует РОВНО ОДНО сообщение кандидату по инструкции
Наблюдателя; решений по существу не принимает, END не пишет.

**Stateless**: истории не видит: всё, что нужно сказать, приходит в инструкции этого хода, а код
добавляет ровно два факта — сид с именами участников и предыдущее отправленное сообщение (чтобы
переспрос не вышел дословно тем же текстом). Ни одно правило промпта Интервьюера историей не
пользуется: запреты на повтор там ограничены одним сообщением. Раз conversation нет, `store` гасится,
как во всех остальных вызовах харнесса (`core.llm_client.STORE_RESPONSES`), и прогоны не сорят в логи
platform, где смотрят прод.

Адаптация под ai-agents: spec (`screening_interviewer` из пакета `prompts`) и OpenAI-клиент
инжектятся. Возвращает (text, usage) — QA нужен учёт.
"""

from typing import Any

from qa_harness.core.llm_client import STORE_RESPONSES


def _params(spec: Any, kwargs: dict, attrs: tuple[str, ...]) -> dict:
    """None => «не задано в config.yaml» — параметр не передаём (как у потребителя пакета)."""
    for attr in attrs:
        val = getattr(spec, attr, None)
        if val is not None:
            kwargs[attr] = val
    return kwargs


class PolicyInterviewer:
    """(instruction, message) + сид и предыдущая реплика → одно сообщение кандидату."""

    def __init__(self, spec: Any, client: Any) -> None:
        self._spec = spec
        self._client = client

    def run(self, instruction: str, message: str, *, seed: str = "",
            last_sent: str = "") -> tuple[str, Any]:
        """Вернуть (text_сообщения, usage)."""
        kwargs: dict[str, Any] = {
            "model": self._spec.model,
            "input": self._build_turn(instruction, message, seed, last_sent),
            "instructions": self._spec.system_text,
            "text": {"format": self._spec.text_format},
            "store": STORE_RESPONSES,
        }
        kwargs = _params(self._spec, kwargs, ("temperature", "top_p", "max_output_tokens"))
        resp = self._client.responses.create(**kwargs)
        return (getattr(resp, "output_text", "") or "").strip(), getattr(resp, "usage", None)

    @staticmethod
    def _build_turn(instruction: str, message: str, seed: str = "", last_sent: str = "") -> str:
        parts = []
        if seed:
            parts.append(f"[Кто ты]: {seed}")
        if last_sent:
            parts.append(f"[Твоё предыдущее сообщение кандидату, не повторяй его дословно]: {last_sent}")
        if message:
            parts.append(f"[Сообщение кандидата]: {message}")
        if instruction:
            parts.append(f"[Внутренняя инструкция]: {instruction}")
        return "\n\n".join(parts)

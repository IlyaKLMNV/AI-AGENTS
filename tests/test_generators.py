"""Офлайн-тесты генератора сообщений и валидации (фейковый клиент, без сети)."""

from __future__ import annotations

import random

import pytest

from qa_harness.domain.generators import (
    CandidateMessageGenerator,
    MessageSpec,
    pick_scenario_hint,
    validate_candidate_message,
)
from qa_harness.domain.judge import CLASSES


class _FakeGenClient:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, input_text):
        self.calls.append(input_text)
        return self._text, {"input_tokens": 5, "output_tokens": 7}


def test_generator_parses_and_accumulates_usage():
    gen = CandidateMessageGenerator(_FakeGenClient("  Спасибо, но мне не подходит формат, вынужден отказаться.  "))
    spec = MessageSpec(cdm={"vacancy": {"title": "QA"}, "candidate": {}}, target_class="reason_farewell", scenario_hint="hint", noise_level=1)
    msg = gen.generate(spec)
    assert msg == "Спасибо, но мне не подходит формат, вынужден отказаться."
    assert gen.usage == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
    # инструкция + payload реально ушли в клиент
    assert "TARGET_CLASS = reason_farewell" in gen._client.calls[0]


def test_generator_collapses_newlines():
    gen = CandidateMessageGenerator(_FakeGenClient("первая строка\n\nвторая строка"))
    assert gen.generate(MessageSpec({}, "acceptance", "h", 0)) == "первая строка вторая строка"


def test_generator_empty_raises():
    gen = CandidateMessageGenerator(_FakeGenClient("   "))
    with pytest.raises(ValueError):
        gen.generate(MessageSpec({}, "acceptance", "h", 0))


@pytest.mark.parametrize(
    "target,message,is_valid",
    [
        ("reason_farewell", "Вынужден отказаться, не подходит формат работы.", True),
        ("reason_farewell", "Вынужден отказаться.", False),  # нет причины
        ("no_reason", "Нет, спасибо.", True),
        ("no_reason", "Отказываюсь, формат не подходит.", False),  # протекла причина
        ("acceptance", "Интересно, пришлите описание вакансии?", True),
        ("acceptance", "Вынужден отказаться.", False),  # маркеры отказа
        ("human_needed", "Что за бред, откуда нашли мой контакт?", True),
    ],
)
def test_validate_candidate_message(target, message, is_valid):
    err = validate_candidate_message(target, message)
    assert (err is None) == is_valid


def test_pick_scenario_hint_cycle_is_deterministic():
    rng = random.Random(0)
    cycle = {c: 0 for c in CLASSES}
    h1 = pick_scenario_hint("no_reason", rng, "cycle", None, cycle)
    h2 = pick_scenario_hint("no_reason", rng, "cycle", None, cycle)
    assert h1 != h2  # цикл идёт по пулу
    # ограничение пула
    rng2 = random.Random(0)
    h = pick_scenario_hint("no_reason", rng2, "cycle", 1, {c: 0 for c in CLASSES})
    assert isinstance(h, str) and h

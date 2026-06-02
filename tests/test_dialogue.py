"""Офлайн-тесты диалоговых хелперов, валидации и генератора диалогов."""

from __future__ import annotations

import pytest

from qa_harness.domain.classifiers import HeuristicVerdictClassifier
from qa_harness.domain.generators import DialogueGenerator, DialogueSpec, validate_generated_dialogue
from qa_harness.domain.text import speaker_for_line, split_dialogue_lines

VALID = (
    "Рекрутер: Здравствуйте! Расскажите про опыт?\n"
    "Кандидат: Опыт есть, 5 лет.\n"
    "Рекрутер: Спасибо, информацию передам дальше. END"
)


def test_split_and_speaker():
    lines = split_dialogue_lines(VALID)
    assert len(lines) == 3
    assert speaker_for_line(lines[0]) == "recruiter"
    assert speaker_for_line(lines[1]) == "candidate"
    # обрезаем всё после первой реплики рекрутера с END
    assert len(split_dialogue_lines(VALID + "\nКандидат: лишняя строка")) == 3


def test_validate_ok():
    out = validate_generated_dialogue(VALID, cdm={"vacancy": {}}, target_verdict="passed", scenario_hint="x")
    assert out.strip().endswith("END")


@pytest.mark.parametrize(
    "bad",
    [
        "Рекрутер: привет\nКандидат: ок\nРекрутер: пока",                 # нет END
        "Кандидат: привет\nРекрутер: ок END",                            # не с рекрутера
        "Рекрутер: a\nРекрутер: b\nРекрутер: c END",                     # не чередуется
        "Рекрутер: наш бюджет большой\nКандидат: ок\nРекрутер: спасибо END",  # утечка бюджета
    ],
)
def test_validate_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate_generated_dialogue(bad, cdm={"vacancy": {}}, target_verdict="passed", scenario_hint="x")


class _FakeDialogueClient:
    def create(self, input_text):
        return VALID, {"input_tokens": 3, "output_tokens": 2}


def test_dialogue_generator_validates_and_tracks_usage():
    gen = DialogueGenerator(_FakeDialogueClient())
    out = gen.generate(DialogueSpec(cdm={"vacancy": {}}, target_verdict="passed", scenario_hint="x", noise_level=1))
    assert out.strip().endswith("END")
    assert gen.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_heuristic_verdict_classifier():
    assert HeuristicVerdictClassifier().classify("Кандидат: это не я, ошиблись номером.")[0] == "deadlock"
    assert HeuristicVerdictClassifier().classify("Кандидат: мне не подходит, отказываюсь.")[0] == "failed"
    assert HeuristicVerdictClassifier().classify("Кандидат: всё отлично, готов общаться дальше.")[0] == "passed"

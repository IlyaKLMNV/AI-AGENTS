"""Юнит-тесты qa_harness.domain.judge (Verdict, LabelJudge, extract_label)."""

from __future__ import annotations

from qa_harness.domain.judge import CLASSES, LabelJudge, Verdict, extract_label


def test_extract_label():
    assert extract_label("Класс: acceptance.") == "acceptance"
    assert extract_label("HUMAN_NEEDED") == "human_needed"  # case-insensitive
    assert extract_label("reason_farewell") == "reason_farewell"
    assert extract_label("нет метки тут") is None
    assert extract_label("") is None


def test_label_judge_pass():
    v = LabelJudge(CLASSES).evaluate("acceptance", "acceptance")
    assert v.passed is True
    assert v.score == 1.0
    assert v.reason_codes == []
    assert v.evaluator == "label_match"


def test_label_judge_fail():
    v = LabelJudge(CLASSES).evaluate("no_reason", "acceptance")
    assert v.passed is False
    assert v.score == 0.0
    assert v.reason_codes == ["misclassified->no_reason"]


def test_verdict_to_dict_omits_empty():
    v = Verdict(passed=True, evaluator="label_match", score=1.0)
    d = v.to_dict()
    assert d == {"evaluator": "label_match", "passed": True, "score": 1.0}
    # reason_codes/comment/meta пустые -> не попадают в dict


def test_verdict_to_dict_includes_set_fields():
    v = Verdict(passed=False, evaluator="llm_judge", score=0.0, reason_codes=["x"], comment="bad", model="gpt-4.1", turn_ref=2)
    d = v.to_dict()
    assert d["reason_codes"] == ["x"]
    assert d["comment"] == "bad"
    assert d["model"] == "gpt-4.1"
    assert d["turn_ref"] == 2

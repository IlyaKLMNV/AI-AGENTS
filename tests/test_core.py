"""Юнит-тесты примитивов qa_harness.core — полностью офлайн, без сети.

Это та самая тестируемость, которую старые god-функции не давали
(см. docs/REFACTOR_PLAN.md). Запуск: pytest -q.
"""

from __future__ import annotations

import pytest

from qa_harness.core import (
    PromptCfg,
    accumulate_usage,
    blank_usage,
    component_cfg,
    expect_json_object,
    extract_json_substring,
    extract_usage_numbers,
    load_cfg,
    resolve_prompt,
    safe_json_loads,
    usage_total,
)
from qa_harness.core.llm_client import ModelClient, StoredPromptClient


# ----------------------------- usage -----------------------------

def test_blank_usage():
    assert blank_usage() == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_extract_usage_none():
    assert extract_usage_numbers(None) == (0, 0, 0)


def test_extract_usage_dict_total_present():
    assert extract_usage_numbers({"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}) == (2, 3, 5)


def test_extract_usage_dict_total_derived():
    # альтернативные имена полей + total отсутствует -> total = input + output
    assert extract_usage_numbers({"prompt_tokens": 2, "completion_tokens": 3}) == (2, 3, 5)


def test_extract_usage_object():
    class U:
        input_tokens = 4
        output_tokens = 6  # total_tokens отсутствует

    assert extract_usage_numbers(U()) == (4, 6, 10)


def test_accumulate_usage():
    bucket = blank_usage()
    accumulate_usage(bucket, {"input_tokens": 1, "output_tokens": 2})
    accumulate_usage(bucket, {"input_tokens": 3, "output_tokens": 4})
    assert bucket == {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}


def test_usage_total_normalized():
    assert usage_total({"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}) == {
        "input": 4,
        "output": 6,
        "total": 10,
    }


# ----------------------------- jsonio -----------------------------

def test_safe_json_loads_good():
    assert safe_json_loads('{"a": 1}') == ({"a": 1}, None)


def test_safe_json_loads_bad():
    obj, err = safe_json_loads("definitely not json")
    assert obj is None and err


def test_safe_json_loads_strict_rejects_surrounding_text():
    obj, err = safe_json_loads('prefix {"a": 1} suffix')
    assert obj is None and err  # строгий режим (как extractor) не выдёргивает подстроку


def test_safe_json_loads_lenient_extracts_substring():
    obj, err = safe_json_loads('prefix {"a": 1} suffix', lenient=True)
    assert obj == {"a": 1} and err is None


def test_extract_json_substring():
    assert extract_json_substring('x {"a": 1} y') == '{"a": 1}'
    assert extract_json_substring("[1, 2, 3]") == "[1, 2, 3]"
    assert extract_json_substring("no json here") is None


def test_expect_json_object_ok():
    assert expect_json_object({"a": 1}) == {"a": 1}


def test_expect_json_object_raises_on_error():
    with pytest.raises(ValueError):
        expect_json_object(None, "boom")


def test_expect_json_object_raises_on_non_dict():
    with pytest.raises(TypeError):
        expect_json_object([1, 2, 3])


# ----------------------------- config -----------------------------

def test_resolve_prompt_precedence(monkeypatch):
    cfg = {"foo": {"prompt_id": "yaml_id", "prompt_version": 7, "seed": 1234}}

    # 1) только yaml
    monkeypatch.delenv("FOO_PROMPT_ID", raising=False)
    monkeypatch.delenv("FOO_PROMPT_VERSION", raising=False)
    p = resolve_prompt(cfg, "foo")
    assert isinstance(p, PromptCfg)
    assert (p.prompt_id, p.prompt_version, p.seed) == ("yaml_id", "7", 1234)

    # 2) env перебивает yaml
    monkeypatch.setenv("FOO_PROMPT_ID", "env_id")
    monkeypatch.setenv("FOO_PROMPT_VERSION", "9")
    p = resolve_prompt(cfg, "foo")
    assert (p.prompt_id, p.prompt_version) == ("env_id", "9")

    # 3) CLI перебивает env
    p = resolve_prompt(cfg, "foo", cli_id="cli_id", cli_version="3")
    assert (p.prompt_id, p.prompt_version) == ("cli_id", "3")


def test_resolve_prompt_missing_raises():
    with pytest.raises(ValueError):
        resolve_prompt({}, "nonexistent_component")


def test_component_cfg_missing_returns_empty():
    assert component_cfg({}, "nope") == {}


def test_load_cfg_reads_real_model_yaml():
    # Валидирует, что default_cfg_path() резолвит tests/tools/model.yaml в корне репо.
    cfg = load_cfg()
    assert "screening_assistant" in cfg
    assert str(cfg["screening_assistant"]["prompt_id"]).startswith("pmpt_")


# ----------------------------- llm_client (offline, fake) -----------------------------

class _FakeResp:
    def __init__(self, text, usage):
        self.output_text = text
        self.usage = usage


class _FakeResponses:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResp("  hello  ", {"input_tokens": 3, "output_tokens": 5})


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_stored_prompt_client_passes_prompt_and_strips():
    fake = _FakeClient()
    c = StoredPromptClient("pmpt_x", "2", client=fake)
    text, usage = c.run("INPUT_JSON")
    assert text == "hello"  # .strip() применён
    assert usage == {"input_tokens": 3, "output_tokens": 5}
    assert fake.responses.last_kwargs["prompt"] == {"id": "pmpt_x", "version": "2"}
    assert fake.responses.last_kwargs["input"] == "INPUT_JSON"


def test_model_client_passes_model():
    fake = _FakeClient()
    c = ModelClient("gpt-4.1-mini", client=fake)
    text, usage = c.create("PROMPT")
    assert text == "hello"
    assert fake.responses.last_kwargs["model"] == "gpt-4.1-mini"
    assert fake.responses.last_kwargs["input"] == "PROMPT"

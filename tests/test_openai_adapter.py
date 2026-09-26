from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_engine.adapters.openai_messages import load_openai_traces


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _convo(messages: list[dict[str, object]], **extra: object) -> dict[str, object]:
    return {"id": "c1", "messages": messages, **extra}


def test_stitches_call_to_result(tmp_path: Path) -> None:
    path = _write(tmp_path, [_convo([
        {"role": "user", "content": "find X"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "function": {"name": "search", "arguments": "{\"q\": \"x\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "found X"},
        {"role": "assistant", "content": "X is found"},
    ])])
    r = load_openai_traces(path)[0]
    assert r.input == "find X"
    assert r.generated_answer == "X is found"
    step = r.trajectory.steps[0]
    assert step.tool == "search"
    assert step.args == {"q": "x"}          # JSON-string arguments parsed to dict
    assert step.result == "found X"          # stitched from the separate tool message


def test_arguments_string_is_parsed(tmp_path: Path) -> None:
    path = _write(tmp_path, [_convo([
        {"role": "user", "content": "t"},
        {"role": "assistant", "tool_calls": [
            {"id": "c", "function": {"name": "calc", "arguments": "{\"a\": 1, \"b\": 2}"}}
        ]},
        {"role": "tool", "tool_call_id": "c", "content": "3"},
    ])])
    r = load_openai_traces(path)[0]
    assert r.trajectory.steps[0].args == {"a": 1, "b": 2}


def test_call_without_result_becomes_error(tmp_path: Path) -> None:
    # tool_call with no matching tool message -> error step (XOR stays clean)
    path = _write(tmp_path, [_convo([
        {"role": "user", "content": "t"},
        {"role": "assistant", "tool_calls": [
            {"id": "orphan", "function": {"name": "search", "arguments": "{}"}}
        ]},
    ])])
    r = load_openai_traces(path)[0]
    step = r.trajectory.steps[0]
    assert step.error == "no tool result in trace"
    assert step.result is None


def test_multiple_steps_stitched_in_order(tmp_path: Path) -> None:
    path = _write(tmp_path, [_convo([
        {"role": "user", "content": "t"},
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "r1"},
        {"role": "assistant", "tool_calls": [
            {"id": "2", "function": {"name": "b", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "2", "content": "r2"},
    ])])
    steps = load_openai_traces(path)[0].trajectory.steps
    assert [s.tool for s in steps] == ["a", "b"]
    assert [s.result for s in steps] == ["r1", "r2"]


def test_non_array_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, {"not": "array"})
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_openai_traces(path)
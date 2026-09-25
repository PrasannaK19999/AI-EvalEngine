from __future__ import annotations

import pytest

from eval_engine.core.contracts import EvaluationRecord, MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.deterministic.arg_validity import ArgValidityMetric

SCHEMAS: dict[str, dict[str, object]] = {
    "search": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "calculator": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
        },
        "required": ["a", "b", "op"],
        "additionalProperties": False,
    },
}


def _record(steps: list[TrajectoryStep]) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="av-1",
        input="q",
        generated_answer="a",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        trajectory=Trajectory(steps=steps),
        expected_tools=None,
    )


def _step(tool: str, args: dict[str, object]) -> TrajectoryStep:
    return TrajectoryStep(tool=tool, args=args, result=1)


def test_all_args_valid() -> None:
    r = ArgValidityMetric(SCHEMAS).run(
        _record([
            _step("search", {"query": "france"}),
            _step("calculator", {"a": 1, "b": 2, "op": "add"}),
        ])
    )
    assert r.status == MetricStatus.SCORED
    assert r.score == 1.0


def test_missing_required_field_fails() -> None:
    r = ArgValidityMetric(SCHEMAS).run(_record([_step("search", {})]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.0
    assert "search" in (r.reasoning or "")


def test_wrong_type_fails() -> None:
    r = ArgValidityMetric(SCHEMAS).run(_record([_step("search", {"query": 123})]))
    assert r.score == 0.0


def test_enum_violation_fails() -> None:
    r = ArgValidityMetric(SCHEMAS).run(
        _record([_step("calculator", {"a": 1, "b": 2, "op": "power"})])
    )
    assert r.score == 0.0


def test_extra_key_fails_when_additional_properties_false() -> None:
    r = ArgValidityMetric(SCHEMAS).run(
        _record([_step("search", {"query": "x", "sneaky": True})])
    )
    assert r.score == 0.0


def test_partial_valid_scores_ratio() -> None:
    r = ArgValidityMetric(SCHEMAS).run(
        _record([
            _step("search", {"query": "ok"}),          # valid
            _step("calculator", {"a": 1, "op": "add"}),  # missing b -> invalid
        ])
    )
    assert r.score == 0.5


def test_unschemad_tool_is_skipped_not_failed() -> None:
    # one schemad (valid) + one unknown tool -> judged=1, valid=1 -> 1.0, unknown named
    r = ArgValidityMetric(SCHEMAS).run(
        _record([
            _step("search", {"query": "ok"}),
            _step("mystery_tool", {"whatever": 1}),
        ])
    )
    assert r.score == 1.0
    assert "mystery_tool" in (r.reasoning or "")


def test_all_unschemad_skips() -> None:
    r = ArgValidityMetric(SCHEMAS).run(_record([_step("mystery_tool", {"x": 1})]))
    assert r.status == MetricStatus.SKIPPED


def test_no_steps_skips() -> None:
    r = ArgValidityMetric(SCHEMAS).run(_record([]))
    assert r.status == MetricStatus.SKIPPED


def test_plain_record_skips() -> None:
    plain = EvaluationRecord(
        record_id="plain-1", 
        input="q",
        generated_answer="a", 
        model_id="test-model",
        latency_ms=1.0, 
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    r = ArgValidityMetric(SCHEMAS).run(plain)
    assert r.status == MetricStatus.SKIPPED


def test_invalid_schema_file_raises_at_construction() -> None:
    bad = {"search": {"type": "not_a_real_type"}}
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        ArgValidityMetric(bad)
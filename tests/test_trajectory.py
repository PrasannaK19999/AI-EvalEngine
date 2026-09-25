from __future__ import annotations

import pytest
from pydantic import ValidationError

from eval_engine.core.contracts import EvaluationRecord, MetricStatus, TokenUsage
from eval_engine.core.engine import EvaluationEngine
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.deterministic.latency import LatencyMetric
from eval_engine.metrics.deterministic.tool_selection import ToolSelectionMetric


def test_step_result_only_ok() -> None:
    step = TrajectoryStep(tool="search", args={"q": "x"}, result={"hits": 3})
    assert step.result == {"hits": 3}
    assert step.error is None


def test_step_error_only_ok() -> None:
    step = TrajectoryStep(tool="search", args={"q": "x"}, error="timeout")
    assert step.error == "timeout"
    assert step.result is None


def test_step_both_set_raises() -> None:
    with pytest.raises(ValidationError, match="both a result and an error"):
        TrajectoryStep(tool="search", result={"hits": 3}, error="timeout")


def test_step_neither_set_raises() -> None:
    with pytest.raises(ValidationError, match="result or an error"):
        TrajectoryStep(tool="search", args={"q": "x"})


def test_trajectory_allows_empty_steps() -> None:
    assert Trajectory().steps == []


def test_trajectory_holds_ordered_steps() -> None:
    a = TrajectoryStep(tool="search", result=1)
    b = TrajectoryStep(tool="fetch", result=2)
    traj = Trajectory(steps=[a, b])
    assert [s.tool for s in traj.steps] == ["search", "fetch"]


def _make_trajectory_record() -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="traj-1",
        input="What is the capital of France?",
        generated_answer="Paris.",
        model_id="gemini-3.5-flash-lite",
        latency_ms=120.0,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        trajectory=Trajectory(
            steps=[TrajectoryStep(tool="search", args={"q": "france"}, result={"hits": 1})]
        ),
    )


def test_trajectory_record_flows_through_phase1_engine() -> None:
    record = _make_trajectory_record()
    engine = EvaluationEngine(metrics=[LatencyMetric()])
    results = engine.evaluate([record])

    scored = [r for r in results if r.status == MetricStatus.SCORED]
    assert scored, "TrajectoryRecord must score on an inherited Phase-1 metric"
    assert all(r.record_id == "traj-1" for r in results)

def test_gate_skips_agent_metric_on_plain_record() -> None:
    plain = EvaluationRecord(
        record_id="plain-1",
        input="q",
        generated_answer="a",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    engine = EvaluationEngine(metrics=[ToolSelectionMetric()])
    results = engine.evaluate([plain])

    assert len(results) == 1
    assert results[0].status == MetricStatus.SKIPPED
    assert "lacks field" in (results[0].skip_reason or "")
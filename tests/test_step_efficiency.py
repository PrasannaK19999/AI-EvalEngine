from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord, MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.deterministic.step_efficiency import StepEfficiencyMetric


def _record(steps: list[TrajectoryStep]) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="se-1",
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


def test_all_unique_is_perfect() -> None:
    r = StepEfficiencyMetric().run(_record([
        _step("search", {"q": "france"}),
        _step("search", {"q": "germany"}),
        _step("calc", {"a": 1}),
    ]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 1.0


def test_exact_repeat_is_penalized() -> None:
    r = StepEfficiencyMetric().run(_record([
        _step("search", {"q": "france"}),
        _step("search", {"q": "france"}),  # identical (tool, args) -> redundant
    ]))
    assert r.score == 0.5
    assert "[1]search" in (r.reasoning or "")


def test_same_tool_different_args_not_penalized() -> None:
    r = StepEfficiencyMetric().run(_record([
        _step("search", {"q": "france"}),
        _step("search", {"q": "spain"}),
    ]))
    assert r.score == 1.0  # different queries are legitimate work


def test_arg_key_order_does_not_matter() -> None:
    # same args, different dict insertion order -> must count as identical
    r = StepEfficiencyMetric().run(_record([
        _step("calc", {"a": 1, "b": 2}),
        _step("calc", {"b": 2, "a": 1}),
    ]))
    assert r.score == 0.5  # sorted-keys canonicalization catches this


def test_multiple_duplicates_ratio() -> None:
    # 4 steps, 2 unique -> 0.5
    r = StepEfficiencyMetric().run(_record([
        _step("search", {"q": "x"}),
        _step("search", {"q": "x"}),
        _step("fetch", {"url": "u"}),
        _step("fetch", {"url": "u"}),
    ]))
    assert r.score == 0.5


def test_non_serializable_args_do_not_crash() -> None:
    # args value that isn't natively JSON-serializable -> default=str guard
    r = StepEfficiencyMetric().run(_record([_step("t", {"when": object()})]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 1.0


def test_no_steps_skips() -> None:
    r = StepEfficiencyMetric().run(_record([]))
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
    r = StepEfficiencyMetric().run(plain)
    assert r.status == MetricStatus.SKIPPED
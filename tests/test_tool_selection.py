from __future__ import annotations

from eval_engine.core.contracts import MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.deterministic.tool_selection import ToolSelectionMetric


def _record(used: list[str], expected: list[str] | None) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="ts-1",
        input="q",
        generated_answer="a",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        trajectory=Trajectory(steps=[TrajectoryStep(tool=t, result=1) for t in used]),
        expected_tools=expected,
    )


def test_perfect_match() -> None:
    r = ToolSelectionMetric().run(_record(["search", "calc"], ["search", "calc"]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 1.0
    assert "classification=perfect" in (r.reasoning or "")


def test_fewer_tools_hurts_recall() -> None:
    r = ToolSelectionMetric().run(_record(["search"], ["search", "calc"]))
    # precision 1.0, recall 0.5 -> F1 = 0.666...
    assert r.status == MetricStatus.SCORED
    assert 0.66 < (r.score or 0) < 0.67
    assert "classification=fewer_tools" in (r.reasoning or "")


def test_extra_tools_hurts_precision() -> None:
    used = ["a", "b", "c", "d", "e", "f", "g", "h", "x", "y"]  # 8 required + 2 junk
    expected = ["a", "b", "c", "d", "e", "f", "g", "h"]
    r = ToolSelectionMetric().run(_record(used, expected))
    # precision 0.8, recall 1.0 -> F1 = 0.888...
    assert 0.88 < (r.score or 0) < 0.89
    assert "classification=extra_tools" in (r.reasoning or "")
    assert "spurious=['x', 'y']" in (r.reasoning or "")


def test_wrong_tools_hurts_both() -> None:
    r = ToolSelectionMetric().run(_record(["search", "junk"], ["search", "calc"]))
    # precision 0.5, recall 0.5 -> F1 = 0.5
    assert r.score == 0.5
    assert "classification=wrong_tools" in (r.reasoning or "")


def test_empty_expected_and_no_tools_is_perfect() -> None:
    r = ToolSelectionMetric().run(_record([], []))
    assert r.status == MetricStatus.SCORED
    assert r.score == 1.0  # needed none, used none — not a crash, not a skip


def test_none_expected_skips() -> None:
    r = ToolSelectionMetric().run(_record(["search"], None))
    assert r.status == MetricStatus.SKIPPED
    assert r.skip_reason


def test_plain_record_without_trajectory_skips() -> None:
    from eval_engine.core.contracts import EvaluationRecord

    plain = EvaluationRecord(
        record_id="plain-1",
        input="q",
        generated_answer="a",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    r = ToolSelectionMetric().run(plain)
    assert r.status == MetricStatus.SKIPPED
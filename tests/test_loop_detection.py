from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord, MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.deterministic.loop_detection import LoopDetectionMetric


def _record(steps: list[TrajectoryStep]) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="ld-1",
        input="q",
        generated_answer="a",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        trajectory=Trajectory(steps=steps),
        expected_tools=None,
    )


def _step(tool: str, args: dict[str, object], result: object = 1) -> TrajectoryStep:
    return TrajectoryStep(tool=tool, args=args, result=result)


def test_no_repetition_is_perfect() -> None:
    r = LoopDetectionMetric().run(_record([
        _step("search", {"q": "france"}),
        _step("calc", {"a": 1}),
        _step("fetch", {"url": "u"}),
    ]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 1.0
    assert "no consecutive repetition" in (r.reasoning or "")


def test_consecutive_identical_is_a_loop() -> None:
    # same tool+args+result twice in a row, in a 2-step trajectory -> 1 - 1/2 = 0.5
    r = LoopDetectionMetric().run(_record([
        _step("search", {"q": "france"}, result="R"),
        _step("search", {"q": "france"}, result="R"),
    ]))
    assert r.score == 0.5
    assert "longest_consecutive_run=2" in (r.reasoning or "")


def test_longer_run_degrades_more() -> None:
    # 4 identical in a row out of 4 -> 1 - 3/4 = 0.25
    steps = [_step("search", {"q": "x"}, result="R") for _ in range(4)]
    r = LoopDetectionMetric().run(_record(steps))
    assert r.score == 0.25
    assert "longest_consecutive_run=4" in (r.reasoning or "")


def test_same_tool_different_args_is_not_a_loop() -> None:
    r = LoopDetectionMetric().run(_record([
        _step("search", {"q": "france"}),
        _step("search", {"q": "germany"}),
    ]))
    assert r.score == 1.0  # different query -> legitimate work


def test_same_action_different_result_is_not_a_loop() -> None:
    # identical tool+args, but the outcome changed -> progress, not stuck
    r = LoopDetectionMetric().run(_record([
        _step("write", {"doc": "d"}, result="draft1"),
        _step("write", {"doc": "d"}, result="draft2"),
    ]))
    assert r.score == 1.0


def test_drafting_agent_iteration_scores_clean() -> None:
    # write -> check -> write -> polish -> polish -> write, all evolving content
    r = LoopDetectionMetric().run(_record([
        _step("write", {"doc": "d"}, result="v1"),
        _step("check", {"doc": "d"}, result="ok1"),
        _step("write", {"doc": "d"}, result="v2"),
        _step("polish", {"doc": "d"}, result="p1"),
        _step("polish", {"doc": "d"}, result="p2"),
        _step("write", {"doc": "d"}, result="v3"),
    ]))
    assert r.score == 1.0  # no consecutive identical action+outcome -> no false positive


def test_identical_failed_retries_are_a_loop() -> None:
    # same failing call 3x in a row (result XOR error -> error rides the key)
    steps = [TrajectoryStep(tool="fetch", args={"url": "dead"}, error="timeout") for _ in range(3)]
    r = LoopDetectionMetric().run(_record(steps))
    assert r.score < 1.0
    assert "longest_consecutive_run=3" in (r.reasoning or "")


def test_non_consecutive_repeat_is_not_a_loop() -> None:
    # A B A -> same key at 0 and 2 but NOT consecutive -> longest run 1 -> 1.0
    r = LoopDetectionMetric().run(_record([
        _step("search", {"q": "x"}, result="R"),
        _step("calc", {"a": 1}, result="C"),
        _step("search", {"q": "x"}, result="R"),
    ]))
    assert r.score == 1.0  # step-efficiency would ding this; loop-detection correctly does not


def test_single_step_skips() -> None:
    r = LoopDetectionMetric().run(_record([_step("search", {"q": "x"})]))
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
    r = LoopDetectionMetric().run(plain)
    assert r.status == MetricStatus.SKIPPED
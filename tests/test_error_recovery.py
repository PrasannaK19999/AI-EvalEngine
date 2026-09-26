from __future__ import annotations

from eval_engine.core.contracts import MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.judges.error_recovery import ErrorRecoveryMetric


class FakeClient:
    """Returns a canned reply and records whether it was called at all."""

    def __init__(self, reply: str, model: str = "fake-model") -> None:
        self._reply = reply
        self._model = model
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._reply


class FakeCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def make_key(self, prompt: str, model: str, version: str) -> str:
        return f"{model}:{version}:{hash(prompt)}"

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


def _record(steps: list[TrajectoryStep]) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="er-1",
        input="Fetch the page and summarize it.",
        generated_answer="done",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        trajectory=Trajectory(steps=steps),
    )


def _ok(tool: str) -> TrajectoryStep:
    return TrajectoryStep(tool=tool, args={}, result="ok")


def _fail(tool: str) -> TrajectoryStep:
    return TrajectoryStep(tool=tool, args={}, error="timeout")


def test_clean_trajectory_skips_without_calling_client() -> None:
    client = FakeClient('{"score": 1.0, "reasoning": "unused"}')
    judge = ErrorRecoveryMetric(client, FakeCache())  # type: ignore[arg-type]
    r = judge.run(_record([_ok("fetch"), _ok("summarize")]))
    assert r.status == MetricStatus.SKIPPED
    assert client.calls == 0  # the O(n) gate skipped BEFORE any API call


def test_trajectory_with_error_runs_judge() -> None:
    client = FakeClient('{"score": 0.8, "reasoning": "retried with a fix"}')
    judge = ErrorRecoveryMetric(client, FakeCache())  # type: ignore[arg-type]
    r = judge.run(_record([_fail("fetch"), _ok("fetch")]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.8
    assert client.calls == 1  # a failure existed, so the judge WAS called


def test_malformed_reply_errors() -> None:
    client = FakeClient("garbage")
    judge = ErrorRecoveryMetric(client, FakeCache())  # type: ignore[arg-type]
    r = judge.run(_record([_fail("fetch"), _ok("fetch")]))
    assert r.status == MetricStatus.ERRORED
    assert r.error


def test_fenced_json_reply_scores() -> None:
    client = FakeClient('```json\n{"score": 0.4, "reasoning": "ignored the error"}\n```')
    judge = ErrorRecoveryMetric(client, FakeCache())  # type: ignore[arg-type]
    r = judge.run(_record([_fail("fetch"), _ok("other")]))
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.4
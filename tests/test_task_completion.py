from __future__ import annotations

from eval_engine.core.contracts import MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.judges.task_completion import TaskCompletionMetric


class FakeClient:
    """Stand-in for GeminiJudgeClient: returns a canned reply, never calls the network."""

    def __init__(self, reply: str, model: str = "fake-model") -> None:
        self._reply = reply
        self._model = model
        self.prompts: list[str] = []  # captures what the judge asked, for assertions

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._reply


class FakeCache:
    """Stand-in for JudgeCache: a plain dict, no SQLite file."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def make_key(self, prompt: str, model: str, version: str) -> str:
        return f"{model}:{version}:{hash(prompt)}"

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


def _record() -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id="tc-1",
        input="Find the capital of France and report it.",
        generated_answer="The capital is Paris.",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        trajectory=Trajectory(
            steps=[TrajectoryStep(tool="search", args={"q": "capital of France"}, result="Paris")]
        ),
    )


def _judge(reply: str) -> TaskCompletionMetric:
    return TaskCompletionMetric(FakeClient(reply), FakeCache())  # type: ignore[arg-type]


def test_valid_verdict_scores() -> None:
    judge = _judge('{"score": 0.9, "reasoning": "search found the capital, task done"}')
    r = judge.run(_record())
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.9
    assert r.judge_prompt_version == "task_completion_v1"


def test_malformed_reply_errors() -> None:
    judge = _judge("not json at all")
    r = judge.run(_record())
    assert r.status == MetricStatus.ERRORED
    assert r.error


def test_prompt_contains_task_and_steps() -> None:
    client = FakeClient('{"score": 1.0, "reasoning": "ok"}')
    judge = TaskCompletionMetric(client, FakeCache())  # type: ignore[arg-type]
    judge.run(_record())
    prompt = client.prompts[0]
    assert "capital of France" in prompt      # the task made it in
    assert "search" in prompt                 # the action made it in
    assert "Paris" in prompt                  # the result made it in


def test_fenced_json_reply_scores() -> None:
    # base._parse strips ```json fences — prove a fenced reply still works
    judge = _judge('```json\n{"score": 0.5, "reasoning": "partial"}\n```')
    r = judge.run(_record())
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.5
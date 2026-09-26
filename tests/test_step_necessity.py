from __future__ import annotations

from eval_engine.core.contracts import MetricStatus, TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.judges.step_necessity import StepNecessityMetric


class FakeClient:
    """Returns a canned reply, never hits the network."""

    def __init__(self, reply: str, model: str = "fake-model") -> None:
        self._reply = reply
        self._model = model
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._reply


class FakeCache:
    """Dict-backed stand-in for JudgeCache."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def make_key(self, prompt: str, model: str, version: str) -> str:
        return f"{model}:{version}:{hash(prompt)}"

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


def _record() -> TrajectoryRecord:
    # Task needs the value 4; step 1 finds it, steps 2-3 re-derive it (unnecessary).
    return TrajectoryRecord(
        record_id="sn-1",
        input="Compute the target value.",
        generated_answer="4",
        model_id="test-model",
        latency_ms=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        trajectory=Trajectory(steps=[
            TrajectoryStep(tool="calc", args={"expr": "1+3"}, result="4"),
            TrajectoryStep(tool="calc", args={"expr": "5-1"}, result="4"),
            TrajectoryStep(tool="calc", args={"expr": "2+2"}, result="4"),
        ]),
    )


def _judge(reply: str) -> StepNecessityMetric:
    return StepNecessityMetric(FakeClient(reply), FakeCache())  # type: ignore[arg-type]


def test_valid_verdict_scores() -> None:
    judge = _judge('{"score": 0.33, "reasoning": "only step 1 needed; 2-3 re-derived 4"}')
    r = judge.run(_record())
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.33
    assert r.judge_prompt_version == "step_necessity_v1"


def test_malformed_reply_errors() -> None:
    judge = _judge("garbage")
    r = judge.run(_record())
    assert r.status == MetricStatus.ERRORED
    assert r.error


def test_prompt_has_task_and_steps_but_not_final_answer() -> None:
    client = FakeClient('{"score": 1.0, "reasoning": "ok"}')
    judge = StepNecessityMetric(client, FakeCache())  # type: ignore[arg-type]
    judge.run(_record())
    prompt = client.prompts[0]
    assert "target value" in prompt        # task is in
    assert "1+3" in prompt                  # the actions are in
    assert "4" in prompt                    # results are in
    # necessity judges the PATH, not the outcome — the generated_answer is a value
    # already present via results, so we assert the design intent via the label instead:
    assert "FINAL RESPONSE" not in prompt   # no answer-outcome block, unlike task_completion


def test_fenced_json_reply_scores() -> None:
    judge = _judge('```json\n{"score": 0.5, "reasoning": "half unnecessary"}\n```')
    r = judge.run(_record())
    assert r.status == MetricStatus.SCORED
    assert r.score == 0.5
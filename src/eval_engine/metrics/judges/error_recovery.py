"""Recovery judge: when a step failed, did the following steps handle it well?"""

from __future__ import annotations

from typing import cast

from eval_engine.core.contracts import EvaluationRecord, MetricResult, MetricStatus
from eval_engine.core.trajectory import TrajectoryRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric
from eval_engine.metrics.judges.trajectory_render import render_trajectory

_TEMPLATE = """You are evaluating how well failures were handled during a task.

You are given a TASK and a log of ACTIONS taken to carry it out (each action is \
a tool call with its arguments and the result or error it produced). At least \
one action FAILED (produced an error).

Judge only how well the failures were handled by what came AFTER them. Good \
recovery means the later actions responded sensibly to the error — retrying with \
a fix, trying a reasonable alternative, or adjusting course. Poor recovery means \
ignoring the error, repeating the same failing action blindly, or giving up when \
the task was still achievable. Judge the response to failure only — not whether \
the task ultimately succeeded, and not the steps that had nothing to do with a \
failure.

TASK:
{task}

ACTIONS TAKEN (tool call -> result or error):
{trajectory}

Score how well the failures were handled, from 0.0 to 1.0:
- 1.0 = every failure was handled sensibly
- around 0.5 = some failures handled well, others poorly
- 0.0 = failures were ignored, blindly repeated, or led to giving up needlessly

Respond with a JSON object: {{"score": <0.0-1.0>, "reasoning": "<one sentence \
naming how each failure was handled>"}}"""


class ErrorRecoveryMetric(JudgeMetric):
    """ Judges how well failures were handled — but ONLY when a failure exists. """

    name = "recovery"
    required_fields = ("trajectory",)
    prompt_version = "error_recovery_v1"

    def run(self, record: EvaluationRecord) -> MetricResult:
        rec = cast(TrajectoryRecord, record)  # gate guarantees a TrajectoryRecord

        # Cheap deterministic pre-check guards the expensive Gemini call:
        # no failure in the trajectory -> nothing to recover from -> SKIP, no API call.
        has_error = any(step.error is not None for step in rec.trajectory.steps)
        if not has_error:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.SKIPPED,
                skip_reason="no failed step — nothing to recover from",
            )

        return super().run(record)  # a failure exists: run the judge

    def build_prompt(self, record: EvaluationRecord) -> str:
        rec = cast(TrajectoryRecord, record)
        return _TEMPLATE.format(
            task=rec.input,
            trajectory=render_trajectory(rec.trajectory),
        )
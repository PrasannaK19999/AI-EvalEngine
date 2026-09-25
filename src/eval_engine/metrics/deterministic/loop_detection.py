"""Loop-detection metric: did the agent get stuck repeating the same action?"""

from __future__ import annotations

import json

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricCategory,
    MetricResult,
    MetricStatus,
)
from eval_engine.core.trajectory import TrajectoryRecord, TrajectoryStep
from eval_engine.metrics.base import BaseMetric


def _identity(step: TrajectoryStep) -> str:
    """Stable key for a step: same tool + same args + same outcome = same action.

    A step carries result XOR error, so whichever is present rides in the key.
    default=str hardens against non-JSON-serializable args/results (Any).
    """
    outcome = step.result if step.error is None else {"__error__": step.error}
    return json.dumps(
        {"tool": step.tool, "args": step.args, "outcome": outcome},
        sort_keys=True,
        default=str,
    )


class LoopDetectionMetric(BaseMetric):
    """Catches an agent stuck spinning: the same (tool, args, outcome) repeated
    CONSECUTIVELY with no progress. Distinct from step-efficiency, which measures
    total redundancy as a ratio; this flags consecutive stuck-repetition.

    Score = 1.0 - (longest_consecutive_run - 1) / total_steps.
    A run of 1 (no repeat) -> 1.0. Longer stuck runs degrade sharply.
    """

    name = "loop_detection"
    category = MetricCategory.DETERMINISTIC
    required_fields = ("trajectory",)

    def run(self, record: EvaluationRecord) -> MetricResult:
        if not isinstance(record, TrajectoryRecord):
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="record has no trajectory",
            )

        steps = record.trajectory.steps
        if len(steps) < 2:
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED,
                skip_reason="need at least 2 steps to detect a consecutive loop",
            )

        keys = [_identity(s) for s in steps]

        longest_run = 1
        current_run = 1
        longest_end = 0  # index where the longest run ends
        for i in range(1, len(keys)):
            if keys[i] == keys[i - 1]:
                current_run += 1
                if current_run > longest_run:
                    longest_run = current_run
                    longest_end = i
            else:
                current_run = 1

        total = len(steps)
        score = 1.0 - (longest_run - 1) / total

        if longest_run >= 2:
            start = longest_end - longest_run + 1
            looped_tool = steps[longest_end].tool
            reasoning = (
                f"longest_consecutive_run={longest_run} "
                f"tool={looped_tool} steps=[{start}..{longest_end}]"
            )
        else:
            reasoning = "no consecutive repetition"

        return MetricResult(
            metric_name=self.name, record_id=record.record_id,
            status=MetricStatus.SCORED, score=score, reasoning=reasoning,
        )
"""Step-efficiency metric: how much of the agent's work was non-redundant?"""

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
    """Stable key for a step: same tool + same args = same work.

    args is a dict (unhashable), so serialize with sorted keys for a
    canonical, comparable string. Distinct queries -> distinct keys.
    """
    return f"{step.tool}:{json.dumps(step.args, sort_keys=True, default=str)}"


class StepEfficiencyMetric(BaseMetric):
    name = "step_efficiency"
    category = MetricCategory.DETERMINISTIC
    required_fields = ("trajectory",)

    def run(self, record: EvaluationRecord) -> MetricResult:
        if not isinstance(record, TrajectoryRecord):
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="record has no trajectory",
            )

        steps = record.trajectory.steps
        if not steps:
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="no steps to score",
            )

        total = len(steps)
        keys = [_identity(s) for s in steps]
        unique = len(set(keys))
        score = unique / total

        seen: set[str] = set()
        duplicates: list[str] = []
        for i, k in enumerate(keys):
            if k in seen:
                duplicates.append(f"[{i}]{steps[i].tool}")
            seen.add(k)

        reasoning = f"unique={unique}/{total} duplicates={duplicates}"
        return MetricResult(
            metric_name=self.name, record_id=record.record_id,
            status=MetricStatus.SCORED, score=score, reasoning=reasoning,
        )
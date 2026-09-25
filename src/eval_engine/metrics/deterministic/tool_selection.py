"""Tool-selection metric: did the agent call the tools the task required — and only those?"""

from __future__ import annotations

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricCategory,
    MetricResult,
    MetricStatus,
)
from eval_engine.core.trajectory import TrajectoryRecord
from eval_engine.metrics.base import BaseMetric


def _classify(missed: set[str], spurious: set[str]) -> str:
    if not missed and not spurious:
        return "perfect"
    if missed and spurious:
        return "wrong_tools"
    if spurious:
        return "extra_tools"
    return "fewer_tools"


class ToolSelectionMetric(BaseMetric):
    """F1 of tools-used vs tools-expected. Recall catches missed required tools,
    precision catches spurious ones. Decision-only: ignores whether a tool then
    succeeded. Classifies the failure mode and names the offending tools."""

    name = "tool_selection"
    category = MetricCategory.DETERMINISTIC
    required_fields = ("expected_tools",)

    def run(self, record: EvaluationRecord) -> MetricResult:
        if not isinstance(record, TrajectoryRecord):
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="record has no trajectory",
            )

        # None = no answer key authored -> can't judge. [] = key says "no tools needed" -> judge it.
        if record.expected_tools is None:
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="no expected_tools authored",
            )

        used = {step.tool for step in record.trajectory.steps}
        expected = set(record.expected_tools)
        hits = len(used & expected)

        precision = hits / len(used) if used else 1.0        # no calls -> nothing spurious
        recall = hits / len(expected) if expected else 1.0    # nothing required -> nothing missed
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        missed = expected - used
        spurious = used - expected
        classification = _classify(missed, spurious)
        reasoning = (
            f"classification={classification} "
            f"missed={sorted(missed)} spurious={sorted(spurious)} "
            f"precision={precision:.2f} recall={recall:.2f}"
        )

        return MetricResult(
            metric_name=self.name, record_id=record.record_id,
            status=MetricStatus.SCORED, score=f1, reasoning=reasoning,
        )
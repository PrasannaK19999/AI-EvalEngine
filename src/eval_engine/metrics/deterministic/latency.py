"""Latency metric: packages the already-measured latency_ms into a MetricResult."""

from __future__ import annotations

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricResult,
    MetricStatus,
    MetricUnit,
)
from eval_engine.metrics.base import BaseMetric


class LatencyMetric(BaseMetric):
    """Reports request latency. No computation — latency is measured upstream."""

    name = "latency"
    required_fields: tuple[str, ...] = ("latency_ms",)

    def __init__(self, unit: MetricUnit = MetricUnit.MS) -> None:
        super().__init__(unit)

    def run(self, record: EvaluationRecord) -> MetricResult:
        return MetricResult(
            metric_name="latency",
            record_id=record.record_id,
            status=MetricStatus.SCORED,
            score=record.latency_ms,
        )
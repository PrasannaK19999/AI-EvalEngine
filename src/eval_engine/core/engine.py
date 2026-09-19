"""The evaluation engine: runs a chosen set of metrics over records and gates them."""

from __future__ import annotations

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricResult,
    MetricStatus,
)
from eval_engine.metrics.base import BaseMetric


class EvaluationEngine:
    """Runs a fixed set of metrics (the 'checked boxes') over evaluation records."""

    def __init__(self, metrics: list[BaseMetric]) -> None:
        self._metrics = metrics

    def evaluate(self, records: list[EvaluationRecord]) -> list[MetricResult]:
        """Evaluate every record against every configured metric. Returns a flat list."""
        results: list[MetricResult] = []
        for record in records:
            results.extend(self._evaluate_record(record))
        return results

    def _evaluate_record(self, record: EvaluationRecord) -> list[MetricResult]:
        """Run all metrics for ONE record, Considering the CLI Phase Failure,
        we are reverting the gate where following the principle of 
        Eval Engine should expose all metrics itself can't fail """
        
        return [self._run_one(metric, record) for metric in self._metrics]

    def _run_one(self, metric: BaseMetric, record: EvaluationRecord) -> MetricResult:
        """Field gate, then run. Missing required field -> SKIPPED, metric never runs."""
        for field in metric.required_fields:
            if getattr(record, field) is None:
                return MetricResult(
                    metric_name=metric.name,
                    record_id=record.record_id,
                    status=MetricStatus.SKIPPED,
                    skip_reason=f"missing required field: {field}",
                )
        return metric.run(record)
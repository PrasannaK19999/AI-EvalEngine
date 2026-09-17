"""The evaluation engine: runs a chosen set of metrics over records and gates them."""

from __future__ import annotations

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricCategory,
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
        """Run all metrics for ONE record, applying the field gate and the tier gate."""
        record_results: list[MetricResult] = []

        # Pass 1: deterministic metrics run first — they are the gates.
        deterministic_failed = False
        for metric in self._metrics:
            if metric.category is not MetricCategory.DETERMINISTIC:
                continue
            result = self._run_one(metric, record)
            record_results.append(result)
            # A deterministic gate that scored 0.0 means "this record is broken."
            if result.status is MetricStatus.SCORED and result.score == 0.0:
                deterministic_failed = True

        # Pass 2: judge metrics — skipped entirely if any gate failed.
        for metric in self._metrics:
            if metric.category is not MetricCategory.JUDGE:
                continue
            if deterministic_failed:
                record_results.append(
                    MetricResult(
                        metric_name=metric.name,
                        record_id=record.record_id,
                        status=MetricStatus.SKIPPED,
                        skip_reason="a deterministic gate failed for this record",
                    )
                )
                continue
            record_results.append(self._run_one(metric, record))

        return record_results

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
# ── src\eval_engine\metrics\deterministic\schema_validity.py ──
"""Schema validity: checks the generated answer parses as JSON and has the expected keys.

Tier-1 gate. A malformed answer scoring 0.0 is a valid FINDING, not a metric error,
so this metric never returns ERRORED — it either skips (no schema) or scores 1.0/0.0.
"""

from __future__ import annotations

import json

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricResult,
    MetricStatus,
)
from eval_engine.metrics.base import BaseMetric


class SchemaValidityMetric(BaseMetric):
    """Score 1.0 if the answer is valid JSON containing every key the schema names, else 0.0."""

    name = "schema_validity"
    required_fields: tuple[str, ...] = ("generated_answer", "schema_definition")

    def run(self, record: EvaluationRecord) -> MetricResult:
        schema = record.schema_definition

        # No schema demanded -> nothing to validate against.
        if schema is None:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.SKIPPED,
                skip_reason="no schema defined",
            )

        # Try to parse the answer as JSON. A parse failure is a real finding: score 0.0.
        try:
            parsed = json.loads(record.generated_answer)
        except json.JSONDecodeError:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.SCORED,
                score=0.0,
            )

        # Must be a JSON object, and must contain every key the schema names.
        if not isinstance(parsed, dict):
            score = 0.0
        else:
            expected_keys = set(schema.keys())
            score = 1.0 if expected_keys.issubset(parsed.keys()) else 0.0

        return MetricResult(
            metric_name=self.name,
            record_id=record.record_id,
            status=MetricStatus.SCORED,
            score=score,
        )
# ── src\eval_engine\metrics\deterministic\cost.py ──
"""Cost metric: computes USD spend for a record from its token usage and a pricing table."""

from __future__ import annotations

import json
from pathlib import Path

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricResult,
    MetricStatus,
    MetricUnit,
)
from eval_engine.metrics.base import BaseMetric


class CostMetric(BaseMetric):
    """USD cost = (prompt_tokens/1M * input_rate) + (completion_tokens/1M * output_rate)."""

    name = "cost"
    required_fields: tuple[str, ...] = ("model_id", "token_usage")

    def __init__(self, pricing_table_path: Path, unit: MetricUnit = MetricUnit.USD) -> None:
        super().__init__(unit)
        with pricing_table_path.open(encoding="utf-8") as f:
            self._pricing: dict[str, dict[str, float]] = json.load(f)

    def run(self, record: EvaluationRecord) -> MetricResult:
        rates = self._pricing.get(record.model_id)

        # Unknown model -> we won't guess a price. Real error, not a $0 lie.
        if rates is None:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.ERRORED,
                error=f"no pricing for model_id '{record.model_id}'",
            )

        usage = record.token_usage
        cost = (
            (usage.prompt_tokens / 1_000_000) * rates["input_per_1m"]
            + (usage.completion_tokens / 1_000_000) * rates["output_per_1m"]
        )

        return MetricResult(
            metric_name=self.name,
            record_id=record.record_id,
            status=MetricStatus.SCORED,
            score=cost,
        )
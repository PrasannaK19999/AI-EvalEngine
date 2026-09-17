"""Citation validity: checks that cited chunk IDs actually exist in the retrieved set.

Deterministic. This is the cheap 'do the citations exist' half of citation checking;
the 'do they support the claim' half is a judge metric built later.
"""

from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord, MetricCategory, MetricResult, MetricStatus
from eval_engine.metrics.base import BaseMetric


class CitationValidityMetric(BaseMetric):
    """Score = (cited IDs that were actually retrieved) / (total cited IDs)."""

    name = "citation_validity"
    category = MetricCategory.DETERMINISTIC
    required_fields: tuple[str, ...] = ("expected_citation_ids", "retrieved_contexts")

    def run(self, record: EvaluationRecord) -> MetricResult:
        cited = record.expected_citation_ids

    # No citations claimed at all -> nothing to validate. Honest skip, not a score.
        if cited is None:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.SKIPPED,
                skip_reason="no citations to validate",
            )

        # Citations field present but empty -> vacuously correct: nothing was cited wrong.
        if len(cited) == 0:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.SCORED,
                score=1.0,
            )

        retrieved_ids = {ctx.id for ctx in record.retrieved_contexts}
        valid = sum(1 for cid in cited if cid in retrieved_ids)
        score = valid / len(cited)

        return MetricResult(
            metric_name=self.name,
            record_id=record.record_id,
            status=MetricStatus.SCORED,
            score=score,
        )

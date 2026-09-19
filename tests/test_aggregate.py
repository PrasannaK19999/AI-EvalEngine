"""Tests for result aggregation — grouping, means, and deltas. Pure math, no LLM."""

from __future__ import annotations

from eval_engine.core.aggregate import aggregate
from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricResult,
    MetricStatus,
    TokenUsage,
)


def _rec(record_id: str, group: str) -> EvaluationRecord:
    return EvaluationRecord(
        record_id=record_id,
        experiment_group=group,
        input="q",
        generated_answer="a",
        model_id="gemini-3.5-flash-lite",
        latency_ms=100.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _scored(record_id: str, metric: str, score: float) -> MetricResult:
    return MetricResult(
        metric_name=metric, record_id=record_id, status=MetricStatus.SCORED, score=score
    )


def _skipped(record_id: str, metric: str) -> MetricResult:
    return MetricResult(
        metric_name=metric, record_id=record_id, status=MetricStatus.SKIPPED,
        skip_reason="x",
    )


def test_mean_averages_scored_values() -> None:
    records = [_rec("r1", "baseline"), _rec("r2", "baseline")]
    results = [_scored("r1", "faithfulness", 1.0), _scored("r2", "faithfulness", 0.0)]
    report = aggregate(results, records)
    summary = report.summaries["baseline"]["faithfulness"]
    assert summary.mean_score == 0.5
    assert summary.scored_count == 2


def test_skipped_excluded_from_mean_but_counted() -> None:
    records = [_rec("r1", "baseline"), _rec("r2", "baseline")]
    results = [_scored("r1", "faithfulness", 1.0), _skipped("r2", "faithfulness")]
    report = aggregate(results, records)
    summary = report.summaries["baseline"]["faithfulness"]
    assert summary.mean_score == 1.0  # only the scored one counts
    assert summary.scored_count == 1
    assert summary.skipped_count == 1


def test_delta_computes_baseline_vs_variant() -> None:
    records = [_rec("r1", "baseline"), _rec("r2", "degraded")]
    results = [_scored("r1", "faithfulness", 1.0), _scored("r2", "faithfulness", 0.0)]
    report = aggregate(results, records)
    delta = report.deltas["degraded"]["faithfulness"]
    assert delta.baseline_mean == 1.0
    assert delta.variant_mean == 0.0
    assert delta.absolute_delta == -1.0
    assert delta.percent_delta == -100.0


def test_delta_guards_divide_by_zero() -> None:
    records = [_rec("r1", "baseline"), _rec("r2", "degraded")]
    results = [_scored("r1", "faithfulness", 0.0), _scored("r2", "faithfulness", 0.5)]
    report = aggregate(results, records)
    delta = report.deltas["degraded"]["faithfulness"]
    assert delta.absolute_delta == 0.5
    assert delta.percent_delta is None  # baseline is 0.0 -> no percent


def test_records_counted_per_group() -> None:
    records = [_rec("r1", "baseline"), _rec("r2", "baseline"), _rec("r3", "degraded")]
    results = [_scored("r1", "latency", 1.0), 
               _scored("r2", "latency", 1.0),
               _scored("r3", "latency", 1.0)]
    
    report = aggregate(results, records)
    assert report.total_records_by_group["baseline"] == 2
    assert report.total_records_by_group["degraded"] == 1
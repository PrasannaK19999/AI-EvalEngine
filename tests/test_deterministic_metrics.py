"""Tests for deterministic metrics — pure logic, no LLM."""

from __future__ import annotations

from pathlib import Path

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricStatus,
    RetrievedContext,
    TokenUsage,
)
from eval_engine.metrics.deterministic.citation_validity import CitationValidityMetric
from eval_engine.metrics.deterministic.cost import CostMetric
from eval_engine.metrics.deterministic.latency import LatencyMetric
from eval_engine.metrics.deterministic.schema_validity import SchemaValidityMetric


def _record(**overrides: object) -> EvaluationRecord:
    """Build a minimal valid record, with fields overridable per test."""
    base: dict[str, object] = {
        "record_id": "t1",
        "input": "q",
        "generated_answer": "a",
        "model_id": "gemini-3.5-flash-lite",
        "latency_ms": 100.0,
        "token_usage": TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    }
    base.update(overrides)
    return EvaluationRecord(**base)  # type: ignore[arg-type]


# --- latency ---

def test_latency_reports_the_value() -> None:
    result = LatencyMetric().run(_record(latency_ms=480.0))
    assert result.status is MetricStatus.SCORED
    assert result.score == 480.0


# --- citation validity ---

def test_citation_validity_all_valid_scores_one() -> None:
    rec = _record(
        retrieved_contexts=[RetrievedContext(id="c1", text="x")],
        expected_citation_ids=["c1"],
    )
    assert CitationValidityMetric().run(rec).score == 1.0


def test_citation_validity_half_valid_scores_half() -> None:
    rec = _record(
        retrieved_contexts=[RetrievedContext(id="c1", text="x")],
        expected_citation_ids=["c1", "c_fake"],
    )
    assert CitationValidityMetric().run(rec).score == 0.5


def test_citation_validity_none_is_skipped() -> None:
    rec = _record(expected_citation_ids=None)
    assert CitationValidityMetric().run(rec).status is MetricStatus.SKIPPED


def test_citation_validity_empty_list_scores_one() -> None:
    rec = _record(expected_citation_ids=[])
    result = CitationValidityMetric().run(rec)
    assert result.status is MetricStatus.SCORED
    assert result.score == 1.0


# --- schema validity ---

def test_schema_validity_valid_json_with_keys_scores_one() -> None:
    rec = _record(generated_answer='{"answer": "12"}', schema_definition={"answer": "str"})
    assert SchemaValidityMetric().run(rec).score == 1.0


def test_schema_validity_malformed_json_scores_zero_not_errored() -> None:
    rec = _record(generated_answer="not json", schema_definition={"answer": "str"})
    result = SchemaValidityMetric().run(rec)
    assert result.status is MetricStatus.SCORED  # a finding, not a crash
    assert result.score == 0.0


def test_schema_validity_no_schema_is_skipped() -> None:
    rec = _record(schema_definition=None)
    assert SchemaValidityMetric().run(rec).status is MetricStatus.SKIPPED


# --- cost ---

def test_cost_known_model_computes_price() -> None:
    rec = _record(
        model_id="gemini-2.5-flash-lite",
        token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    result = CostMetric(Path("data/pricing_table.json")).run(rec)
    assert result.status is MetricStatus.SCORED
    assert result.score is not None
    assert abs(result.score - 0.0003) < 1e-9  # 1000*0.10/1M + 500*0.40/1M


def test_cost_unknown_model_is_errored() -> None:
    rec = _record(model_id="made-up-model")
    result = CostMetric(Path("data/pricing_table.json")).run(rec)
    assert result.status is MetricStatus.ERRORED
    assert result.error is not None
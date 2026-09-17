"""Data Model structs for Evaluation Engine."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

# --- Telemetry & RAG Context Shapes ---

class TokenUsage(BaseModel):
    """Raw token counts returned by the LLM provider for one API call."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class RetrievedContext(BaseModel):
    """A single text chunk fetched from the database for a RAG query."""

    id: str
    text: str
    score: float | None = None


# --- Core Engine Input ---

class EvaluationRecord(BaseModel):
    """Normalized payload representing a single evaluation run."""

    # Identity and experiment tracking
    record_id: str
    experiment_group: str = "baseline"

    # Core LLM interaction data
    input: str
    generated_answer: str
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)

    # Optional human ground truth
    expected_answer: str | None = None
    expected_citation_ids: list[str] | None = None

    # Operational telemetry
    model_id: str
    latency_ms: float = Field(ge=0.0)
    token_usage: TokenUsage

    # Schema gate validation
    schema_definition: dict[str, object] | None = None

    # Metadata escape hatch
    metadata: dict[str, object] = Field(default_factory=dict)


# --- Metric Execution Results ---

class MetricStatus(StrEnum):
    """The outcome status of running an evaluation metric."""

    SCORED = "scored"    # Ran fine, score is valid
    SKIPPED = "skipped"  # Intentionally bypassed (e.g., gate failed, missing fields)
    ERRORED = "errored"  # Failed unexpectedly (e.g., API timeout, parse failure)


class MetricResult(BaseModel):
    """The result of scoring a single metric on a single record."""

    metric_name: str
    record_id: str

    # Result state
    status: MetricStatus
    score: float | None = None

    # Explanations (populated based on status)
    skip_reason: str | None = None
    error: str | None = None

    # Judge metadata
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    reasoning: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> MetricResult:
        if self.status == MetricStatus.SCORED:
            if self.score is None:
                raise ValueError("A SCORED metric must have a numeric score.")
            if self.skip_reason or self.error:
                raise ValueError("A SCORED metric must not carry skip_reason or error.")
        elif self.status == MetricStatus.SKIPPED:
            if not self.skip_reason:
                raise ValueError("A SKIPPED metric must include a skip_reason.")
            if self.score is not None:
                raise ValueError("A SKIPPED metric must not carry a score.")
        elif self.status == MetricStatus.ERRORED:
            if not self.error:
                raise ValueError("An ERRORED metric must include an error message.")
            if self.score is not None:
                raise ValueError("An ERRORED metric must not carry a score.")
        return self

class MetricCategory(StrEnum):
    """Classifies a metric so the engine knows how to gate it."""
    DETERMINISTIC = "deterministic"   # cheap, computed — runs as a gate
    JUDGE = "judge"                   # expensive, model-scored — runs only if gates pass
    
class MetricUnit(StrEnum) :
    """Display label for what a metric's score represents. Never used in math."""
    RATIO = "ratio"
    MS = "ms"
    USD = "USD"

class MetricCalibration(BaseModel):
    """How well the judge agreed with human labels for ONE metric."""
    metric_name: str
    sample_size: int = Field(ge=0)          # how many labeled (record, metric) pairs backed this
    mae: float | None = None                # mean |judge - human|; 0 = perfect. None if no labels
    agreement: float | None = None          # 1 - mae, the friendly 0-1 version. None if no labels
    bias: float | None = None               # signed mean (judge - human); +over-scores, -harsh

    @model_validator(mode="after")
    def check_sample_and_scores(self) -> MetricCalibration:
        if self.sample_size == 0 and self.mae is not None:
            raise ValueError("mae must be None when sample_size is 0.")
        if self.sample_size > 0 and self.mae is None:
            raise ValueError("mae must be set when sample_size > 0.")
        return self


class CalibrationReport(BaseModel):
    """Judge trustworthiness across all calibrated metrics."""
    calibrations: dict[str, MetricCalibration] = Field(default_factory=dict)

# --- Aggregate Reporting ---

class MetricSummary(BaseModel):
    """Aggregated statistics for a single metric across a dataset."""

    metric_name: str
    scored_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    errored_count: int = Field(default=0, ge=0)
    mean_score: float | None = None

    @model_validator(mode="after")
    def validate_counts_and_mean(self) -> MetricSummary:
        if self.scored_count == 0 and self.mean_score is not None:
            raise ValueError(
                f"Metric '{self.metric_name}' has scored_count=0 but mean_score={self.mean_score}. "
                "mean_score must be None when no records were scored."
            )
        if self.scored_count > 0 and self.mean_score is None:
            raise ValueError(
                f"Metric '{self.metric_name}' has scored_count={self.scored_count} "
                "but mean_score is None."
            )
        return self


class MetricDelta(BaseModel):
    """Comparison between a variant group and the baseline for one metric."""

    baseline_mean: float | None = None
    variant_mean: float | None = None
    absolute_delta: float | None = None  # variant - baseline
    percent_delta: float | None = None   # ((variant - baseline) / baseline) * 100

    @model_validator(mode="after")
    def validate_arithmetic(self) -> MetricDelta:
        # If either mean is missing, deltas cannot exist
        if self.baseline_mean is None or self.variant_mean is None:
            if self.absolute_delta is not None or self.percent_delta is not None:
                raise ValueError(
                    "Deltas cannot be populated if baseline_mean or variant_mean is None."
                )
            return self

        # Verify absolute delta arithmetic: variant - baseline
        expected_abs = self.variant_mean - self.baseline_mean
        if self.absolute_delta is None or not math.isclose(
            self.absolute_delta, expected_abs, rel_tol=1e-5, abs_tol=1e-7
        ):
            raise ValueError(
                f"Invalid absolute_delta {self.absolute_delta}; expected {expected_abs}."
            )

        # Verify percent delta arithmetic if baseline is non-zero
        if math.isclose(self.baseline_mean, 0.0, abs_tol=1e-9):
            if self.percent_delta is not None:
                raise ValueError("percent_delta must be None when baseline_mean is 0.0.")
        else:
            expected_pct = (expected_abs / self.baseline_mean) * 100.0
            if self.percent_delta is None or not math.isclose(
                self.percent_delta, expected_pct, rel_tol=1e-4, abs_tol=1e-6
            ):
                raise ValueError(
                    f"Invalid percent_delta {self.percent_delta}; expected {expected_pct}."
                )

        return self


class AggregateReport(BaseModel):
    """The full evaluation run summarized across all experiment groups."""

    # Total sample counts per experiment group (e.g. {"baseline": 50, "degraded": 50})
    total_records_by_group: dict[str, int] = Field(default_factory=dict)

    # summaries[experiment_group][metric_name] -> MetricSummary
    summaries: dict[str, dict[str, MetricSummary]] = Field(default_factory=dict)

    # deltas[variant_group][metric_name] -> MetricDelta
    deltas: dict[str, dict[str, MetricDelta]] = Field(default_factory=dict)
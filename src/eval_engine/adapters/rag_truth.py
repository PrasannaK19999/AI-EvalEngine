# ── src\eval_engine\adapters\ragtruth.py ──
"""Adapter: RAGTruth (Hugging Face) -> EvaluationRecords.

RAGTruth gives question + context + LLM answer + human hallucination annotations.
We map those into records and derive a human faithfulness label from the
annotation (no hallucination -> faithful 1.0; any hallucination -> 0.0), which
gives real, independent calibration labels — unlike a hand-authored golden set.

RAGTruth has no citation IDs or token telemetry, so citation metrics don't apply
and telemetry is placeholder. The value here is the judge metrics on real data.
"""

from __future__ import annotations

from typing import Any

from eval_engine.core.contracts import (
    EvaluationRecord,
    RetrievedContext,
    TokenUsage,
)

# Placeholder telemetry — RAGTruth doesn't record latency/tokens.
_PLACEHOLDER_TOKENS = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def _is_hallucinated(row: dict[str, Any]) -> bool:
    """True if the human annotation marks any hallucination in this response."""
    processed = row.get("hallucination_labels_processed") or {}
    return any(int(v) > 0 for v in processed.values())


def ragtruth_row_to_record(row: dict[str, Any]) -> EvaluationRecord:
    """Map one RAGTruth row into an EvaluationRecord."""
    context_text = str(row.get("context", ""))
    return EvaluationRecord(
        record_id=f"ragtruth_{row['id']}",
        experiment_group="ragtruth",
        input=str(row.get("query", "")),
        generated_answer=str(row.get("output", "")),
        retrieved_contexts=[RetrievedContext(id=f"ctx_{row['id']}", text=context_text)],
        model_id="gemini-3.5-flash-lite",  # the JUDGE model; RAGTruth's own model is in metadata
        latency_ms=0.0,
        token_usage=_PLACEHOLDER_TOKENS,
        metadata={
            "task_type": str(row.get("task_type", "")),
            "source_model": str(row.get("model", "")),
            "quality": str(row.get("quality", "")),
        },
    )


def human_faithfulness_label(row: dict[str, Any]) -> float:
    """Derive the human faithfulness label from the hallucination annotation.

    No hallucination -> faithful (1.0). Any hallucination -> unfaithful (0.0).
    """
    return 0.0 if _is_hallucinated(row) else 1.0
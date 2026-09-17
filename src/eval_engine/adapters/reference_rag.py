# ── src\eval_engine\adapters\reference_rag.py ──
"""Reference RAG fixture: builds EvaluationRecords from hand-authored cases.

This is NOT a real RAG system — it's a test fixture that produces records
(including deliberately degraded ones) so the engine has realistic data to grade
before any live LLM/adapter work exists.
"""

from __future__ import annotations

from eval_engine.core.contracts import (
    EvaluationRecord,
    RetrievedContext,
    TokenUsage,
)


def make_record(
    record_id: str,
    question: str,
    answer: str,
    contexts: list[tuple[str, str]],
    *,
    experiment_group: str = "baseline",
    expected_answer: str | None = None,
    expected_citation_ids: list[str] | None = None,
    model_id: str = "gemini-2.5-flash-lite",
    latency_ms: float = 400.0,
    prompt_tokens: int = 500,
    completion_tokens: int = 50,
) -> EvaluationRecord:
    """Build one EvaluationRecord from simple inputs.

    `contexts` is a list of (id, text) pairs — the retrieved chunks.
    """
    retrieved = [RetrievedContext(id=cid, text=text) for cid, text in contexts]
    return EvaluationRecord(
        record_id=record_id,
        experiment_group=experiment_group,
        input=question,
        generated_answer=answer,
        retrieved_contexts=retrieved,
        expected_answer=expected_answer,
        expected_citation_ids=expected_citation_ids,
        model_id=model_id,
        latency_ms=latency_ms,
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
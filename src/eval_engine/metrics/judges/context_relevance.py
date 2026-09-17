# ── src\eval_engine\metrics\judges\context_relevance.py ──
"""Context relevance judge: are the retrieved chunks relevant to the question?"""

from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric

PROMPT_TEMPLATE = """You are evaluating CONTEXT RELEVANCE.

Given the QUESTION and the retrieved CONTEXT, decide whether the CONTEXT is \
relevant and useful for answering the QUESTION. Judge the context itself, not any answer.

QUESTION:
{question}

CONTEXT:
{context}

Respond with ONLY a JSON object, no other text:
{{"score": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}
1.0 = context is highly relevant to the question. 0.0 = context is unrelated."""


class ContextRelevanceMetric(JudgeMetric):
    """Scores whether retrieval fetched context relevant to the question (0.0-1.0)."""

    name = "context_relevance"
    required_fields: tuple[str, ...] = ("input", "retrieved_contexts")

    def build_prompt(self, record: EvaluationRecord) -> str:
        context = "\n".join(c.text for c in record.retrieved_contexts)
        return PROMPT_TEMPLATE.format(question=record.input, context=context)
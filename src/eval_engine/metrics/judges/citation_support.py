# ── src\eval_engine\metrics\judges\citation_support.py ──
"""Citation support judge: do the retrieved chunks actually support the answer's claims """

from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric

PROMPT_TEMPLATE = """You are evaluating CITATION SUPPORT.

Given the ANSWER and the CONTEXT it cites, decide whether the claims in the ANSWER \
are actually supported by the CONTEXT. Judge whether the cited material genuinely \
backs what the answer asserts.

ANSWER:
{answer}

CONTEXT:
{context}

Respond with ONLY a JSON object, no other text:
{{"score": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}
1.0 = claims fully backed by the context. 0.0 = claims not supported by it."""


class CitationSupportMetric(JudgeMetric):
    """Scores whether the cited context genuinely supports the answer's claims (0.0-1.0)."""

    name = "citation_support"
    required_fields: tuple[str, ...] = ("generated_answer", "retrieved_contexts")

    def build_prompt(self, record: EvaluationRecord) -> str:
        context = "\n".join(c.text for c in record.retrieved_contexts)
        return PROMPT_TEMPLATE.format(answer=record.generated_answer, context=context)
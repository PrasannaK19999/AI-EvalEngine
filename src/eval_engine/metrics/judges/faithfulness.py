"""Faithfulness judge: asks the LLM whether the answer is supported by the retrieved context."""

from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric

PROMPT_VERSION = "v1"

PROMPT_TEMPLATE = """You are evaluating FAITHFULNESS.

Given the CONTEXT and the ANSWER, decide whether every claim in the ANSWER is \
supported by the CONTEXT. An answer is faithful only if it makes no claim that \
the context does not support.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object, no other text:
{{"score": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}
1.0 = every claim fully supported. 0.0 = unsupported or contradicted."""


class FaithfulnessMetric(JudgeMetric):
    """Scores how well the answer is grounded in the retrieved context (0.0-1.0)."""

    name = "faithfulness"
    required_fields: tuple[str, ...] = ("generated_answer", "retrieved_contexts")

    def build_prompt(self, record: EvaluationRecord) -> str:
        context = "\n".join(c.text for c in record.retrieved_contexts)
        return PROMPT_TEMPLATE.format(context=context, answer=record.generated_answer)

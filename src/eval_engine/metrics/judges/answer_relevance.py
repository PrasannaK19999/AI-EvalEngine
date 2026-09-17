"""Answer relevance judge: asks the LLM whether the answer addresses the question."""

from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric

PROMPT_VERSION = "v1"

PROMPT_TEMPLATE = """You are evaluating ANSWER RELEVANCE.

Given the QUESTION and the ANSWER, decide whether the ANSWER actually addresses \
what the QUESTION asked. Judge only relevance and responsiveness — not whether the \
answer is factually correct.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object, no other text:
{{"score": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}
1.0 = directly and fully addresses the question. 0.0 = ignores it or answers something else."""


class AnswerRelevanceMetric(JudgeMetric):
    """Scores how well the answer addresses the question asked (0.0-1.0)."""

    name = "answer_relevance"
    required_fields: tuple[str, ...] = ("input", "generated_answer")

    def build_prompt(self, record: EvaluationRecord) -> str:
        return PROMPT_TEMPLATE.format(question=record.input, answer=record.generated_answer)
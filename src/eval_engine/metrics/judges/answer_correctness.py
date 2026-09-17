# ── src\eval_engine\metrics\judges\answer_correctness.py ──
"""Answer correctness judge: does the answer match the known-correct expected answer?"""

from __future__ import annotations

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric

PROMPT_TEMPLATE = """You are evaluating ANSWER CORRECTNESS.

Given the QUESTION, the EXPECTED (correct) answer, and the GENERATED answer, \
decide whether the GENERATED answer is factually consistent with the EXPECTED answer. \
Wording may differ; judge meaning, not phrasing.

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

GENERATED ANSWER:
{answer}

Respond with ONLY a JSON object, no other text:
{{"score": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}
1.0 = fully consistent with the expected answer. 0.0 = contradicts or misses it."""


class AnswerCorrectnessMetric(JudgeMetric):
    """Scores the answer against the known-correct expected answer (0.0-1.0)."""

    name = "answer_correctness"
    required_fields: tuple[str, ...] = ("input", "generated_answer", "expected_answer")

    def build_prompt(self, record: EvaluationRecord) -> str:
        return PROMPT_TEMPLATE.format(
            question=record.input,
            expected=record.expected_answer,
            answer=record.generated_answer,
        )
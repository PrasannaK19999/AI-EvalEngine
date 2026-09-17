"""Faithfulness judge: asks the LLM whether the answer is supported by the retrieved context."""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from eval_engine.core.cache import JudgeCache
from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricCategory,
    MetricResult,
    MetricStatus,
)
from eval_engine.core.gemini_client import GeminiJudgeClient
from eval_engine.metrics.base import BaseMetric

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


class GraderVerdict(BaseModel):
    """The shape we expect the judge to return."""
    score: float
    reasoning: str


class FaithfulnessMetric(BaseMetric):
    """Scores how well the answer is grounded in the retrieved context (0.0-1.0)."""

    name = "faithfulness"
    category = MetricCategory.JUDGE
    required_fields: tuple[str, ...] = ("generated_answer", "retrieved_contexts")

    def __init__(self, client: GeminiJudgeClient, cache: JudgeCache) -> None:
        super().__init__()
        self._client = client
        self._cache = cache

    def run(self, record: EvaluationRecord) -> MetricResult:
        context = "\n".join(c.text for c in record.retrieved_contexts)
        prompt = PROMPT_TEMPLATE.format(context=context, answer=record.generated_answer)

        # Cache first — never pay twice for the same judging input.
        key = self._cache.make_key(prompt, self._client._model, PROMPT_VERSION)
        raw = self._cache.get(key)
        if raw is None:
            raw = self._client.complete(prompt)
            self._cache.set(key, raw)

        # Defensive parse — the model may wrap JSON in fences or add stray text.
        verdict = self._parse(raw)
        if verdict is None:
            return MetricResult(
                metric_name=self.name,
                record_id=record.record_id,
                status=MetricStatus.ERRORED,
                error="could not parse judge response as JSON verdict",
            )

        return MetricResult(
            metric_name=self.name,
            record_id=record.record_id,
            status=MetricStatus.SCORED,
            score=verdict.score,
            judge_model=self._client._model,
            judge_prompt_version=PROMPT_VERSION,
            reasoning=verdict.reasoning,
        )

    @staticmethod
    def _parse(raw: str) -> GraderVerdict | None:
        """Strip common fences, parse JSON, validate into a verdict. None on any failure."""
        cleaned = raw.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
        try:
            data = json.loads(cleaned)
            return GraderVerdict.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            return None

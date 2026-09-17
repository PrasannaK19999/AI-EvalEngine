""" Shared base for all judge metrics: owns the cache-call-parse-provenance flow """

from __future__ import annotations

import json
from abc import abstractmethod

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


class GraderVerdict(BaseModel):
    """The shape every judge expects back from the LLM."""
    score: float
    reasoning: str


class JudgeMetric(BaseMetric):
    """Base for LLM-judge metrics. Subclasses implement build_prompt() only."""

    category = MetricCategory.JUDGE
    prompt_version: str = "v1"

    def __init__(self, client: GeminiJudgeClient, cache: JudgeCache) -> None:
        super().__init__()
        self._client = client
        self._cache = cache

    @abstractmethod
    def build_prompt(self, record: EvaluationRecord) -> str:
        """Build the judge prompt for this record. The one thing each judge differs on."""
        ...

    def run(self, record: EvaluationRecord) -> MetricResult:
        prompt = self.build_prompt(record)

        # Cache-first: never pay twice for an identical judging input.
        key = self._cache.make_key(prompt, self._client.model, self.prompt_version)
        raw = self._cache.get(key)
        if raw is None:
            raw = self._client.complete(prompt)
            self._cache.set(key, raw)

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
            judge_model=self._client.model,
            judge_prompt_version=self.prompt_version,
            reasoning=verdict.reasoning,
        )

    @staticmethod
    def _parse(raw: str) -> GraderVerdict | None:
        """Strip common fences, parse JSON, validate. None on any failure."""
        cleaned = raw.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
        try:
            data = json.loads(cleaned)
            return GraderVerdict.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            return None
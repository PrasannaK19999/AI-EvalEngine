"""Abstract base class for all evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from eval_engine.core.contracts import EvaluationRecord, MetricCategory, MetricResult, MetricUnit


class BaseMetric(ABC):
    """Abstract interface that every evaluation metric must implement."""

    name: str
    category: MetricCategory
    required_fields: tuple[str, ...] = ()
    

    def __init__(self, unit: MetricUnit = MetricUnit.RATIO) -> None:
        self.unit = unit


    @abstractmethod
    def run(self, record: EvaluationRecord) -> MetricResult:
        """Evaluate a single record and return its metric result.

        Assumes required_fields have already been validated by the engine.
        """
        raise NotImplementedError
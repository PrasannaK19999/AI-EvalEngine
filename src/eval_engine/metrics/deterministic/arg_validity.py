"""Arg-validity metric: did each tool call's arguments match the tool's schema?"""

from __future__ import annotations

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from eval_engine.core.contracts import (
    EvaluationRecord,
    MetricCategory,
    MetricResult,
    MetricStatus,
)
from eval_engine.core.trajectory import TrajectoryRecord
from eval_engine.metrics.base import BaseMetric


class ArgValidityMetric(BaseMetric):
    """Validates each step's `args` against its tool's JSON Schema."""

    name = "arg_validity"
    category = MetricCategory.DETERMINISTIC
    required_fields = ("trajectory",)

    def __init__(self, tool_schemas: dict[str, dict[str, object]]) -> None:
        super().__init__()
        # Pre-compile validators once; surface a broken schema file loudly, early.
        self._validators: dict[str, Draft202012Validator] = {}
        for tool, schema in tool_schemas.items():
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as e:
                raise ValueError(f"invalid JSON Schema for tool '{tool}': {e.message}") from e
            self._validators[tool] = Draft202012Validator(schema)

    def run(self, record: EvaluationRecord) -> MetricResult:
        if not isinstance(record, TrajectoryRecord):
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="record has no trajectory",
            )

        steps = record.trajectory.steps
        if not steps:
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED, skip_reason="no tool calls to validate",
            )

        judged = 0
        valid = 0
        failures: list[str] = []
        unschemad: list[str] = []

        for i, step in enumerate(steps):
            validator = self._validators.get(step.tool)
            if validator is None:
                unschemad.append(f"[{i}]{step.tool}")
                continue
            judged += 1
            errors = sorted(validator.iter_errors(step.args), key=lambda e: e.path)
            if errors:
                failures.append(f"[{i}]{step.tool}: {errors[0].message}")
            else:
                valid += 1

        if judged == 0:
            return MetricResult(
                metric_name=self.name, record_id=record.record_id,
                status=MetricStatus.SKIPPED,
                skip_reason=f"no tool calls had a matching schema (unschemad={unschemad})",
            )

        score = valid / judged
        reasoning = (
            f"valid={valid}/{judged} "
            f"failures={failures} unschemad={unschemad}"
        )
        return MetricResult(
            metric_name=self.name, record_id=record.record_id,
            status=MetricStatus.SCORED, score=score, reasoning=reasoning,
        )
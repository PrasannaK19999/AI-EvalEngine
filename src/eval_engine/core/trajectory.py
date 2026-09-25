from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .contracts import EvaluationRecord


class TrajectoryStep(BaseModel):
    model_config = ConfigDict() 
    tool: str
    args: dict[str, Any] = {}
    result: Any | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> TrajectoryStep:
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result and has_error:
            raise ValueError("step cannot carry both a result and an error")
        if not has_result and not has_error:
            raise ValueError("step must carry a result or an error, not neither")
        return self


class Trajectory(BaseModel):
    model_config = ConfigDict()

    steps: list[TrajectoryStep] = []


class TrajectoryRecord(EvaluationRecord):
    trajectory: Trajectory
    expected_tools: list[str] | None = None
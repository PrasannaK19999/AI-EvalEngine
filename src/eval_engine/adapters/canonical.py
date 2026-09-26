"""Canonical agent-trace adapter: flattens the canonical trace JSON into records.

The canonical format is the engine's stable input contract. Other source formats
(OpenAI-style, Gemini-native, tau-bench) are separate adapters that map their
shape into records the same way — this is the reference implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_engine.core.contracts import TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep


def _build_step(raw: dict[str, Any]) -> TrajectoryStep:
    """Map one canonical step into a TrajectoryStep (result XOR error enforced by the model)."""
    return TrajectoryStep(
        tool=raw["tool"],
        args=raw.get("args", {}),
        result=raw.get("result"),
        error=raw.get("error"),
    )


def _build_record(raw: dict[str, Any]) -> TrajectoryRecord:
    """Flatten one canonical trace into a TrajectoryRecord.

    Operational telemetry (model_id/latency/tokens) is optional in a trace that
    focuses on agent behavior; absent values default so behavior metrics run and
    cost/latency metrics simply score zero rather than the load failing.
    """
    steps = [_build_step(s) for s in raw["steps"]]
    return TrajectoryRecord(
        record_id=raw["record_id"],
        input=raw["task"],
        generated_answer=raw["final_answer"],
        expected_tools=raw.get("expected_tools"),
        model_id=raw.get("model_id", "unknown"),
        latency_ms=raw.get("latency_ms", 0.0),
        token_usage=TokenUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        ),
        trajectory=Trajectory(steps=steps),
    )


def load_canonical_traces(path: Path) -> list[TrajectoryRecord]:
    """Read a JSON array of canonical traces and flatten each into a TrajectoryRecord."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"canonical trace file must be a JSON array, got {type(raw).__name__}")
    return [_build_record(item) for item in raw]
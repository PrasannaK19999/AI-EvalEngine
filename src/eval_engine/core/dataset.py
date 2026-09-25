"""Load a golden-set JSON file into EvaluationRecords."""

from __future__ import annotations

import json
from pathlib import Path

from eval_engine.core.contracts import EvaluationRecord


def load_golden_set(path: Path) -> list[EvaluationRecord]:
    """Read a JSON array of records and validate each into an EvaluationRecord."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [EvaluationRecord.model_validate(item) for item in raw]

def load_tool_schemas(path: Path) -> dict[str, dict[str, object]]:
    """Load tool-name -> JSON-Schema mapping for arg-validity.
    Raises on malformed JSON at load time (fail loud, before evaluation runs).
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"tool schemas file must be a JSON object, got {type(data).__name__}")
    return data
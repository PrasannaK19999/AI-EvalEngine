from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_engine.adapters.canonical import load_canonical_traces
from eval_engine.core.trajectory import TrajectoryRecord


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "traces.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loads_canonical_trace(tmp_path: Path) -> None:
    path = _write(tmp_path, [{
        "record_id": "t1",
        "task": "find X",
        "steps": [{"tool": "search", "args": {"q": "x"}, "result": "found"}],
        "final_answer": "X is found",
        "expected_tools": ["search"],
    }])
    records = load_canonical_traces(path)
    assert len(records) == 1
    r = records[0]
    assert isinstance(r, TrajectoryRecord)
    assert r.record_id == "t1"
    assert r.input == "find X"
    assert r.generated_answer == "X is found"
    assert r.expected_tools == ["search"]
    assert r.trajectory.steps[0].tool == "search"
    assert r.trajectory.steps[0].result == "found"


def test_step_with_error_maps(tmp_path: Path) -> None:
    path = _write(tmp_path, [{
        "record_id": "t2",
        "task": "fetch",
        "steps": [{"tool": "fetch", "args": {}, "error": "timeout"}],
        "final_answer": "failed",
    }])
    r = load_canonical_traces(path)[0]
    assert r.trajectory.steps[0].error == "timeout"
    assert r.trajectory.steps[0].result is None


def test_optional_telemetry_defaults(tmp_path: Path) -> None:
    path = _write(tmp_path, [{
        "record_id": "t3",
        "task": "x",
        "steps": [{"tool": "a", "args": {}, "result": "ok"}],
        "final_answer": "done",
    }])
    r = load_canonical_traces(path)[0]
    assert r.model_id == "unknown"
    assert r.latency_ms == 0.0


def test_missing_required_task_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [{
        "record_id": "t4",
        "steps": [{"tool": "a", "args": {}, "result": "ok"}],
        "final_answer": "done",
    }])
    with pytest.raises(KeyError):
        load_canonical_traces(path)


def test_non_array_top_level_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, {"not": "an array"})
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_canonical_traces(path)
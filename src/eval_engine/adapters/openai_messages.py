"""OpenAI-messages agent-trace adapter.

Reads traces in the OpenAI Chat Completions format (also used by tau-bench and
most function-calling agents) and flattens them into TrajectoryRecords. The
core job is stitching: a tool CALL and its RESULT live in separate messages,
linked by tool_call_id, so they are matched back into one step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_engine.core.contracts import TokenUsage
from eval_engine.core.trajectory import Trajectory, TrajectoryRecord, TrajectoryStep


def _parse_args(raw_args: Any) -> dict[str, Any]:
    """OpenAI stores tool-call arguments as a JSON STRING; parse it to a dict."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw_args}
    return {}


def _build_record(convo: dict[str, Any]) -> TrajectoryRecord:
    """Flatten one OpenAI-format conversation into a TrajectoryRecord."""
    messages: list[dict[str, Any]] = convo["messages"]

    # 1. Task = first user message's content.
    task = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )

    # 2. Index tool results by the call id they answer: tool_call_id -> content.
    results_by_id: dict[str, str] = {
        m["tool_call_id"]: str(m.get("content", ""))
        for m in messages
        if m.get("role") == "tool" and "tool_call_id" in m
    }

    # 3. Walk assistant tool_calls, stitch each to its result by id.
    steps: list[TrajectoryStep] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            call_id = call.get("id")
            if call_id in results_by_id:
                steps.append(TrajectoryStep(
                    tool=fn.get("name", "unknown"),
                    args=_parse_args(fn.get("arguments")),
                    result=results_by_id[call_id],
                ))
            else:
                # A call with no matching result message = a step that never
                # returned; record it as an error so the XOR rule stays clean.
                steps.append(TrajectoryStep(
                    tool=fn.get("name", "unknown"),
                    args=_parse_args(fn.get("arguments")),
                    error="no tool result in trace",
                ))

    # 4. Final answer = last assistant message with content but no tool_calls.
    final_answer = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
            final_answer = str(m["content"])
            break

    return TrajectoryRecord(
        record_id=str(convo.get("id", convo.get("record_id", "openai-trace"))),
        input=str(task),
        generated_answer=final_answer,
        expected_tools=convo.get("expected_tools"),
        model_id=str(convo.get("model", "unknown")),
        latency_ms=0.0,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        trajectory=Trajectory(steps=steps),
    )


def load_openai_traces(path: Path) -> list[TrajectoryRecord]:
    """Read a JSON array of OpenAI-format conversations into TrajectoryRecords."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"OpenAI trace file must be a JSON array, got {type(raw).__name__}")
    return [_build_record(item) for item in raw]
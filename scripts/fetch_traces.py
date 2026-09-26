"""One-off: fetch real, COMPLETE agent trajectories from HuggingFace and write
them in the engine's OpenAI-trace format. Run locally (needs network + datasets).

A dataset row is only kept if its conversation actually finishes — the last
message is an assistant reply with content and no pending tool call. This
excludes the SFT training *prefixes* (truncated mid-conversation) that made
task_completion read artificially low.

Usage:  python scripts/fetch_traces.py
Output: data/qwen_traces.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET = "zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory"
OUT = Path("data/qwen_traces.json")
N_PASS = 5
N_FAIL = 5


def _is_complete(messages: list[dict[str, Any]]) -> bool:
    """True only if the conversation ends with a finished assistant answer:
    last message is role=assistant, has text content, and makes no tool call.
    Truncated SFT prefixes end on a tool call or a tool result, so they fail this.
    """
    if not messages:
        return False
    last = messages[-1]
    return (
        last.get("role") == "assistant"
        and bool(last.get("content"))
        and not last.get("tool_calls")
    )


def _to_engine_trace(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "id": f"qwen-{idx}",
        "model": "qwen-3.6-plus",
        "messages": row["messages"],
        "_benchmark_reward": row.get("reward"),
        "_benchmark_score": row.get("score"),
    }


def main() -> None:
    ds = load_dataset(DATASET, split="train")

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped_incomplete = 0
    for row in ds:
        if not _is_complete(row.get("messages", [])):
            skipped_incomplete += 1
            continue  # drop truncated prefixes
        reward = row.get("reward")
        if reward == 1 and len(passed) < N_PASS:
            passed.append(row)
        elif reward == 0 and len(failed) < N_FAIL:
            failed.append(row)
        if len(passed) >= N_PASS and len(failed) >= N_FAIL:
            break

    selected = passed + failed
    traces = [_to_engine_trace(r, i) for i, r in enumerate(selected)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(traces)} COMPLETE traces "
        f"({len(passed)} pass, {len(failed)} fail); "
        f"skipped {skipped_incomplete} truncated prefixes"
    )


if __name__ == "__main__":
    main()
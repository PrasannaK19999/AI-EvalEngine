"""Shared helper: render a trajectory into compact text for judge prompts."""

from __future__ import annotations

from eval_engine.core.trajectory import Trajectory


def render_trajectory(trajectory: Trajectory) -> str:
    """ Small mini version of each steps done by agent. """

    if not trajectory.steps:
        return "(no steps taken)"
    lines: list[str] = []
    for i, step in enumerate(trajectory.steps):
        outcome = f"error: {step.error}" if step.error is not None else f"result: {step.result}"
        lines.append(f"{i + 1}. {step.tool}(args={step.args}) -> {outcome}")
    return "\n".join(lines)
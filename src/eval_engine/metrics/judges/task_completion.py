"""Task-completion judge: did the agent's actual steps accomplish the task?"""

from __future__ import annotations

from typing import cast

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.core.trajectory import TrajectoryRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric
from eval_engine.metrics.judges.trajectory_render import render_trajectory

_TEMPLATE = """You are evaluating whether a task was successfully completed.

You are given a TASK, a log of ACTIONS that were taken to carry it out (each \
action is a tool call with its arguments and the result or error it produced), \
and the FINAL RESPONSE that was returned.

Judge only whether the task was actually accomplished by the actions and their \
results — not by how the actions were chosen, and not by whether the final \
response sounds confident. If the final response claims success but the actions \
did not actually do the required work, the task is NOT complete. If some actions \
failed partway but later actions recovered and finished the work, the task IS \
complete.

TASK:
{task}

ACTIONS TAKEN (tool call -> result or error):
{trajectory}

FINAL RESPONSE:
{final_answer}

Break the task into the concrete things it required, then judge how many were \
actually accomplished by the actions. Score on this scale:
- 1.0 = every required part was accomplished
- around 0.5 = some required parts were done, others were missing or failed
- 0.0 = nothing meaningful was accomplished, or success was only claimed

Respond with a JSON object: {{"score": <0.0-1.0>, "reasoning": "<one sentence \
naming which parts were done and which were missing>"}}"""


class TaskCompletionMetric(JudgeMetric):
    """Reference-free: judges whether the recorded steps accomplished the task."""

    name = "task_completion"
    required_fields = ("trajectory",)
    prompt_version = "task_completion_v1"

    def build_prompt(self, record: EvaluationRecord) -> str:
        # Gate guarantees a TrajectoryRecord (required_fields=("trajectory",)).
        rec = cast(TrajectoryRecord, record)
        return _TEMPLATE.format(
            task=rec.input,
            trajectory=render_trajectory(rec.trajectory),
            final_answer=rec.generated_answer,
        )
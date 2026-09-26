"""Step-necessity judge: what fraction of the steps taken were actually needed?"""

from __future__ import annotations

from typing import cast

from eval_engine.core.contracts import EvaluationRecord
from eval_engine.core.trajectory import TrajectoryRecord
from eval_engine.metrics.judges.base_judge import JudgeMetric
from eval_engine.metrics.judges.trajectory_render import render_trajectory

_TEMPLATE = """You are evaluating how many of the actions taken to carry out a \
task were actually NECESSARY.

You are given a TASK and a log of ACTIONS taken to carry it out (each action is \
a tool call with its arguments and the result or error it produced).

For each action, judge whether it was necessary GIVEN WHAT WAS ALREADY KNOWN at \
that point. An action is UNNECESSARY if the information it produced was already \
available from an earlier action, if it re-established something already known, \
if it ignored a result already obtained, or if it pursued something the task did \
not require. Judge necessity only — do NOT judge whether the task ultimately \
succeeded, and do NOT simply count repeated calls.

Example: if the task needed a value and an early action already produced it, any \
later actions that re-derive or re-confirm that same value are UNNECESSARY, even \
if each does so differently.

TASK:
{task}

ACTIONS TAKEN (tool call -> result or error):
{trajectory}

Judge how many of the actions were necessary. Score = (necessary actions) / \
(total actions), from 0.0 to 1.0:
- 1.0 = every action was necessary
- around 0.5 = about half the actions were unnecessary
- 0.0 = almost none of the actions were needed

Respond with a JSON object: {{"score": <0.0-1.0>, "reasoning": "<one sentence \
naming which actions were unnecessary and why>"}}"""


class StepNecessityMetric(JudgeMetric):
    """Judge will detect where the steps taken by agent is really worthy to give the final result"""

    name = "step_necessity"
    required_fields = ("trajectory",)
    prompt_version = "step_necessity_v1"

    def build_prompt(self, record: EvaluationRecord) -> str:
        # Gate guarantees a TrajectoryRecord (required_fields=("trajectory",)).
        rec = cast(TrajectoryRecord, record)
        return _TEMPLATE.format(
            task=rec.input,
            trajectory=render_trajectory(rec.trajectory),
        )
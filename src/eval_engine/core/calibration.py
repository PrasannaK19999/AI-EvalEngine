"""Load human labels and calibrate judge scores against them."""

from __future__ import annotations

import json
from pathlib import Path

from eval_engine.core.contracts import (
    CalibrationReport,
    MetricCalibration,
    MetricResult,
    MetricStatus,
)

# human_labels shape: {record_id: {metric_name: human_score}}
HumanLabels = dict[str, dict[str, float]]


def load_human_labels(path: Path) -> HumanLabels:
    """Read the (partial) human-label file. Records/metrics may be missing — that's fine."""
    with path.open(encoding="utf-8") as f:
        data: HumanLabels = json.load(f)
    return data


def calibrate(results: list[MetricResult], labels: HumanLabels) -> CalibrationReport:
    """Compare judge scores to human labels, per metric, over only the labeled pairs."""
    # Collect (judge_score, human_score) pairs per metric — only where a label exists
    # and the judge actually produced a score.
    pairs: dict[str, list[tuple[float, float]]] = {}

    for r in results:
        if r.status is not MetricStatus.SCORED or r.score is None:
            continue
        human_for_record = labels.get(r.record_id)
        if human_for_record is None:
            continue
        human_score = human_for_record.get(r.metric_name)
        if human_score is None:
            continue
        pairs.setdefault(r.metric_name, []).append((r.score, human_score))

    report = CalibrationReport()
    for metric_name, score_pairs in pairs.items():
        n = len(score_pairs)
        abs_errors = [abs(judge - human) for judge, human in score_pairs]
        signed_errors = [judge - human for judge, human in score_pairs]
        mae = sum(abs_errors) / n
        bias = sum(signed_errors) / n
        report.calibrations[metric_name] = MetricCalibration(
            metric_name=metric_name,
            sample_size=n,
            mae=mae,
            agreement=1.0 - mae,
            bias=bias,
        )
    return report
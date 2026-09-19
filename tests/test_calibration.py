"""Tests for judge calibration math — MAE, agreement, bias. Pure math, no LLM."""

from __future__ import annotations

from eval_engine.core.calibration import calibrate
from eval_engine.core.contracts import MetricResult, MetricStatus


def _judge(record_id: str, metric: str, score: float) -> MetricResult:
    return MetricResult(
        metric_name=metric, record_id=record_id, status=MetricStatus.SCORED, score=score
    )


def test_perfect_agreement() -> None:
    results = [_judge("r1", "faithfulness", 1.0), _judge("r2", "faithfulness", 0.0)]
    labels = {"r1": {"faithfulness": 1.0}, "r2": {"faithfulness": 0.0}}
    cal = calibrate(results, labels).calibrations["faithfulness"]
    assert cal.sample_size == 2
    assert cal.mae == 0.0
    assert cal.agreement == 1.0
    assert cal.bias == 0.0


def test_mae_and_bias_when_judge_over_scores() -> None:
    # judge says 0.8, human says 0.6 -> off by 0.2, and biased HIGH (+0.2)
    results = [_judge("r1", "faithfulness", 0.8)]
    labels = {"r1": {"faithfulness": 0.6}}
    cal = calibrate(results, labels).calibrations["faithfulness"]
    assert abs(cal.mae - 0.2) < 1e-9
    assert cal.bias is not None and abs(cal.bias - 0.2) < 1e-9  # positive = over-scores


def test_bias_negative_when_judge_too_harsh() -> None:
    # judge 0.3, human 0.7 -> biased LOW (-0.4)
    results = [_judge("r1", "faithfulness", 0.3)]
    labels = {"r1": {"faithfulness": 0.7}}
    cal = calibrate(results, labels).calibrations["faithfulness"]
    assert cal.bias is not None and abs(cal.bias - (-0.4)) < 1e-9


def test_only_labeled_pairs_are_compared() -> None:
    # two results, but only one has a human label -> sample_size 1
    results = [_judge("r1", "faithfulness", 1.0), _judge("r2", "faithfulness", 0.0)]
    labels = {"r1": {"faithfulness": 1.0}}  # r2 unlabeled
    cal = calibrate(results, labels).calibrations["faithfulness"]
    assert cal.sample_size == 1


def test_unlabeled_metric_absent_from_report() -> None:
    results = [_judge("r1", "faithfulness", 1.0)]
    labels: dict[str, dict[str, float]] = {}  # no labels at all
    report = calibrate(results, labels)
    assert "faithfulness" not in report.calibrations
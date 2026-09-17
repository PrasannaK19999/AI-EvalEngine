"""Roll a flat list of MetricResults into an AggregateReport."""

from __future__ import annotations

from eval_engine.core.contracts import (
    AggregateReport,
    EvaluationRecord,
    MetricDelta,
    MetricResult,
    MetricStatus,
    MetricSummary,
)

BASELINE_GROUP = "baseline"


def aggregate(
    results: list[MetricResult],
    records: list[EvaluationRecord],
) -> AggregateReport:
    """Summarize results per (group, metric) and compute deltas vs the baseline group."""

    group_of = {r.record_id: r.experiment_group for r in records}

    scored_values: dict[str, dict[str, list[float]]] = {}
    skipped: dict[str, dict[str, int]] = {}
    errored: dict[str, dict[str, int]] = {}
    group_records: dict[str, set[str]] = {}

    for res in results:
        group = group_of.get(res.record_id)
        if group is None:
            continue  # a result with no matching record — ignore defensively
        group_records.setdefault(group, set()).add(res.record_id)
        m = res.metric_name

        if res.status is MetricStatus.SCORED and res.score is not None:
            scored_values.setdefault(group, {}).setdefault(m, []).append(res.score)
        elif res.status is MetricStatus.SKIPPED:
            skipped.setdefault(group, {})[m] = skipped.setdefault(group, {}).get(m, 0) + 1
        elif res.status is MetricStatus.ERRORED:
            errored.setdefault(group, {})[m] = errored.setdefault(group, {}).get(m, 0) + 1

    report = AggregateReport()

    all_groups = set(scored_values) | set(skipped) | set(errored)
    for group in all_groups:
        report.total_records_by_group[group] = len(group_records.get(group, set()))
        metrics_in_group = (
            set(scored_values.get(group, {}))
            | set(skipped.get(group, {}))
            | set(errored.get(group, {}))
        )
        for metric in metrics_in_group:
            values = scored_values.get(group, {}).get(metric, [])
            n_scored = len(values)
            mean = sum(values) / n_scored if n_scored > 0 else None
            report.summaries.setdefault(group, {})[metric] = MetricSummary(
                metric_name=metric,
                scored_count=n_scored,
                skipped_count=skipped.get(group, {}).get(metric, 0),
                errored_count=errored.get(group, {}).get(metric, 0),
                mean_score=mean,
            )

    baseline_summaries = report.summaries.get(BASELINE_GROUP, {})
    for group, summaries in report.summaries.items():
        if group == BASELINE_GROUP:
            continue
        for metric, variant_summary in summaries.items():
            baseline_summary = baseline_summaries.get(metric)
            if baseline_summary is None:
                continue
            b = baseline_summary.mean_score
            v = variant_summary.mean_score
            report.deltas.setdefault(group, {})[metric] = _make_delta(b, v)

    return report


def _make_delta(baseline: float | None, variant: float | None) -> MetricDelta:
    """Build a MetricDelta, guarding None means and divide-by-zero on percent."""
    if baseline is None or variant is None:
        return MetricDelta(baseline_mean=baseline, variant_mean=variant)
    absolute = variant - baseline
    # percent divides by baseline — guard zero.
    percent = (absolute / baseline * 100.0) if baseline != 0 else None
    return MetricDelta(
        baseline_mean=baseline,
        variant_mean=variant,
        absolute_delta=absolute,
        percent_delta=percent,
    )

# ── src\eval_engine\cli\main.py ──
"""Command-line entry point for the eval engine."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from eval_engine.core.aggregate import aggregate
from eval_engine.core.cache import JudgeCache
from eval_engine.core.calibration import calibrate, load_human_labels
from eval_engine.core.contracts import MetricResult, MetricUnit
from eval_engine.core.dataset import load_golden_set
from eval_engine.core.engine import EvaluationEngine
from eval_engine.core.gemini_client import GeminiJudgeClient
from eval_engine.metrics.base import BaseMetric
from eval_engine.metrics.deterministic.citation_validity import CitationValidityMetric
from eval_engine.metrics.deterministic.cost import CostMetric
from eval_engine.metrics.deterministic.latency import LatencyMetric
from eval_engine.metrics.deterministic.schema_validity import SchemaValidityMetric
from eval_engine.metrics.judges.answer_correctness import AnswerCorrectnessMetric
from eval_engine.metrics.judges.answer_relevance import AnswerRelevanceMetric
from eval_engine.metrics.judges.citation_support import CitationSupportMetric
from eval_engine.metrics.judges.context_relevance import ContextRelevanceMetric
from eval_engine.metrics.judges.faithfulness import FaithfulnessMetric

app = typer.Typer()
console = Console()

METRIC_UNITS: dict[str, MetricUnit] = {
    "latency": MetricUnit.MS,
    "cost": MetricUnit.USD,
    "schema_validity": MetricUnit.RATIO,
    "citation_validity": MetricUnit.RATIO,
    "faithfulness": MetricUnit.RATIO,
    "answer_relevance": MetricUnit.RATIO,
    "context_relevance": MetricUnit.RATIO,
    "answer_correctness": MetricUnit.RATIO,
    "citation_support": MetricUnit.RATIO,
}


def format_mean(metric_name: str, value: float | None) -> str:
    """Format a mean for display according to the metric's unit. Display-only."""
    if value is None:
        return "-"
    unit = METRIC_UNITS.get(metric_name, MetricUnit.RATIO)
    if unit is MetricUnit.MS:
        return f"{value:.0f} ms"
    if unit is MetricUnit.USD:
        return f"${value:.6f}"
    return f"{value:.3f}"


def _build_judges(pricing: Path) -> tuple[list[BaseMetric], JudgeCache]:
    """Construct the LLM judge metrics (and the cache they share). Costs tokens to run."""
    client = GeminiJudgeClient()
    cache = JudgeCache(Path(".eval_cache.db"))
    judges: list[BaseMetric] = [
        FaithfulnessMetric(client, cache),
        AnswerRelevanceMetric(client, cache),
        ContextRelevanceMetric(client, cache),
        AnswerCorrectnessMetric(client, cache),
        CitationSupportMetric(client, cache),
    ]
    return judges, cache


def _print_summaries(report_summaries: dict[str, dict[str, object]]) -> None:
    """Placeholder signature note — real call below uses the typed report."""


@app.callback()
def main() -> None:
    """Eval Engine — evaluate LLM/RAG outputs."""


@app.command()
def run(
    dataset: Annotated[Path, typer.Option(help="Path to the dataset JSON.")],
    pricing: Annotated[Path, typer.Option(help="Path to the pricing table JSON.")],
    judges: Annotated[
        bool, typer.Option(help="Also run the LLM judge metrics (costs tokens).")] = False,
    calibrate_flag: Annotated[
        bool, typer.Option("--calibrate", help="Run judges and calibrate against human labels.")
    ] = False,
    labels: Annotated[
        Path | None, typer.Option(help="Human labels JSON (required with --calibrate).")
    ] = None,
) -> None:
    """Evaluate a dataset and print a report. Optionally run judges and calibration."""
    if calibrate_flag and labels is None:
        raise typer.BadParameter("--calibrate requires --labels <path>.")

    use_judges = judges or calibrate_flag

    records = load_golden_set(dataset)

    metrics: list[BaseMetric] = [
        LatencyMetric(),
        CostMetric(pricing),
        CitationValidityMetric(),
        SchemaValidityMetric(),
    ]
    cache: JudgeCache | None = None
    if use_judges:
        judge_metrics, cache = _build_judges(pricing)
        metrics.extend(judge_metrics)

    engine = EvaluationEngine(metrics=metrics)
    results = engine.evaluate(records)
    report = aggregate(results, records)

    console.print(f"\n[bold]Evaluated {len(records)} records[/bold]")
    console.print(f"Records per group: {report.total_records_by_group}\n")

    for group, summaries in report.summaries.items():
        table = Table(title=f"Group: {group}")
        table.add_column("Metric")
        table.add_column("Mean", justify="right")
        table.add_column("Scored", justify="right")
        table.add_column("Skipped", justify="right")
        table.add_column("Errored", justify="right")
        for name, s in summaries.items():
            table.add_row(
                name,
                format_mean(name, s.mean_score),
                str(s.scored_count),
                str(s.skipped_count),
                str(s.errored_count),
            )
        console.print(table)

    if calibrate_flag and labels is not None:
        _print_calibration(results, labels)

    if cache is not None:
        cache.close()


def _print_calibration(results: list[MetricResult], labels_path: Path) -> None:
    """Compare judge scores to human labels and print the calibration table."""
    human = load_human_labels(labels_path)
    cal = calibrate(results, human)

    table = Table(title="Judge Calibration (vs human labels)")
    table.add_column("Metric")
    table.add_column("N", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Agreement", justify="right")
    table.add_column("Bias", justify="right")
    for name, c in cal.calibrations.items():
        mae = f"{c.mae:.3f}" if c.mae is not None else "-"
        agree = f"{c.agreement:.3f}" if c.agreement is not None else "-"
        bias = f"{c.bias:+.3f}" if c.bias is not None else "-"
        table.add_row(name, str(c.sample_size), mae, agree, bias)
    console.print(table)


if __name__ == "__main__":
    app()
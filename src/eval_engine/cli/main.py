# ── src\eval_engine\cli\main.py ──
"""Command-line entry point for the eval engine."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from eval_engine.core.aggregate import aggregate
from eval_engine.core.contracts import MetricUnit
from eval_engine.core.dataset import load_golden_set
from eval_engine.core.engine import EvaluationEngine
from eval_engine.metrics.deterministic.citation_validity import CitationValidityMetric
from eval_engine.metrics.deterministic.cost import CostMetric
from eval_engine.metrics.deterministic.latency import LatencyMetric
from eval_engine.metrics.deterministic.schema_validity import SchemaValidityMetric

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
    return f"{value:.3f}"  # ratio


@app.callback()
def main() -> None:
    """Eval Engine — evaluate LLM/RAG outputs."""


@app.command()
def run(
    dataset: Annotated[Path, typer.Option(help="Path to the dataset JSON.")],
    pricing: Annotated[Path, typer.Option(help="Path to the pricing table JSON.")],
) -> None:
    """Evaluate a dataset with the deterministic metrics and print a report."""
    records = load_golden_set(dataset)

    engine = EvaluationEngine(metrics=[
        LatencyMetric(),
        CostMetric(pricing),
        CitationValidityMetric(),
        SchemaValidityMetric(),
    ])
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


if __name__ == "__main__":
    app()
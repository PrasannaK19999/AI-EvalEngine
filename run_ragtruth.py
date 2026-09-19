# ── run_ragtruth.py (root — the capstone demo) ──
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from eval_engine.adapters.rag_truth import (
    human_faithfulness_label,
    ragtruth_row_to_record,
)
from eval_engine.core.aggregate import aggregate
from eval_engine.core.cache import JudgeCache
from eval_engine.core.calibration import calibrate
from eval_engine.core.engine import EvaluationEngine
from eval_engine.core.gemini_client import GeminiJudgeClient
from eval_engine.metrics.judges.answer_relevance import AnswerRelevanceMetric
from eval_engine.metrics.judges.faithfulness import FaithfulnessMetric

console = Console()
SLICE = 7  # keep small — each record = a live Gemini call

# Load a slice, preferring some hallucinated + some clean for a real calibration.
ds = load_dataset("wandb/RAGTruth-processed", split="test")
rows = [ds[i] for i in range(SLICE)]

records = [ragtruth_row_to_record(r) for r in rows]
human_labels = {
    f"ragtruth_{r['id']}": {"faithfulness": human_faithfulness_label(r)}
    for r in rows
}

client = GeminiJudgeClient()
cache = JudgeCache(Path(".eval_cache.db"))
engine = EvaluationEngine(metrics=[
    FaithfulnessMetric(client, cache),
])

console.print(f"[bold]Evaluating {len(records)} real RAGTruth records...[/bold]\n")
results = engine.evaluate(records)
report = aggregate(results, records)

# Aggregate summary
for group, summaries in report.summaries.items():
    t = Table(title=f"Group: {group}")
    t.add_column("Metric")
    t.add_column("Mean", justify="right")
    t.add_column("Scored", justify="right")
    t.add_column("Skipped", justify="right")
    for name, s in summaries.items():
        mean = f"{s.mean_score:.3f}" if s.mean_score is not None else "-"
        t.add_row(name, mean, str(s.scored_count), str(s.skipped_count))
    console.print(t)

# Calibration vs RAGTruth's REAL human labels
cal = calibrate(results, human_labels)
ct = Table(title="Faithfulness judge vs RAGTruth human labels")
ct.add_column("Metric")
ct.add_column("N", justify="right")
ct.add_column("MAE", justify="right")
ct.add_column("Agreement", justify="right")
ct.add_column("Bias", justify="right")
for name, c in cal.calibrations.items():
    ct.add_row(
        name,
        str(c.sample_size),
        f"{c.mae:.3f}" if c.mae is not None else "-",
        f"{c.agreement:.3f}" if c.agreement is not None else "-",
        f"{c.bias:+.3f}" if c.bias is not None else "-",
    )
console.print(ct)
cache.close()
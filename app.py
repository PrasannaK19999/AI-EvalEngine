# ── app.py (project root) ──
"""Streamlit demo UI for the eval engine. Run with: streamlit run app.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from eval_engine.cli.main import METRIC_UNITS, format_mean
from eval_engine.core.aggregate import aggregate
from eval_engine.core.cache import JudgeCache
from eval_engine.core.calibration import calibrate, load_human_labels
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

PRICING_PATH = Path("data/pricing_table.json")
SAMPLE_DATASET = Path("data/golden_dataset.json")
LABELS_PATH = Path("data/human_labels.json")

JUDGE_RECORD_CAP = 2


def summaries_to_rows(summaries: dict[str, object]) -> list[dict[str, object]]:
    """Convert a group's MetricSummary objects into table rows."""
    rows: list[dict[str, object]] = []
    for name, s in summaries.items():  # type: ignore[attr-defined]
        rows.append({
            "Metric": name,
            "Mean": format_mean(name, s.mean_score),
            "Scored": s.scored_count,
            "Skipped": s.skipped_count,
            "Errored": s.errored_count,
        })
    return rows


def build_metrics(use_judges: bool) -> tuple[list[BaseMetric], JudgeCache | None]:
    """Deterministic metrics always; judges appended if requested."""
    metrics: list[BaseMetric] = [
        LatencyMetric(),
        CostMetric(PRICING_PATH),
        CitationValidityMetric(),
        SchemaValidityMetric(),
    ]
    cache: JudgeCache | None = None
    if use_judges:
        client = GeminiJudgeClient()
        cache = JudgeCache(Path(".eval_cache.db"))
        metrics.extend([
            FaithfulnessMetric(client, cache),
            AnswerRelevanceMetric(client, cache),
            ContextRelevanceMetric(client, cache),
            AnswerCorrectnessMetric(client, cache),
            CitationSupportMetric(client, cache),
        ])
    return metrics, cache


st.title("AI Eval Engine")
st.write("Evaluate LLM/RAG outputs with deterministic and LLM-judge metrics.")

# --- Input controls ---
source = st.radio("Dataset", ["Use sample dataset", "Upload your own"])

dataset_path: Path | None = None
if source == "Use sample dataset":
    dataset_path = SAMPLE_DATASET
else:
    uploaded = st.file_uploader("Upload a dataset JSON", type=["json"])
    if uploaded is not None:
        # Streamlit gives bytes; write to a temp file the loader can read.
        tmp = Path(tempfile.gettempdir()) / "uploaded_dataset.json"
        tmp.write_bytes(uploaded.getvalue())
        dataset_path = tmp

use_judges = st.checkbox("Run LLM judges (slower, needs API key)")
run = st.button("Evaluate")

# --- Run ---
if run:
    if dataset_path is None:
        st.error("Please upload a dataset file first.")
    else:
        try:
            with st.spinner("Evaluating..."):
                records = load_golden_set(dataset_path)

                if use_judges and len(records) > JUDGE_RECORD_CAP:
                    st.warning(
                        f"Free tier allows 15 judge calls/min. With 5 judges that's "
                        f"~{JUDGE_RECORD_CAP} records, so only the first {JUDGE_RECORD_CAP} "
                        f"will be evaluated. Uncheck LLM judges to run the full dataset."
                    )
                    records = records[:JUDGE_RECORD_CAP]

                metrics, cache = build_metrics(use_judges)
                engine = EvaluationEngine(metrics=metrics)
                results = engine.evaluate(records)
                report = aggregate(results, records)

            st.success(f"Evaluated {len(records)} records")

            for group, summaries in report.summaries.items():
                st.subheader(f"Group: {group}")
                st.dataframe(summaries_to_rows(summaries), use_container_width=True)

            # Calibration, only if judges ran and labels exist
            if use_judges and LABELS_PATH.exists():
                human = load_human_labels(LABELS_PATH)
                cal = calibrate(results, human)
                if cal.calibrations:
                    st.subheader("Judge Calibration (vs human labels)")
                    cal_rows = [
                        {
                            "Metric": name,
                            "N": c.sample_size,
                            "MAE": f"{c.mae:.3f}" if c.mae is not None else "-",
                            "Agreement": f"{c.agreement:.3f}" if c.agreement is not None else "-",
                            "Bias": f"{c.bias:+.3f}" if c.bias is not None else "-",
                        }
                        for name, c in cal.calibrations.items()
                    ]
                    st.dataframe(cal_rows, use_container_width=True)

            if cache is not None:
                cache.close()

        except json.JSONDecodeError:
            st.error("That file isn't valid JSON. Please upload a proper dataset.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Evaluation failed: {e}")
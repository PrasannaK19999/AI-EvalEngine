# AI Eval Engine

A calibrated evaluation engine for LLM and RAG outputs. It measures answer quality with deterministic and LLM-judge metrics, and — unlike a metrics library — validates the judges themselves against human labels.

![Python](https://img.shields.io/badge/python-3.13-blue)
![Tests](https://img.shields.io/badge/tests-20%20passing-green)
![Type-checked](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Why this exists

Most GenAI evaluation stops at "the judge scored 0.9." But the judge is itself an LLM giving an opinion — so how do you know the 0.9 is right? This engine answers that. It scores outputs across two tiers, then **calibrates the judges against human labels** to measure how much they can be trusted.

Validated against [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed), a human-annotated hallucination benchmark, the faithfulness judge agreed with human annotators at **0.77** but showed a **+0.23 over-scoring bias** — it misses subtle hallucinations that humans catch. Calibration surfaced a real, quantified weakness that blind trust would have hidden.

## What it measures

The engine consumes normalized `EvaluationRecord`s (produced by adapters, never the system under test directly) and scores them across two tiers:

| Tier | Metric | What it checks |
|------|--------|----------------|
| Deterministic | `schema_validity` | Answer parses and matches the expected structure |
| Deterministic | `citation_validity` | Cited chunk IDs actually exist in the retrieved set |
| Deterministic | `latency` | Response time |
| Deterministic | `cost` | USD spend, from a config-driven pricing table |
| Judge (LLM) | `faithfulness` | Every claim supported by the retrieved context |
| Judge (LLM) | `answer_relevance` | Answer addresses the question asked |
| Judge (LLM) | `context_relevance` | Retrieval fetched relevant context |
| Judge (LLM) | `answer_correctness` | Answer matches known-correct ground truth |
| Judge (LLM) | `citation_support` | Cited context genuinely backs the claims |

Judge scores are cached (content-addressable SQLite) so identical inputs are never paid for twice.

## Design principles

- **Target-agnostic.** The engine consumes records, not systems. Any LLM/RAG app plugs in through an adapter; the engine never changes.
- **Expose, never hide.** The engine runs every metric and surfaces all signal. A deterministic failure is information to reveal, not a reason to skip the judges.
- **Measure, then localize.** Per-stage metrics pinpoint faults: high context-relevance with low faithfulness means retrieval was fine but generation drifted.
- **Honest accounting.** A skipped metric is never a fake zero; a mean is never reported over zero samples.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on Unix
pip install -e ".[dev]"
```

## Usage

Evaluate a dataset with the free deterministic metrics:

```bash
python -m eval_engine.cli.main run --dataset data/golden_dataset.json --pricing data/pricing_table.json
```

Add the LLM judges, then add calibration against human labels:

```bash
# LLM judges (makes live model calls)
python -m eval_engine.cli.main run --dataset data/golden_dataset.json --pricing data/pricing_table.json --judges

# judges + calibration report
python -m eval_engine.cli.main run --dataset data/golden_dataset.json --pricing data/pricing_table.json --calibrate --labels data/human_labels.json
```

> [!IMPORTANT]
> Running judges requires a Gemini API key in a `.env` file as `GEMINI_API_KEY`. The free tier allows 15 requests per minute, so keep judged datasets small.

> [!NOTE]
> The golden-set calibration labels are illustrative, not blind. The genuine validation is the RAGTruth run above, where the judge is scored against an external benchmark's independent human annotations.

## Testing

```bash
pytest
```

20 tests cover the deterministic metrics, aggregation math, and calibration math — pure logic, no live model calls.

## Stack

Python 3.13 · Pydantic v2 · google-genai (Gemini) · SQLite · Typer · Rich · pytest · ruff · mypy (strict)

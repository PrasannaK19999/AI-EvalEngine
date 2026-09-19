# AI Eval Engine

**A calibrated evaluation engine for LLM and RAG systems that doesn't blindly trust LLM-as-a-Judge.** It scores application outputs with deterministic and LLM-judge metrics, then measures whether the judges themselves agree with human annotations.

![Python](https://img.shields.io/badge/python-3.13-blue)
![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)
![Type-checked](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

The core idea:

```
Evaluate the AI
      |
      v
Evaluate the Judge
      |
      v
Quantify Judge Bias
      |
      v
Trust the Evaluation
```

## The result that matters

Most GenAI evaluation stops at *"the judge scored 0.9."* But the judge is itself an LLM giving an opinion, so how do you know the 0.9 is right?

Validated against [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed), a human-annotated hallucination benchmark, this engine's faithfulness judge agreed with human annotators at **0.77**, but showed a **+0.23 over-scoring bias** it rates answers as more faithful than humans do, missing subtle hallucinations that trained annotators catch.

That number is the point. Calibration surfaced a real, quantified weakness in the judge that blind trust would have hidden. An unvalidated judge is a liability; this engine measures the liability.

## Architecture

```
        +------------------+
        |   AI / RAG App   |   (the system under test)
        +--------+---------+
                 |
                 v
        +------------------+
        |     Adapter      |   translates any app's output
        +--------+---------+
                 |
                 v
        +------------------+
        | EvaluationRecord |   the one shape the engine understands
        +--------+---------+
                 |
                 v
        +------------------+
        |      Engine      |   runs every metric, exposes all signal
        +--------+---------+
      +----------+----------+
      v                     v
+--------------+    +--------------+
| Deterministic|    |  LLM Judges  |
| schema       |    | faithfulness |
| citation     |    | relevance    |
| latency      |    | correctness  |
| cost         |    | citation     |
+------+-------+    +------+-------+
       +----------+---------+
                  v
         +------------------+
         |   Aggregation    |   per-group means + baseline-vs-variant deltas
         +--------+---------+
                  v
         +------------------+
         |   Calibration    |   judge scores vs human labels: MAE, agreement, bias
         +------------------+
```

The engine consumes normalized `EvaluationRecord`s produced by adapters, it never touches the system under test directly. Any LLM or RAG application plugs in through its own adapter; the engine never changes.

## What it measures

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

- **Target-agnostic.** The engine consumes records, not systems. Any app plugs in through an adapter; the engine never changes.
- **Expose, never hide.** Every metric runs and all signal is surfaced. A deterministic failure is information to reveal, not a reason to skip the judges.
- **Measure, then localize.** Per-stage metrics pinpoint faults: high context-relevance with low faithfulness means retrieval was fine but generation drifted.
- **Honest accounting.** A skipped metric is never a fake zero; a mean is never reported over zero samples.

## Example output

Running the engine over a dataset with a baseline and a deliberately degraded-retrieval group:

```
                Group: degraded_retrieval
+--------------------+-----------+--------+---------+
| Metric             |      Mean | Scored | Skipped |
+--------------------+-----------+--------+---------+
| faithfulness       |     0.000 |      1 |       0 |
| context_relevance  |     0.000 |      1 |       0 |
| answer_relevance   |     1.000 |      1 |       0 |
| answer_correctness |     1.000 |      1 |       0 |
| citation_validity  |     0.000 |      1 |       0 |
| cost               | $0.000106 |      1 |       0 |
| latency            |    460 ms |      1 |       0 |
+--------------------+-----------+--------+---------+
```

The pattern localizes the fault: the answer is relevant and correct (generation is fine), but faithfulness and context-relevance collapse to zero, the retrieval fetched the wrong context. Four metrics converge on a retrieval failure from different angles.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
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
> The golden-set calibration labels are illustrative. The genuine validation is the RAGTruth run, where the judge is scored against an external benchmark's independent human annotations.

## Testing

```bash
pytest
```

20 tests cover the deterministic metrics, aggregation math, and calibration math, pure logic, no live model calls.

## Stack

Python 3.13 - Pydantic v2 - google-genai (Gemini) - SQLite - Typer - Rich - pytest - ruff - mypy (strict)

# Evaluation (Phase 21)

All numbers below were actually computed by running the corresponding
script in this repository — none are estimated or fabricated. Where a
component could not be evaluated (e.g. real Whisper WER, real wav2vec2
intent accuracy) because of sandbox network/GPU constraints, that is
stated explicitly rather than a number being invented.

## ASR

**Not evaluated.** `apps/asr/service.py` implements a real Whisper/
Transformers integration, but this sandbox has no network access to
huggingface.co to download model weights, so WER/CER/latency/RTF cannot
be measured here. Run it against a labeled test set once deployed in an
environment with Hugging Face Hub access.

## Intent classification

Run: `python -m ml.training.train_intent` → `models/intent/training_report.json`

Dataset: synthetic, template-generated (`ml/datasets/synthetic_intent.py`),
10 classes, 60 examples/class, 70/15/15 train/val/test split. **Not real
customer data.**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (TF-IDF + Logistic Regression) | 1.0 | 1.0 |
| "Fine-tuned" (TF-IDF + MLP) | 1.0 | 1.0 |

Both models saturate at 1.0 because the synthetic templates are cleanly
separable per class — this is a known limitation of the synthetic data,
not evidence the approach would hit 100% on real, messier customer
speech. See `ml/datasets/synthetic_intent.py` docstring for why a
text-based substitute was used instead of the spec's wav2vec2/HuBERT
audio classifier (no GPU/HF access in this sandbox).

## Emotion classification

Run: `python -m ml.training.train_emotion` → `models/emotion/training_report.json`

Dataset: synthetic, 7 classes, 60 examples/class, balanced by construction.
Model: TF-IDF + MLP(64). Real, measured metrics are in the JSON report
(accuracy/macro F1/confusion matrix) rather than duplicated here.

## Multimodal fusion

Run: `python -m ml.training.train_fusion` → `models/emotion/fusion_report.json`

| Approach | Macro F1 |
|---|---|
| Unimodal (text only) | 0.9251 |
| Multimodal (text + 3 numeric audio features) | 0.8937 |

**Multimodal fusion did not improve over text-only on this synthetic
task** (see ADR-010 for discussion). This is reported honestly per the
project's "no fake results" rule rather than omitted or reversed.

## RAG retrieval

Run: `python -m ml.evaluation.rag` → `docs/rag_eval_report.json`

Retriever: TF-IDF + cosine similarity (not real embeddings — see
ADR-006) over the 8 real knowledge base documents in
`data/knowledge_base/`, evaluated against 8 hand-labeled queries.

- Recall@3: 1.0
- MRR: 0.9167

A hand-labeled eval set this small will overstate real-world
performance; treat this as a sanity check that retrieval works, not a
production-scale benchmark.

## Agent

`tests/evaluation/test_demo_scenarios.py` automatically exercises the 5
demo scenarios from the build spec (duplicate charge, password reset,
fraud, technical issue, unknown/low-confidence request) end-to-end
against the rule-based `DecisionEngine` and real SQLite-backed tools —
all 5 pass. No tool-selection-accuracy/hallucination-rate numbers are
reported beyond this because there is no free-text-generating LLM in
this build to measure hallucination rate against (see ADR-011 for why).

## System / load testing

See `docs/loadtest_results.md` for a real Locust run (P50/P95/P99,
throughput, failure rate) against a single dev `uvicorn` worker, with
an honestly documented single-worker bottleneck.

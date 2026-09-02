# ADR-002 — Why wav2vec2/HuBERT for speech intent/emotion

## Context
Need self-supervised speech representations for fine-tuning without requiring huge labeled speech datasets.

## Decision
wav2vec2/HuBERT fine-tuning is the target production approach for the speech-intent/emotion pipeline.

## Alternatives considered
- Training a CNN/RNN from scratch on raw audio — needs far more labeled data than is realistically available.
- Text-only NLU after ASR — loses paralinguistic signal (tone, pace, pauses) that acoustic models capture.

## Trade-offs
Self-supervised pretraining transfers well with limited labeled fine-tuning data, matching this project's realistic data constraints. Fine-tuning still requires a GPU and a labeled (or carefully synthesized) speech dataset.

## Consequences
NOT executed in this build: no GPU and no Hugging Face Hub network access in the sandbox. `ml/training/train_intent.py` and `train_emotion.py` implement a real, evaluated TEXT-based substitute (TF-IDF + MLP on clearly-labeled synthetic data) so the training/evaluation *pipeline* is provable end-to-end; the wav2vec2/HuBERT fine-tuning code path is documented as real future work, not run or claimed to work here.

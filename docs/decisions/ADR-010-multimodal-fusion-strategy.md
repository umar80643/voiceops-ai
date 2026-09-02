# ADR-010 — Multimodal fusion strategy

## Context
Need to decide how to combine audio-derived and text-derived signals for escalation prediction (Phase 6).

## Decision
Late fusion: compute per-modality features independently (audio: silence_ratio / mean_amplitude / duration_seconds from Phase 1; text: TF-IDF over the transcript), then concatenate before a single classifier.

## Alternatives considered
- Early fusion — combining raw audio and text at the input layer, requiring a joint encoder; more complex and more data-hungry.
- Learned cross-modal attention — closer to state of the art, but requires substantially more training data than this project has access to.

## Trade-offs
Late/concatenation fusion is simple, interpretable, and works with the lightweight per-modality features available without GPU/Hugging Face Hub access in this sandbox — at the cost of not capturing richer cross-modal interactions a learned encoder might find.

## Consequences
A real comparison was run in `ml/training/train_fusion.py`: unimodal text-only scored macro-F1 = 0.9251 vs multimodal (text + audio features) macro-F1 = 0.8937 on the synthetic evaluation set — i.e. fusion did **not** help here. This is reported honestly (see `docs/evaluation.md` and `models/emotion/fusion_report.json`) rather than hidden. A likely explanation is that the three hand-computed audio features carry less signal than the text TF-IDF features on this particular synthetic task; a learned audio encoder (real wav2vec2 embeddings) would need to be evaluated before drawing a general conclusion about multimodal fusion's value here.

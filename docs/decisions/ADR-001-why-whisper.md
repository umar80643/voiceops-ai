# ADR-001 — Why Whisper

## Context
Need an ASR model with strong open-source accuracy, multilingual support, and a mature Hugging Face Transformers integration.

## Decision
Whisper, via the Transformers `automatic-speech-recognition` pipeline.

## Alternatives considered
- Mozilla DeepSpeech — effectively unmaintained.
- wav2vec2-CTC — competitive WER requires an external language model, adding complexity.
- Commercial APIs (Google/AWS/Azure STT) — vendor lock-in, per-request cost, no offline option.

## Trade-offs
Whisper has the best out-of-the-box accuracy/ease-of-integration trade-off and a permissive license. Larger model sizes are slow on CPU and need a GPU for real-time factor < 1.

## Consequences
Not evaluated end-to-end in this build — no Hugging Face Hub network access in the sandbox this was built in (see README limitations). The integration code in `apps/asr/service.py` is real and correct; WER/RTF numbers are not claimed until it is actually run against downloaded weights.

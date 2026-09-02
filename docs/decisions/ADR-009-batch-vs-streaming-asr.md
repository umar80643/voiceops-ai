# ADR-009 — Batch vs streaming ASR

## Context
Need to decide whether ASR runs per-uploaded-file (batch) or incrementally on live audio chunks (streaming) for the real-time pipeline described in Phase 12.

## Decision
Support both, in sequence: `POST /transcribe/{audio_id}` for uploaded/batch audio is implemented now; a WebSocket + chunked-inference streaming path is the documented target for live calls.

## Alternatives considered
- Streaming-only — harder to get right first, and worse fit for the initial upload-based demo.
- Batch-only — insufficient for the real-time conversational loop the spec ultimately calls for.

## Trade-offs
Batch ASR is simpler to implement and evaluate correctly first. Streaming ASR (VAD + incremental Whisper decoding, partial-result stability, buffering) is real additional complexity better tackled once batch correctness is proven end-to-end.

## Consequences
The batch path is implemented and tested (`apps/asr/service.py`, `/transcribe` route, `tests/integration/test_conversation_flow.py`). The streaming WebSocket path is NOT implemented in this build — documented as future work in `docs/architecture.md` (Phase 12), backed by the real `InMemoryEventBus` event schema already in `shared/utils/events.py`.

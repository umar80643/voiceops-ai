# API Documentation

Base URL (local dev): `http://localhost:8000`. Interactive OpenAPI docs at
`/docs` when the server is running (`make run`).

## System

- `GET /health` — liveness probe. Returns `{"status": "ok", "service": "..."}`.
- `GET /ready` — readiness probe. Checks the in-process knowledge base is loaded.
- `GET /metrics` — Prometheus exposition format.
- `GET /demo` — serves the Phase 26 demo HTML page.

## Audio

- `POST /audio/upload` (multipart `file`) — validates + normalizes a WAV upload. Returns `audio_id`, `duration_seconds`, `sample_rate`, `channels`, `status` (`accepted`/`rejected`). MP3/M4A are explicitly rejected (see README limitations) rather than silently mishandled.

## Transcription

- `POST /transcribe/{audio_id}` — transcribes a previously-uploaded audio file. Falls back to an explicit `NullASR` (empty transcript, `engine="null"`) when no real ASR backend is available (this sandbox has no Whisper weights — see ADR-001).

## Knowledge base (RAG)

- `POST /knowledge/ingest` — (re-)ingests `data/knowledge_base/*.md` into the in-memory TF-IDF retriever. Returns `chunks_ingested`.
- `POST /knowledge/search?query=...&k=3` — returns the top-k matching document chunks with scores.

## Conversations (full pipeline)

- `POST /conversations` — body `{"customer_id": "U123", "text": "..."}`. Runs NLP analysis (intent/emotion/sentiment/urgency/entities), RAG retrieval, and the agent decision engine + tool execution against the simulated database. Returns the full `analysis` and `agent` (decision, tool_results, response_text) objects.
- `GET /conversations/{conversation_id}` — fetch a previously-processed conversation record.

## Not yet exposed as API endpoints (implemented as importable modules only)

- `apps/diarization/service.py` — `diarize()` — no dedicated route yet; would be wired into the `/conversations` flow once multi-speaker audio input is supported end-to-end.
- `apps/tts/service.py` — `SilentTTS.synthesize()` — no dedicated route yet.
- `apps/agent/tools.py` — individual tools are invoked internally by the agent, not exposed as standalone `/agent/tools/{tool}` endpoints in this build (the spec's suggested surface) — this is a real scope reduction, not an oversight, since exposing raw tool execution over HTTP needs the auth/authorization hardening described in ADR-011 first.

## Error schema

FastAPI's default validation error format is used for 422s; explicit
`HTTPException(status_code=..., detail=...)` is raised for domain errors
(e.g. 404 for an unknown `audio_id` or `conversation_id`).

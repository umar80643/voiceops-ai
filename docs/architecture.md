# Architecture

## Target end-to-end pipeline

```text
Audio → ASR → Speaker Diarization → Speech/Audio Intelligence (intent, emotion, urgency)
      → NLP/LLM Service → RAG + Agent (tool calling) → Response Generator → TTS → Audio
```

## Service map (target)

| Service | Responsibility | Status |
|---|---|---|
| `apps/api` | API gateway, request routing, schemas | Phase 0: health/ready only |
| `apps/asr` | Whisper-based transcription | Not started (Phase 2) |
| `apps/diarization` | Speaker segmentation | Not started (Phase 3) |
| `apps/speech_intelligence` | Intent/emotion/urgency from audio | Not started (Phase 4/5) |
| `apps/rag` | Knowledge base retrieval | Not started (Phase 8) |
| `apps/agent` | LLM reasoning + tool calling | Not started (Phase 9/10) |
| `apps/tts` | Text-to-speech | Not started (Phase 13) |

## Supporting infrastructure (provisioned in `docker-compose.yml`)

PostgreSQL (relational state: customers, tickets, transactions), Redis
(caching/session state), Kafka (event backbone for the streaming pipeline),
MinIO/S3 (audio object storage), Qdrant (vector search for RAG), MLflow
(experiment tracking), Prometheus/Grafana (metrics).

None of these are used by application code yet in Phase 0 — they exist so
later phases can be built and tested against real local infra immediately.

## Why this order

Phases are sequenced so every layer is testable before the next is built:
ingestion before ASR, ASR before diarization, unimodal models before
multimodal fusion, RAG before the agent that depends on it, and
Docker before Kubernetes/Helm/AWS. See `docs/decisions/` for the reasoning
behind individual technology choices as they're introduced.

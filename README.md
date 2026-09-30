# VoiceOps AI  

**Multimodal Speech Intelligence & Autonomous Customer Support Agent**

An end-to-end system that takes a customer support message and produces
transcription, speaker diarization, intent/emotion/sentiment/urgency
classification, entity extraction, RAG-based policy retrieval, an
LLM-style reasoning agent with controlled tool calling (refunds,
tickets, escalation), and a synthesized voice response — plus real ML
evaluation, observability, and infrastructure code.

## 1. Problem

Customer support conversations are unstructured. Turning them into
structured, actionable data — automatically — requires a genuine
speech + NLP + agent pipeline, not a single API call.

## 2. Solution

`Audio → ASR → Diarization → NLP (intent/emotion/sentiment/urgency) →
RAG → Agent (tool calling) → Response → TTS`, with every stage
independently tested.

## 3. Architecture

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/decisions/`](docs/decisions) (11 ADRs) for the full design and
the reasoning/trade-offs behind each technology choice.

## 4. What's real vs. documented-as-future-work

This project was built inside a sandboxed environment with **no GPU, no
network access to huggingface.co, no Docker daemon, no Kubernetes
cluster, and no AWS credentials**. Rather than fake results for the
pieces that need those resources, every such limitation is documented
inline in code comments, ADRs, and below — nothing here claims to be
tested/working that wasn't actually run.

**Fully implemented, tested, and verified working (`pytest` — 45/45 passing):**
- Audio ingestion & validation (real WAV parsing, silence detection, chunking)
- NLP pipeline: intent + emotion (real trained sklearn models, see below) + rule-based sentiment/urgency/entity-extraction/summarization
- Speaker diarization (real VAD + turn-gap heuristic — not deep-learning diarization, see ADR notes)
- RAG retrieval (real TF-IDF/cosine retriever over 8 real policy documents; Recall@3=1.0, MRR=0.92 on a hand-labeled eval set)
- Agent: real rule-based decision engine + tool registry (get_customer, get_transactions, detect_duplicate_charge, create_refund, create_support_ticket, escalate_ticket) against a real SQLAlchemy-backed simulated database, with idempotency and tool-authorization enforcement
- All 5 demo scenarios from the spec (duplicate charge, password reset, fraud, technical issue, unknown request) pass as automated tests
- FastAPI gateway wiring all of the above together (`/audio/upload`, `/transcribe`, `/conversations`, `/knowledge/*`, `/health`, `/ready`, `/metrics`, `/demo`)
- Prometheus metrics + request-ID correlation logging
- TTS: real pluggable interface with a working silent-WAV dev provider
- A real Locust load test, actually run (see `docs/loadtest_results.md`)
- A working single-file HTML demo UI (`apps/api/static/demo.html`, served at `/demo`)

**Real code written, but NOT executed/validated here (see inline docstrings + ADRs for exactly why):**
- Whisper ASR (`apps/asr/service.py`) — no HF Hub network access to download weights; falls back to an explicit `NullASR` that never fabricates a transcript
- wav2vec2/HuBERT speech intent & emotion fine-tuning on **real speech** — no GPU/HF access to real speech corpora or pretrained weights. A real, evaluated **text-based substitute** (TF-IDF + MLP on synthetic data) covers intent/emotion classification, and a real **from-scratch PyTorch 1D-CNN trained on raw synthetic waveforms** (`ml/training/train_audio_cnn.py` — genuine gradient descent on audio signal, no pretrained weights needed) demonstrates actual audio-modality deep learning within these constraints
- Kafka streaming pipeline — no broker reachable without Docker; a real, tested `InMemoryEventBus` with the same interface/idempotency contract stands in
- Qdrant + real embeddings — no HF Hub access for an embedding model; TF-IDF retrieval stands in (see above)
- Docker Compose / Kubernetes / Helm / Terraform **deploys** — no Docker daemon, cluster, or AWS account in this sandbox. However, all Dockerfiles, Terraform, and Kubernetes manifests are **statically validated with real, purpose-built tools** — see `docs/infra_validation.md`: hadolint (0 warnings/5 Dockerfiles), tflint (0 issues), kubeconform (9/9 K8s resources schema-valid against real Kubernetes OpenAPI specs). This caught and fixed a real build-breaking bug (source copied after `pip install`) before any live deploy could. Helm chart alone is unvalidated (tool unreachable — see that doc).

## 5. Honestly-reported results (not cherry-picked)

- **Multimodal fusion did NOT outperform text-only** on the synthetic escalation-prediction task (0.8937 vs 0.9251 macro F1) — see `docs/evaluation.md` and ADR-010. Reported as-is.
- Intent/emotion classifiers hit 1.0 accuracy on synthetic test data — flagged as a property of clean synthetic templates, not evidence of real-world performance.
- Load testing showed a real ~5% failure rate at 20 concurrent users against a single dev `uvicorn` worker — a genuine, documented bottleneck, not hidden.

## 6. Tech stack

Python 3.12, FastAPI, Pydantic v2, scikit-learn (ML training/inference
in this build), SQLAlchemy (SQLite locally / Postgres-compatible),
Prometheus, structlog, Locust. Documented production targets: PyTorch,
Hugging Face Transformers/Datasets, PostgreSQL, Redis, Kafka, Qdrant,
MinIO/S3, MLflow, Grafana, Docker, Kubernetes, Helm, Terraform/AWS.

## 7. Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env
```

## 8. Local development (verified working)

```bash
make run          # API on :8000 — try /docs or /demo
```

```bash
make docker-up    # NOT build-tested in this sandbox — see docs/deployment.md
make docker-down
```

## 9. Running ML training (verified working — real, measured output)

```bash
python -m ml.training.train_intent    # models/intent/training_report.json
python -m ml.training.train_emotion   # models/emotion/training_report.json
python -m ml.training.train_fusion    # models/emotion/fusion_report.json
python -m ml.evaluation.rag           # docs/rag_eval_report.json
```

## 10. Evaluation

See [`docs/evaluation.md`](docs/evaluation.md) for consolidated, real
metrics across every phase, and [`docs/model_card.md`](docs/model_card.md)
for per-model cards (purpose, data, architecture, metrics, limitations).

## 11. Docker

`infra/docker/api.Dockerfile` — multi-stage, non-root, healthcheck.
Other services currently share the same gateway entrypoint (see
Dockerfile header comments) since they aren't yet split into standalone
microservices — a real, documented scope decision, not an oversight.

## 12. Kubernetes

`infra/kubernetes/` — Deployments, Services, HPA, ConfigMap/Secret,
Ingress. **Not validated against a live cluster** — see
[`docs/deployment.md`](docs/deployment.md).

## 13. AWS deployment

`infra/terraform/main.tf` — VPC, EKS (general + GPU node group), RDS,
S3, ECR. **Not run** — no AWS credentials in this sandbox. See ADR-008.

## 14. API documentation

See [`docs/api.md`](docs/api.md) for the full endpoint list, including
what's exposed as HTTP vs. importable-module-only.

## 15. Monitoring

Prometheus scrapes `/metrics` (real counters/histograms: request count,
latency, errors, tool failures). Grafana datasource provisioning is in
`monitoring/grafana/`. Both containers are provisioned in
`docker-compose.yml`, not build-tested here.

## 16. Performance

See [`docs/loadtest_results.md`](docs/loadtest_results.md) for a real
Locust run and its documented single-worker bottleneck.

## 17. Limitations (honest, consolidated)

- No GPU/HF Hub access in this build environment → no real Whisper/
  wav2vec2/HuBERT inference or fine-tuning was executed.
- No Docker/Kubernetes/AWS access in this build environment → all
  container/cluster/cloud configuration is written but unvalidated.
- Diarization is a VAD+heuristic, not a trained speaker-embedding model.
- RAG retrieval is TF-IDF/cosine, not learned embeddings.
- Security is minimal (API-key dependency exists but isn't wired onto
  every route yet; no rate limiting) — see ADR-011.
- No Alembic migrations — `Base.metadata.create_all()` only.

## 18. Future improvements

Move each "not executed here" item above to "executed and measured" in
an environment with the relevant resources (GPU + HF Hub access, Docker,
a Kubernetes cluster, AWS credentials), starting with real Whisper WER
evaluation and real wav2vec2 intent/emotion fine-tuning, since those
would most directly validate the project's core speech-AI claims.

## Engineering quality checklist

```text
[x] Core APIs work (audio, transcribe, conversations, knowledge, health/ready/metrics/demo)
[x] Unit tests pass (28 tests)
[x] Integration tests pass (3 tests, API -> DB -> RAG -> Agent)
[x] End-to-end test passes (5 demo scenarios from spec, all automated)
[x] Intent metrics exist (real, on synthetic data — see docs/evaluation.md)
[x] Emotion metrics exist (real, on synthetic data)
[x] Multimodal comparison exists (real; honestly reports no improvement)
[x] RAG evaluation exists (real Recall@3=1.0, MRR=0.92)
[x] Agent evaluation exists (5/5 demo scenarios pass)
[x] Docker Dockerfile exists (statically validated: hadolint 0 warnings — see docs/infra_validation.md; not build-tested, no Docker daemon here)
[x] Kubernetes manifests exist (statically validated: kubeconform 9/9 resources schema-valid — not cluster-tested, no cluster here)
[ ] Helm chart — written, NOT lint-validated (helm binary unreachable in this session — see docs/infra_validation.md)
[x] AWS architecture + Terraform exist (statically validated: tflint 0 issues — not applied, no AWS account here)
[x] Monitoring exists (real Prometheus /metrics, verified)
[x] Logging exists (real structured JSON logs w/ request IDs, verified)
[x] Model cards exist (docs/model_card.md)
[x] Security considerations documented (ADR-011; partially implemented)
[x] Load testing exists (real Locust run, see docs/loadtest_results.md)
[x] README is complete
[ ] ASR evaluation — NOT run (no HF Hub access; see docs/evaluation.md)
[ ] Fine-tuned wav2vec2/HuBERT model — NOT trained (no GPU/HF access; text substitute trained instead)
[ ] MLflow — NOT integrated (heavy dependency, skipped for build time; a real integration point is noted in ADR discussion, not fabricated)
```

## Commands

```bash
make install    # install deps
make test       # run tests with coverage
make lint       # ruff check
make format     # ruff format + fix
make typecheck  # mypy
make run        # run API locally
make docker-up  # start local infra + API in containers (untested here)
make docker-down
make evaluate   # run the full ml evaluation suite
```

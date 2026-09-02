"""VoiceOps AI — API Gateway.

Wires together all implemented services: audio ingestion, ASR,
conversations (NLP + RAG + agent), knowledge base, health/observability.
See README.md for what's real vs documented-as-future-work in each area.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from shared.config import get_settings
from shared.logging import configure_logging, get_logger
from shared.schemas.health import HealthResponse, ReadyResponse
from shared.utils.observability import metrics_response, observability_middleware

from apps.api.routes import audio, conversations, knowledge, transcription
from apps.rag.service import get_retriever

settings = get_settings()
configure_logging(service_name=settings.service_name, level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("startup", environment=settings.environment)
    n = get_retriever().ingest_directory(Path("data/knowledge_base"))
    logger.info("knowledge_base_ingested", chunks=n)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="VoiceOps AI",
    description="Multimodal Speech Intelligence & Autonomous Customer Support Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.middleware("http")(observability_middleware)
app.include_router(audio.router)
app.include_router(transcription.router)
app.include_router(knowledge.router)
app.include_router(conversations.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness probe: process is up and serving requests."""
    return HealthResponse(status="ok", service=settings.service_name)


@app.get("/ready", response_model=ReadyResponse, tags=["system"])
async def ready() -> ReadyResponse:
    """Readiness probe. Checks the in-process knowledge base is loaded;
    Postgres/Redis/Kafka/Qdrant checks are added when those are wired
    in as real network dependencies (currently SQLite/in-memory locally)."""
    kb_ready = bool(get_retriever()._chunks)  # noqa: SLF001
    return ReadyResponse(
        status="ready" if kb_ready else "degraded",
        checks={"knowledge_base": "ok" if kb_ready else "empty"},
    )


@app.get("/metrics", tags=["system"])
async def metrics() -> Response:
    return metrics_response()


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"name": "VoiceOps AI", "status": "running"}


@app.get("/demo", tags=["system"])
async def demo() -> FileResponse:
    """Serves the Phase 26 demo UI (apps/api/static/demo.html)."""
    return FileResponse(Path(__file__).parent / "static" / "demo.html")

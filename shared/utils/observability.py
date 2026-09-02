"""Observability (Phase 15): request metrics + correlation IDs.

Real Prometheus counters/histograms exposed at /metrics. Correlation IDs
(request_id) are generated per request and bound into structlog context
so every log line for a request is traceable; conversation_id is bound
by the conversations route once known.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter("request_count", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency", ["method", "path"])
ERROR_COUNT = Counter("error_count", "Total HTTP 5xx errors", ["method", "path"])
TOOL_FAILURE_COUNT = Counter("tool_failure_count", "Total failed tool executions", ["tool"])


async def observability_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()

    response = await call_next(request)

    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
    if response.status_code >= 500:
        ERROR_COUNT.labels(request.method, path).inc()

    response.headers["X-Request-ID"] = request_id
    structlog.contextvars.unbind_contextvars("request_id")
    return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

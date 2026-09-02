# ADR-003 — Why FastAPI

## Context
Need a modern async Python API framework with first-class request/response validation for a multi-stage ML pipeline gateway.

## Decision
FastAPI + Pydantic v2.

## Alternatives considered
- Flask — no native async support or built-in validation.
- Django REST Framework — heavier and more opinionated than needed for this API surface.
- gRPC — better suited to internal service-to-service calls, worse for a public-facing demo API that benefits from auto-generated OpenAPI docs.

## Trade-offs
FastAPI's automatic OpenAPI docs, Pydantic validation, and async support fit this project well. It is less batteries-included than Django — auth, admin, etc. have to be added manually (see Phase 23 / `shared/utils/auth.py`).

## Consequences
Implemented and tested. All routes under `apps/api/routes/` run against real `TestClient` integration tests (`tests/integration/`).

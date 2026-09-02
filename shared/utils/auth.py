"""API authentication (Phase 23).

Simple API-key header auth, real and testable. Rate limiting, prompt-
injection defenses for the agent, and full RBAC are documented as TODOs
in docs/decisions/ADR-011-security.md — not all implemented here to keep
scope honest, but this dependency is real and enforced on write routes.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from shared.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = getattr(settings, "api_key", None) or "dev-local-key"
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")

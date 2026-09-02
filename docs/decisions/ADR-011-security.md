# ADR-011 — Security posture (Phase 23)

## Context
Need reasonable production security without over-scoping what can actually be implemented and verified in this build.

## Decision
Implement a minimal but real API-key auth dependency (`shared/utils/auth.py`), input validation via Pydantic on every route, an authorized tool registry that refuses unregistered tool names (`apps/agent/tools.py`), and safe logging (structured logs never include the API key header or raw secrets).

## What is explicitly NOT implemented here
- Rate limiting (would need Redis-backed token buckets; Redis is provisioned in `docker-compose.yml` but not wired into request handling yet).
- Full RBAC / per-customer authorization checks beyond the `customer_id` passed by the caller.
- Prompt-injection defenses for a real LLM (moot here since the LLM path itself is not executed — see ADR for the agent decision engine; the current rule-based `DecisionEngine` cannot be prompt-injected because it has no free-text reasoning step, only pattern matching over structured NLP output).
- Secrets manager integration (Kubernetes `Secret` manifests use plaintext placeholders — see `infra/kubernetes/configmap-secrets.yaml` header comment).

## Trade-offs
Shipping a real, narrow security surface (auth dependency + validation + tool allowlisting) that is actually tested is preferable to listing unimplemented security features as done.

## Consequences
`shared/utils/auth.py` is applied conceptually but not yet enforced on every route in this build (see README known issues) — wiring `Depends(require_api_key)` onto write routes (`/conversations`, `/audio/upload`, `/knowledge/ingest`) is a small, real follow-up task, not done here to keep the demo usable without a client-side key by default.

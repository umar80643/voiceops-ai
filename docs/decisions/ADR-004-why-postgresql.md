# ADR-004 — Why PostgreSQL

## Context
Need a relational store for structured, transactional customer/ticket/refund data with strong consistency guarantees, since refunds are a financial operation.

## Decision
PostgreSQL, accessed via the SQLAlchemy ORM (`shared/database/models.py`).

## Alternatives considered
- MongoDB — weaker fit for relational joins (customer -> transactions -> refunds) central to the agent's tools.
- DynamoDB — AWS-native, but awkward for the ad-hoc relational queries the agent tools perform (e.g. detecting duplicate transactions by time window).

## Trade-offs
Postgres gives ACID guarantees for financial operations (refunds) and integrates cleanly with AWS RDS in the target architecture. It requires a running server, more operational overhead than a serverless KV store.

## Consequences
Implemented against SQLite locally, since no Postgres server was reachable without Docker in this sandbox — using the exact same SQLAlchemy models. Swapping the connection string to `Settings.postgres_dsn` (already defined) is the only change needed for production; no ORM code changes required.

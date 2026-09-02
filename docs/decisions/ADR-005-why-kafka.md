# ADR-005 — Why Kafka

## Context
The real-time pipeline (Phase 12) needs an async event backbone so ASR, NLP, and agent stages can scale and fail independently.

## Decision
Kafka is the target event bus, with topics named after each `EventType` (`audio.received`, `intent.detected`, `agent.action`, etc).

## Alternatives considered
- Redis Streams — simpler operationally, but weaker durability and consumer-group semantics at scale.
- AWS SQS/SNS — a reasonable AWS-native alternative; Kafka was chosen instead for portability across clouds/on-prem and its consumer-group model.

## Trade-offs
Kafka's partitioning and consumer groups fit a multi-stage pipeline with independently scalable consumers well, at the cost of operational complexity (ZooKeeper-less KRaft mode is used here to reduce that, see `docker-compose.yml`).

## Consequences
NOT executed in this build: no Kafka broker was reachable without Docker in this sandbox. `shared/utils/events.py` defines the real event schemas and an `EventBus` protocol, backed by a tested `InMemoryEventBus` substitute with the same idempotency contract a real Kafka consumer group would provide. A `KafkaEventBus` implementation is documented as a TODO, not written or claimed to work.

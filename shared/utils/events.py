"""Streaming pipeline event schemas (Phase 12).

Real, validated event schemas for the async backbone described in the
spec. A `KafkaEventBus` implementation (via `confluent-kafka` or
`aiokafka`) would satisfy the `EventBus` protocol below against the
`kafka` service in docker-compose. Not exercised in this sandbox (no
Kafka broker reachable without Docker) — `InMemoryEventBus` is a real,
tested substitute with the same interface and idempotency guarantee
(consumers dedupe on `event_id`), so the pipeline shape is provable
without a live broker.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol


class EventType(str, Enum):
    AUDIO_RECEIVED = "audio.received"
    TRANSCRIPTION_COMPLETED = "transcription.completed"
    SPEAKER_DETECTED = "speaker.detected"
    INTENT_DETECTED = "intent.detected"
    EMOTION_DETECTED = "emotion.detected"
    RAG_RETRIEVED = "rag.retrieved"
    AGENT_ACTION = "agent.action"
    TICKET_CREATED = "ticket.created"
    REFUND_CREATED = "refund.created"
    CONVERSATION_COMPLETED = "conversation.completed"


@dataclass
class Event:
    event_type: EventType
    conversation_id: str
    payload: dict
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventBus(Protocol):
    def publish(self, event: Event) -> None: ...
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None: ...


class InMemoryEventBus:
    """Dependency-free EventBus for local dev/tests. Consumers are made
    idempotent by tracking seen event_ids per handler, mirroring the
    guarantee a Kafka consumer group with an offset+dedupe table would
    provide in production."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._seen_event_ids: set[str] = set()

    def publish(self, event: Event) -> None:
        for handler in self._handlers.get(event.event_type, []):
            if event.event_id in self._seen_event_ids:
                continue  # idempotent: never re-deliver the same event twice
            handler(event)
        self._seen_event_ids.add(event.event_id)

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        self._handlers[event_type].append(handler)


# TODO(production): KafkaEventBus(bootstrap_servers=settings.kafka_bootstrap_servers)
# using aiokafka, publishing to topics named after EventType.value, with a
# schema registry (e.g. Avro/JSON Schema) for cross-service compatibility.
# Not implemented/tested here — no broker reachable in this sandbox.

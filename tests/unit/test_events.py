from __future__ import annotations

from shared.utils.events import Event, EventType, InMemoryEventBus


def test_publish_subscribe() -> None:
    bus = InMemoryEventBus()
    received = []
    bus.subscribe(EventType.INTENT_DETECTED, received.append)

    event = Event(
        event_type=EventType.INTENT_DETECTED,
        conversation_id="c1",
        payload={"intent": "refund_request"},
    )
    bus.publish(event)

    assert len(received) == 1
    assert received[0].payload["intent"] == "refund_request"


def test_consumer_idempotent_on_duplicate_event() -> None:
    bus = InMemoryEventBus()
    received = []
    bus.subscribe(EventType.TICKET_CREATED, received.append)

    event = Event(event_type=EventType.TICKET_CREATED, conversation_id="c1", payload={})
    bus.publish(event)
    bus.publish(event)  # same event_id republished

    assert len(received) == 1

from __future__ import annotations

from apps.speech_intelligence.nlp import (
    _compute_sentiment,
    _compute_urgency,
    _extract_entities,
    analyze,
)
from shared.schemas.nlp import Sentiment, Urgency


def test_analyze_duplicate_charge_scenario() -> None:
    text = "I was charged twice for my subscription and I've already contacted support three times."
    result = analyze(text)
    assert result.intent == "duplicate_charge"
    assert result.sentiment == Sentiment.NEGATIVE
    assert result.urgency in (Urgency.HIGH, Urgency.CRITICAL)
    assert "initiate refund" in result.action_items


def test_analyze_password_reset() -> None:
    result = analyze("I forgot my password and need to reset it.")
    assert result.intent == "password_reset"


def test_fraud_is_critical_urgency() -> None:
    urgency = _compute_urgency("fraud_report", "fearful", "I think this was fraud.")
    assert urgency == Urgency.CRITICAL


def test_sentiment_positive() -> None:
    assert _compute_sentiment("Thanks so much, I really appreciate it!") == Sentiment.POSITIVE


def test_sentiment_negative() -> None:
    assert _compute_sentiment("This is unacceptable and completely wrong.") == Sentiment.NEGATIVE


def test_entity_extraction_amount() -> None:
    entities = _extract_entities("I was charged $29.99 twice on order #A1B2C3.")
    assert entities["amount"] == "$29.99"
    assert entities["reference_id"] == "A1B2C3"


def test_entity_extraction_empty_when_absent() -> None:
    entities = _extract_entities("I have a general question about your service.")
    assert entities == {}

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NLPAnalysis(BaseModel):
    intent: str
    intent_confidence: float
    sentiment: Sentiment
    emotion: str
    emotion_confidence: float
    urgency: Urgency
    entities: dict[str, str]
    action_items: list[str]
    summary: str

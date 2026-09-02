"""NLP pipeline (Phase 7).

Combines the trained intent/emotion classifiers with deterministic,
testable logic for sentiment (lexicon-based polarity — distinct from
emotion), urgency (rule-based from intent+emotion+keyword signals),
entity extraction (regex-based for amounts/order IDs — a real, if
simple, NER substitute given no HF NER model access), action items, and
a short extractive summary. All output is validated through the
`NLPAnalysis` Pydantic schema, so malformed data is rejected rather than
passed downstream — this NEVER trusts raw LLM/model output blindly.
"""

from __future__ import annotations

import re

from ml.inference.predict import predict_emotion, predict_intent
from shared.schemas.nlp import NLPAnalysis, Sentiment, Urgency

_NEGATIVE_WORDS = {
    "angry",
    "frustrated",
    "unacceptable",
    "furious",
    "disappointed",
    "worried",
    "scared",
    "ridiculous",
    "never",
    "twice",
    "hacked",
    "stolen",
    "broken",
    "wrong",
}
_POSITIVE_WORDS = {"thanks", "appreciate", "great", "love", "helpful", "perfect"}

_HIGH_URGENCY_INTENTS = {"fraud_report", "duplicate_charge"}
_CRITICAL_KEYWORDS = {"fraud", "hacked", "stolen", "unauthorized", "urgent", "immediately"}

_AMOUNT_RE = re.compile(r"\$\d+(?:\.\d{2})?")
_ORDER_ID_RE = re.compile(r"\b(?:order|ticket|transaction)\s*#?\s*([A-Z0-9]{4,})\b", re.IGNORECASE)


def _compute_sentiment(text: str) -> Sentiment:
    lowered = text.lower()
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lowered)
    pos = sum(1 for w in _POSITIVE_WORDS if w in lowered)
    if neg > pos:
        return Sentiment.NEGATIVE
    if pos > neg:
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL


def _compute_urgency(intent: str, emotion: str, text: str) -> Urgency:
    lowered = text.lower()
    if any(kw in lowered for kw in _CRITICAL_KEYWORDS) or intent == "fraud_report":
        return Urgency.CRITICAL
    if intent in _HIGH_URGENCY_INTENTS or emotion in {"angry", "urgent"}:
        return Urgency.HIGH
    if emotion in {"frustrated", "sad", "fearful"}:
        return Urgency.MEDIUM
    return Urgency.LOW


def _extract_entities(text: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    amounts = _AMOUNT_RE.findall(text)
    if amounts:
        entities["amount"] = amounts[0]
    order_match = _ORDER_ID_RE.search(text)
    if order_match:
        entities["reference_id"] = order_match.group(1)
    return entities


_ACTION_MAP: dict[str, list[str]] = {
    "duplicate_charge": ["verify duplicate transaction", "initiate refund"],
    "refund_request": ["verify eligibility", "initiate refund"],
    "fraud_report": ["freeze account activity", "escalate to fraud team"],
    "account_access": ["verify identity", "unlock account"],
    "password_reset": ["send password reset link"],
    "subscription_cancel": ["confirm cancellation", "process cancellation"],
    "technical_issue": ["gather diagnostic details", "escalate to engineering if unresolved"],
    "delivery_issue": ["check shipment tracking", "offer reshipment or refund"],
    "billing_issue": ["review billing history"],
    "general_question": ["answer from knowledge base"],
}


def _summarize(text: str) -> str:
    """Short extractive summary: first sentence, truncated. Deterministic,
    not an LLM call — cheap and avoids hallucination for this structured field."""
    first_sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    return first_sentence[:200]


def analyze(text: str) -> NLPAnalysis:
    intent_result = predict_intent(text)
    emotion_result = predict_emotion(text)

    return NLPAnalysis(
        intent=intent_result["intent"],
        intent_confidence=intent_result["confidence"],
        sentiment=_compute_sentiment(text),
        emotion=emotion_result["emotion"],
        emotion_confidence=emotion_result["confidence"],
        urgency=_compute_urgency(intent_result["intent"], emotion_result["emotion"], text),
        entities=_extract_entities(text),
        action_items=_ACTION_MAP.get(intent_result["intent"], []),
        summary=_summarize(text),
    )

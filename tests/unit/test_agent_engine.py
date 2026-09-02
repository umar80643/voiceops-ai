from __future__ import annotations

from apps.agent.engine import DecisionEngine
from shared.schemas.nlp import NLPAnalysis, Sentiment, Urgency


def _analysis(**overrides: object) -> NLPAnalysis:
    defaults = {
        "intent": "duplicate_charge",
        "intent_confidence": 0.9,
        "sentiment": Sentiment.NEGATIVE,
        "emotion": "frustrated",
        "emotion_confidence": 0.9,
        "urgency": Urgency.HIGH,
        "entities": {},
        "action_items": [],
        "summary": "",
    }
    defaults.update(overrides)
    return NLPAnalysis(**defaults)


def test_fraud_always_escalates() -> None:
    decision = DecisionEngine().decide(
        _analysis(intent="fraud_report", urgency=Urgency.CRITICAL), "U1", 0
    )
    assert decision.decision == "escalate"
    assert decision.requires_escalation is True


def test_low_confidence_asks_for_clarification() -> None:
    decision = DecisionEngine().decide(_analysis(intent_confidence=0.2), "U1", 0)
    assert decision.decision == "clarify"


def test_duplicate_charge_with_many_prior_tickets_is_high_priority() -> None:
    decision = DecisionEngine().decide(_analysis(), "U1", prior_open_tickets=3)
    assert "high" in decision.reason
    assert any(tc.tool == "create_support_ticket" for tc in decision.tool_calls)


def test_account_access_never_refunds() -> None:
    decision = DecisionEngine().decide(_analysis(intent="account_access"), "U1", 0)
    assert decision.decision == "account_recovery"
    assert decision.tool_calls == []

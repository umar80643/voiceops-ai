"""End-to-end scenario tests (Phase 24 / Phase 34 demo scenarios).

Each scenario runs the full pipeline: NLP analysis -> agent decision
(retrieval + tools) exactly as the API does, and asserts the expected
outcome described in the build spec.
"""

from __future__ import annotations

import pytest
from apps.agent.engine import run_agent
from apps.rag.service import get_retriever
from apps.speech_intelligence.nlp import analyze
from shared.database.session import make_engine, make_session_factory, seed_demo_data
from shared.schemas.nlp import Urgency
from sqlalchemy.orm import Session


@pytest.fixture(autouse=True, scope="module")
def _ingest_kb() -> None:
    from pathlib import Path

    if not get_retriever()._chunks:  # noqa: SLF001
        get_retriever().ingest_directory(Path("data/knowledge_base"))


@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)
    s = factory()
    seed_demo_data(s)
    yield s
    s.close()


def test_scenario_1_duplicate_charge(session: Session) -> None:
    text = "I was charged twice for my subscription and I've already contacted support three times."
    analysis = analyze(text)
    assert analysis.intent == "duplicate_charge"
    assert analysis.emotion in {"frustrated", "angry"}
    assert analysis.urgency in {Urgency.HIGH, Urgency.CRITICAL}

    result = run_agent(session, text, analysis, "U123")
    assert result.decision.decision == "refund"
    assert any(
        r["tool"] == "detect_duplicate_charge" and r["data"]["duplicate_found"]
        for r in result.tool_results
    )


def test_scenario_2_password_problem(session: Session) -> None:
    text = "I forgot my password and can't log into my account."
    analysis = analyze(text)
    result = run_agent(session, text, analysis, "U123")
    assert analysis.intent in {"password_reset", "account_access"}
    assert result.decision.decision == "account_recovery"
    assert result.decision.tool_calls == []  # no refund tool ever called


def test_scenario_3_fraud(session: Session) -> None:
    text = "Someone used my card without my permission, I think this is fraud."
    analysis = analyze(text)
    result = run_agent(session, text, analysis, "U123")
    assert analysis.intent == "fraud_report"
    assert result.decision.requires_escalation is True
    assert result.decision.decision == "escalate"


def test_scenario_4_technical_problem(session: Session) -> None:
    text = "The app keeps crashing every time I open it on my phone."
    analysis = analyze(text)
    result = run_agent(session, text, analysis, "U123")
    assert analysis.intent == "technical_issue"
    assert result.decision.decision == "troubleshoot"


def test_scenario_5_unknown_request_does_not_hallucinate(session: Session) -> None:
    text = "xk29 ping status wobble unrelated gibberish"
    analysis = analyze(text)
    result = run_agent(session, text, analysis, "U123")
    # System must not confidently invent an answer for a low-confidence/unclear request.
    assert result.decision.decision in {"clarify", "answer_from_knowledge_base"}
    if result.decision.decision == "answer_from_knowledge_base":
        assert analysis.intent_confidence >= 0.6

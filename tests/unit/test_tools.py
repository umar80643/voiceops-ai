from __future__ import annotations

import pytest
from apps.agent.tools import execute_tool
from shared.database.session import make_engine, make_session_factory, seed_demo_data
from sqlalchemy.orm import Session


@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)
    s = factory()
    seed_demo_data(s)
    yield s
    s.close()


def test_get_customer(session: Session) -> None:
    result = execute_tool("get_customer", session, customer_id="U123")
    assert result.success
    assert result.data["customer_id"] == "U123"


def test_get_customer_not_found(session: Session) -> None:
    result = execute_tool("get_customer", session, customer_id="does-not-exist")
    assert not result.success


def test_detect_duplicate_charge_finds_duplicate(session: Session) -> None:
    result = execute_tool("detect_duplicate_charge", session, customer_id="U123")
    assert result.success
    assert result.data["duplicate_found"] is True


def test_create_refund_is_idempotent(session: Session) -> None:
    r1 = execute_tool(
        "create_refund", session, customer_id="U123", transaction_id="T1", amount=29.99
    )
    r2 = execute_tool(
        "create_refund", session, customer_id="U123", transaction_id="T1", amount=29.99
    )
    assert r1.data["refund_id"] == r2.data["refund_id"]


def test_unregistered_tool_denied(session: Session) -> None:
    result = execute_tool("delete_everything", session)
    assert not result.success
    assert "not registered" in result.error

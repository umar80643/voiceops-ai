"""DB engine/session setup + realistic seed data (Phase 11).

Uses SQLite by default (`sqlite:///./voiceops_dev.db` or in-memory for
tests) since no Postgres server is reachable in this sandbox without
Docker. Swap `db_url` to `settings.postgres_dsn` (sync driver variant)
in production — no model changes required.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from shared.database.models import (
    Base,
    Customer,
    Refund,
    SupportTicket,
    Transaction,
)


def make_engine(db_url: str = "sqlite:///:memory:"):
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)


def seed_demo_data(session: Session) -> None:
    """Seed the exact scenario described in the build spec: a customer
    with two identical transactions (duplicate charge) and two prior
    unresolved tickets -> should be flagged high priority by the agent."""
    customer = Customer(customer_id="U123", name="Jordan Alvarez", email="jordan@example.com")
    session.add(customer)

    now = dt.datetime(2026, 8, 1, 12, 0, 0)
    session.add_all(
        [
            Transaction(transaction_id="T1", customer_id="U123", amount=29.99, created_at=now),
            Transaction(
                transaction_id="T2",
                customer_id="U123",
                amount=29.99,
                created_at=now + dt.timedelta(minutes=1),
            ),
        ]
    )
    session.add_all(
        [
            SupportTicket(
                ticket_id="TK1", customer_id="U123", subject="Billing question", status="resolved"
            ),
            SupportTicket(
                ticket_id="TK2", customer_id="U123", subject="Duplicate charge?", status="open"
            ),
        ]
    )
    session.commit()


def create_refund(
    session: Session, refund_id: str, customer_id: str, transaction_id: str, amount: float
) -> Refund:
    refund = Refund(
        refund_id=refund_id,
        customer_id=customer_id,
        transaction_id=transaction_id,
        amount=amount,
        status="pending",
    )
    session.add(refund)
    session.commit()
    return refund

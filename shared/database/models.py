"""Customer simulation backend schema (Phase 11).

Real SQLAlchemy models against SQLite for local dev/tests in this sandbox
(no Postgres server reachable without Docker here); the same models point
at Postgres in production via `Settings.postgres_dsn` — no ORM code
changes needed, only the connection string, since standard SQLAlchemy
Core types are used throughout (see docs/decisions).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)

    subscriptions: Mapped[list[Subscription]] = relationship(back_populates="customer")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="customer")
    tickets: Mapped[list[SupportTicket]] = relationship(back_populates="customer")


class Subscription(Base):
    __tablename__ = "subscriptions"

    subscription_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    plan: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # active | cancelled

    customer: Mapped[Customer] = relationship(back_populates="subscriptions")


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    status: Mapped[str] = mapped_column(String)


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    amount: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)

    customer: Mapped[Customer] = relationship(back_populates="transactions")


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    ticket_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    subject: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # open | resolved | escalated

    customer: Mapped[Customer] = relationship(back_populates="tickets")


class Refund(Base):
    __tablename__ = "refunds"

    refund_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.transaction_id"))
    amount: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String)  # pending | completed


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    transcript: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)

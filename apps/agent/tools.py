"""Tool-calling abstraction (Phase 10).

Real tools backed by the simulated Postgres/SQLite database (Phase 11).
Tools never let the agent mutate the DB directly — each tool validates
arguments and performs one bounded, auditable operation. Idempotency is
enforced by deterministic IDs (refund_id derived from transaction_id).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from shared.database.models import Customer, Refund, SupportTicket, Transaction
from shared.database.session import create_refund
from shared.logging import get_logger
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = get_logger(__name__)


@dataclass
class ToolResult:
    tool: str
    success: bool
    data: dict
    error: str | None = None


class Tool(Protocol):
    name: str

    def execute(self, session: Session, **kwargs: object) -> ToolResult: ...


class GetCustomerTool:
    name = "get_customer"

    def execute(self, session: Session, **kwargs: object) -> ToolResult:
        customer_id = str(kwargs["customer_id"])
        customer = session.get(Customer, customer_id)
        if customer is None:
            return ToolResult(self.name, False, {}, error=f"customer '{customer_id}' not found")
        return ToolResult(
            self.name, True, {"customer_id": customer.customer_id, "name": customer.name}
        )


class GetTransactionsTool:
    name = "get_transactions"

    def execute(self, session: Session, **kwargs: object) -> ToolResult:
        customer_id = str(kwargs["customer_id"])
        rows = session.scalars(
            select(Transaction).where(Transaction.customer_id == customer_id)
        ).all()
        return ToolResult(
            self.name,
            True,
            {
                "transactions": [
                    {
                        "transaction_id": t.transaction_id,
                        "amount": t.amount,
                        "created_at": t.created_at.isoformat(),
                    }
                    for t in rows
                ]
            },
        )


class DetectDuplicateChargeTool:
    """Real duplicate-detection logic: same amount within a short window."""

    name = "detect_duplicate_charge"

    def execute(self, session: Session, **kwargs: object) -> ToolResult:
        customer_id = str(kwargs["customer_id"])
        rows = sorted(
            session.scalars(
                select(Transaction).where(Transaction.customer_id == customer_id)
            ).all(),
            key=lambda t: t.created_at,
        )
        for i in range(1, len(rows)):
            if rows[i].amount == rows[i - 1].amount:
                delta = (rows[i].created_at - rows[i - 1].created_at).total_seconds()
                if delta <= 3600:  # within 1 hour
                    return ToolResult(
                        self.name,
                        True,
                        {
                            "duplicate_found": True,
                            "transaction_ids": [rows[i - 1].transaction_id, rows[i].transaction_id],
                            "amount": rows[i].amount,
                        },
                    )
        return ToolResult(self.name, True, {"duplicate_found": False})


class CreateRefundTool:
    name = "create_refund"

    def execute(self, session: Session, **kwargs: object) -> ToolResult:
        customer_id = str(kwargs["customer_id"])
        transaction_id = str(kwargs["transaction_id"])
        amount = float(kwargs["amount"])  # type: ignore[arg-type]

        transaction = session.get(Transaction, transaction_id)
        if transaction is None or transaction.customer_id != customer_id:
            return ToolResult(self.name, False, {}, error="transaction not found for customer")

        # Idempotent: check for an existing refund first rather than relying
        # on the DB to reject a duplicate insert with an IntegrityError —
        # returns the existing refund so repeated calls are safe no-ops.
        refund_id = f"refund-{transaction_id}"
        existing = session.get(Refund, refund_id)
        if existing is not None:
            return ToolResult(
                self.name, True, {"refund_id": existing.refund_id, "status": existing.status}
            )
        refund = create_refund(session, refund_id, customer_id, transaction_id, amount)
        return ToolResult(self.name, True, {"refund_id": refund.refund_id, "status": refund.status})


class CreateSupportTicketTool:
    name = "create_support_ticket"

    def execute(self, session: Session, **kwargs: object) -> ToolResult:
        customer_id = str(kwargs["customer_id"])
        subject = str(kwargs["subject"])
        ticket_id = f"ticket-{uuid.uuid4().hex[:8]}"
        ticket = SupportTicket(
            ticket_id=ticket_id, customer_id=customer_id, subject=subject, status="open"
        )
        session.add(ticket)
        session.commit()
        return ToolResult(self.name, True, {"ticket_id": ticket.ticket_id, "status": ticket.status})


class EscalateTicketTool:
    name = "escalate_ticket"

    def execute(self, session: Session, **kwargs: object) -> ToolResult:
        ticket_id = str(kwargs["ticket_id"])
        ticket = session.get(SupportTicket, ticket_id)
        if ticket is None:
            return ToolResult(self.name, False, {}, error="ticket not found")
        ticket.status = "escalated"
        session.commit()
        return ToolResult(self.name, True, {"ticket_id": ticket.ticket_id, "status": ticket.status})


_all_tools: list[Any] = [
    GetCustomerTool(),
    GetTransactionsTool(),
    DetectDuplicateChargeTool(),
    CreateRefundTool(),
    CreateSupportTicketTool(),
    EscalateTicketTool(),
]
TOOL_REGISTRY: dict[str, Any] = {t.name: t for t in _all_tools}


def execute_tool(name: str, session: Session, **kwargs: object) -> ToolResult:
    """Authorization gate: only tools present in TOOL_REGISTRY can ever run —
    prevents arbitrary tool execution from LLM-generated tool names."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        logger.warning("tool_execution_denied", tool=name)
        return ToolResult(name, False, {}, error=f"tool '{name}' is not registered")
    try:
        result = tool.execute(session, **kwargs)
        logger.info("tool_executed", tool=name, success=result.success)
        return result
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("tool_execution_failed", tool=name, error=str(exc))
        return ToolResult(name, False, {}, error=f"invalid arguments: {exc}")

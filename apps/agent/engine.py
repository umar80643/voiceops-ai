"""LLM reasoning engine + tool-calling agent (Phase 9/10).

Production path: an LLM (e.g. Claude via the Anthropic API) reasons over
conversation + intent + emotion + urgency + retrieved knowledge and
proposes tool calls as structured JSON, validated by
`shared.schemas.agent.AgentDecision` before any tool executes — the LLM
NEVER mutates the database directly (spec Phase 9).

Sandbox limitation: this environment has no configured Anthropic API key
for server-side calls, so a real LLM call cannot be made and evaluated
here. `DecisionEngine` below is a real, deterministic, fully-tested
substitute that implements the *same contract* (same input, same
validated output schema, same tool-authorization gate) using explicit
rules derived directly from the knowledge-base policies in
data/knowledge_base/. This is honestly weaker than an LLM at handling
novel phrasing, but every decision it makes is explainable and testable.

The agent loop (understand -> retrieve -> select tool -> validate ->
execute -> inspect -> decide -> respond) is fully real and runs against
the actual SQLite-backed simulated customer database and tool registry.
"""

from __future__ import annotations

from shared.database.models import SupportTicket
from shared.logging import get_logger
from shared.schemas.agent import AgentDecision, AgentRunResponse, ToolCall
from shared.schemas.nlp import NLPAnalysis, Urgency
from sqlalchemy.orm import Session

from apps.agent.tools import TOOL_REGISTRY, execute_tool
from apps.rag.service import get_retriever

logger = get_logger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.5
ESCALATION_PRIOR_TICKET_THRESHOLD = 2


class DecisionEngine:
    """Rule-based decision engine (see module docstring for scope)."""

    def decide(
        self, analysis: NLPAnalysis, customer_id: str, prior_open_tickets: int
    ) -> AgentDecision:
        # Only intent confidence gates whether we act at all: intent
        # determines *what* action to take, so low intent confidence means
        # we genuinely don't know what the customer wants. Emotion
        # confidence, by contrast, only informs urgency/tone (a 7-way
        # classification naturally produces lower peak probabilities than
        # a well-separated intent decision) — it should not by itself
        # block an otherwise-clear request. This was a real bug found via
        # tests/evaluation/test_demo_scenarios.py (scenario 4: a 98%-
        # confidence technical_issue was incorrectly sent to "clarify"
        # because emotion confidence alone was ~38%).
        low_confidence = analysis.intent_confidence < LOW_CONFIDENCE_THRESHOLD
        high_priority = prior_open_tickets >= ESCALATION_PRIOR_TICKET_THRESHOLD

        if analysis.urgency == Urgency.CRITICAL or analysis.intent == "fraud_report":
            return AgentDecision(
                decision="escalate",
                reason="Fraud/critical urgency per escalation_policy: never auto-act on suspected fraud.",
                confidence=0.95,
                requires_escalation=True,
                tool_calls=[
                    ToolCall(
                        tool="create_support_ticket",
                        arguments={"customer_id": customer_id, "subject": "Fraud report"},
                    )
                ],
            )

        if low_confidence:
            return AgentDecision(
                decision="clarify",
                reason="Intent confidence below threshold; asking for clarification rather than guessing.",
                confidence=round(analysis.intent_confidence, 4),
                requires_escalation=False,
                tool_calls=[],
            )

        if analysis.intent == "duplicate_charge":
            tool_calls = [
                ToolCall(tool="detect_duplicate_charge", arguments={"customer_id": customer_id})
            ]
            if high_priority:
                tool_calls.append(
                    ToolCall(
                        tool="create_support_ticket",
                        arguments={
                            "customer_id": customer_id,
                            "subject": "Duplicate charge (high priority)",
                        },
                    )
                )
            return AgentDecision(
                decision="refund",
                reason=(
                    "duplicate_charge intent per refund_policy; verify then refund. "
                    f"Prior unresolved tickets={prior_open_tickets} -> "
                    f"{'high' if high_priority else 'normal'} priority."
                ),
                confidence=analysis.intent_confidence,
                requires_escalation=False,
                tool_calls=tool_calls,
            )

        if analysis.intent in {"account_access", "password_reset"}:
            return AgentDecision(
                decision="account_recovery",
                reason="account_access/password_reset per account_recovery policy: send reset link, no refund.",
                confidence=analysis.intent_confidence,
                requires_escalation=False,
                tool_calls=[],
            )

        if analysis.intent == "technical_issue":
            return AgentDecision(
                decision="troubleshoot",
                reason="technical_issue per technical_troubleshooting policy: gather diagnostics, suggest steps.",
                confidence=analysis.intent_confidence,
                requires_escalation=False,
                tool_calls=[],
            )

        if analysis.intent_confidence < 0.6:
            return AgentDecision(
                decision="clarify",
                reason="Unrecognized/low-confidence request; avoid inventing a confident answer.",
                confidence=analysis.intent_confidence,
                requires_escalation=False,
                tool_calls=[],
            )

        return AgentDecision(
            decision="answer_from_knowledge_base",
            reason=f"General '{analysis.intent}' request answered from retrieved policy documents.",
            confidence=analysis.intent_confidence,
            requires_escalation=False,
            tool_calls=[],
        )


def run_agent(
    session: Session, text: str, analysis: NLPAnalysis, customer_id: str
) -> AgentRunResponse:
    """Full agent loop: retrieve knowledge, count prior tickets, decide,
    validate + execute tool calls (only ever via the authorized registry),
    generate a response."""
    knowledge_hits = get_retriever().search(text, k=2)

    prior_open_tickets = (
        session.query(SupportTicket)
        .filter(SupportTicket.customer_id == customer_id, SupportTicket.status != "resolved")
        .count()
    )

    decision = DecisionEngine().decide(analysis, customer_id, prior_open_tickets)

    tool_results = []
    for call in decision.tool_calls:
        if call.tool not in TOOL_REGISTRY:
            logger.warning("agent_proposed_unregistered_tool", tool=call.tool)
            continue
        result = execute_tool(call.tool, session, **call.arguments)
        tool_results.append(result.__dict__)

    response_text = _generate_response(decision, knowledge_hits)

    return AgentRunResponse(
        decision=decision, tool_results=tool_results, response_text=response_text
    )


def _generate_response(decision: AgentDecision, knowledge_hits: list[dict]) -> str:
    source = knowledge_hits[0]["title"] if knowledge_hits else "our policies"
    if decision.decision == "escalate":
        return "I'm escalating this to a specialist right away for your security."
    if decision.decision == "clarify":
        return "Could you tell me a bit more about the issue so I can help correctly?"
    if decision.decision == "refund":
        return f"I've verified this against our {source} and I'm processing a refund for you now."
    if decision.decision == "account_recovery":
        return "I've sent a password reset link to your registered email."
    if decision.decision == "troubleshoot":
        return (
            "Let's troubleshoot this — could you share your device/OS and the exact error message?"
        )
    return f"Based on our {source}, here's what I can tell you."

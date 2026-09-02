from __future__ import annotations

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool: str
    arguments: dict


class AgentDecision(BaseModel):
    decision: str
    reason: str
    confidence: float
    requires_escalation: bool
    tool_calls: list[ToolCall]


class AgentRunResponse(BaseModel):
    decision: AgentDecision
    tool_results: list[dict]
    response_text: str

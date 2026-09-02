from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

from apps.agent.engine import run_agent
from apps.rag.service import get_retriever
from apps.speech_intelligence.nlp import analyze
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shared.database.models import Conversation
from shared.database.session import make_engine, make_session_factory, seed_demo_data
from shared.logging import get_logger
from shared.schemas.agent import AgentRunResponse
from shared.schemas.nlp import NLPAnalysis
from sqlalchemy.orm import Session

router = APIRouter(tags=["conversations"])
logger = get_logger(__name__)

# Module-level engine/session: a single shared SQLite dev DB per process.
# Production would inject a Postgres-backed session per request (FastAPI
# dependency) instead of this module-global — kept simple here since
# Phase 11's DB is a local simulation, not a production connection pool.
_engine = make_engine("sqlite:///./voiceops_dev.db")
_SessionFactory = make_session_factory(_engine)


def _get_session() -> Session:
    return _SessionFactory()


def ensure_seeded() -> None:
    with _get_session() as session:
        existing = session.query(Conversation).count()  # cheap probe
        del existing
        from shared.database.models import Customer  # noqa: PLC0415

        if session.get(Customer, "U123") is None:
            seed_demo_data(session)


class ConversationCreateRequest(BaseModel):
    customer_id: str
    text: str


class ConversationResponse(BaseModel):
    conversation_id: str
    customer_id: str
    text: str
    analysis: NLPAnalysis
    agent: AgentRunResponse


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(req: ConversationCreateRequest) -> ConversationResponse:
    ensure_seeded()
    if not get_retriever()._chunks:  # noqa: SLF001 (internal check, avoid re-ingesting every call)
        get_retriever().ingest_directory(Path("data/knowledge_base"))

    analysis = analyze(req.text)

    with _get_session() as session:
        agent_response = run_agent(session, req.text, analysis, req.customer_id)

        conversation_id = str(uuid.uuid4())
        session.add(
            Conversation(
                conversation_id=conversation_id,
                customer_id=req.customer_id,
                transcript=req.text,
                created_at=dt.datetime.now(dt.UTC),
            )
        )
        session.commit()

    logger.info(
        "conversation_processed",
        conversation_id=conversation_id,
        intent=analysis.intent,
        decision=agent_response.decision.decision,
    )

    return ConversationResponse(
        conversation_id=conversation_id,
        customer_id=req.customer_id,
        text=req.text,
        analysis=analysis,
        agent=agent_response,
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    with _get_session() as session:
        convo = session.get(Conversation, conversation_id)
        if convo is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {
            "conversation_id": convo.conversation_id,
            "customer_id": convo.customer_id,
            "transcript": convo.transcript,
            "created_at": convo.created_at.isoformat(),
        }

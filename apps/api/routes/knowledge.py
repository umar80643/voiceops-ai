from __future__ import annotations

from pathlib import Path

from apps.rag.service import get_retriever
from fastapi import APIRouter
from shared.schemas.rag import RAGSearchResponse, RetrievedDocument

router = APIRouter(prefix="/knowledge", tags=["rag"])


@router.post("/ingest")
async def ingest() -> dict:
    n = get_retriever().ingest_directory(Path("data/knowledge_base"))
    return {"chunks_ingested": n}


@router.post("/search", response_model=RAGSearchResponse)
async def search(query: str, k: int = 3) -> RAGSearchResponse:
    results = get_retriever().search(query, k=k)
    return RAGSearchResponse(query=query, results=[RetrievedDocument(**r) for r in results])

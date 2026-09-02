from __future__ import annotations

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str
    score: float
    excerpt: str


class RAGSearchResponse(BaseModel):
    query: str
    results: list[RetrievedDocument]

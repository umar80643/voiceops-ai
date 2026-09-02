from __future__ import annotations

from pathlib import Path

import pytest
from apps.rag.service import TfidfRetriever


@pytest.fixture
def retriever() -> TfidfRetriever:
    r = TfidfRetriever()
    r.ingest_directory(Path("data/knowledge_base"))
    return r


def test_ingest_returns_chunks(retriever: TfidfRetriever) -> None:
    assert len(retriever._chunks) > 0  # noqa: SLF001


def test_search_fraud_query_hits_fraud_policy(retriever: TfidfRetriever) -> None:
    results = retriever.search("someone used my card without my permission", k=3)
    doc_ids = [r["doc_id"] for r in results]
    assert "fraud_policy" in doc_ids


def test_search_before_ingest_raises() -> None:
    r = TfidfRetriever()
    with pytest.raises(RuntimeError):
        r.search("anything")

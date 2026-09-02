"""RAG retrieval (Phase 8).

Production target: Qdrant vector DB with a sentence-embedding model
(see ADR in docs/decisions). Qdrant is provisioned in docker-compose but
this sandbox has no network access to download a sentence-transformers
embedding model from huggingface.co, so a real embedding-based Qdrant
pipeline can't be exercised here.

What IS implemented and real: chunking of the knowledge base documents,
and a TF-IDF + cosine-similarity retriever (scikit-learn) that serves the
same interface (`ingest`, `search`) a Qdrant-backed implementation would.
This is a legitimate sparse-retrieval baseline (comparable in spirit to
BM25), not a stand-in that pretends to be semantic embeddings — it's
labeled `retriever="tfidf_cosine"` throughout. `apps/rag/qdrant_backend.py`
documents the production embedding+Qdrant integration as unexecuted code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE_CHARS = 500
CHUNK_OVERLAP_CHARS = 50


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str


def _chunk_text(doc_id: str, title: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE_CHARS, len(text))
        chunks.append(Chunk(doc_id=doc_id, title=title, text=text[start:end].strip()))
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP_CHARS
    return [c for c in chunks if c.text]


class TfidfRetriever:
    """In-memory TF-IDF retriever implementing the RAG `ingest`/`search` interface."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def ingest_directory(self, directory: str | Path) -> int:
        directory = Path(directory)
        self._chunks = []
        for path in sorted(directory.glob("*.md")):
            title = path.stem.replace("_", " ").title()
            self._chunks.extend(_chunk_text(path.stem, title, path.read_text()))

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])
        return len(self._chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self._vectorizer is None or self._matrix is None or not self._chunks:
            raise RuntimeError("Knowledge base not ingested yet. Call ingest_directory() first.")
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = scores.argsort()[::-1][:k]
        return [
            {
                "doc_id": self._chunks[i].doc_id,
                "title": self._chunks[i].title,
                "score": round(float(scores[i]), 4),
                "excerpt": self._chunks[i].text[:200],
            }
            for i in ranked_idx
            if scores[i] > 0
        ]


_retriever = TfidfRetriever()


def get_retriever() -> TfidfRetriever:
    return _retriever

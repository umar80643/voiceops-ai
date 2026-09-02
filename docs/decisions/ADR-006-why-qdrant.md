# ADR-006 — Why Qdrant

## Context
RAG retrieval over the policy knowledge base needs a vector database.

## Decision
Qdrant is the target production vector store.

## Alternatives considered
- Pinecone — managed/paid SaaS, less portable for a self-hostable demo.
- pgvector — simpler (reuses Postgres) but weaker filtering/performance at the scale this architecture targets.
- FAISS — excellent for pure vector search, but no built-in server/API, harder to operate as an independent service.

## Trade-offs
Qdrant is open-source, exposes a simple REST/gRPC API, and is easy to self-host alongside the rest of the stack (see `docker-compose.yml`). It still requires a real embedding model to be useful.

## Consequences
NOT executed in this build: real semantic embeddings require downloading a sentence-transformers model, unavailable via Hugging Face Hub in this sandbox. `apps/rag/service.py` instead implements a real, measured TF-IDF + cosine-similarity retriever (a legitimate sparse-retrieval baseline) against the actual knowledge base documents, scoring Recall@3 = 1.0 and MRR = 0.9167 on a small hand-labeled query set (`ml/evaluation/rag.py`, `docs/rag_eval_report.json`). The Qdrant + real-embeddings path is documented, not executed or claimed to work.

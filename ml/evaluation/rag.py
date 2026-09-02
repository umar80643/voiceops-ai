"""RAG evaluation (Phase 8/21).

Hand-labeled query -> expected-document pairs, evaluated for real against
`TfidfRetriever` over the actual knowledge base in data/knowledge_base/.

Run: python -m ml.evaluation.rag
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.rag.service import TfidfRetriever

EVAL_SET: list[tuple[str, str]] = [
    ("I was charged twice for my subscription", "refund_policy"),
    ("someone used my card without permission", "fraud_policy"),
    ("I forgot my password and can't log in", "account_recovery"),
    ("the app keeps crashing on my phone", "technical_troubleshooting"),
    ("I want to cancel my membership", "subscription_policy"),
    ("my package says delivered but never arrived", "shipping_policy"),
    ("my bill amount looks different than expected", "billing_policy"),
    ("this needs to be escalated to a human", "escalation_policy"),
]

K = 3


def evaluate() -> dict:
    retriever = TfidfRetriever()
    n_chunks = retriever.ingest_directory(Path("data/knowledge_base"))

    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    details = []

    for query, expected_doc in EVAL_SET:
        results = retriever.search(query, k=K)
        doc_ids = [r["doc_id"] for r in results]
        hit = expected_doc in doc_ids
        hits_at_k += int(hit)
        rank = doc_ids.index(expected_doc) + 1 if hit else 0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        details.append({"query": query, "expected": expected_doc, "retrieved": doc_ids, "hit": hit})

    report = {
        "retriever": "tfidf_cosine",
        "n_chunks_ingested": n_chunks,
        "k": K,
        "n_queries": len(EVAL_SET),
        f"recall_at_{K}": round(hits_at_k / len(EVAL_SET), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "details": details,
    }

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "rag_eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate()

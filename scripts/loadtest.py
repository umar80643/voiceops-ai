"""Load testing (Phase 25).

Run against a locally running API (`make run` in another terminal):

    locust -f scripts/loadtest.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 1m --headless

This exercises /conversations end-to-end (NLP + RAG + agent + tool
execution against the real SQLite-backed simulated DB). Results (P50/P95/
P99, throughput, failure rate) are printed by Locust's own summary; no
numbers are fabricated here — run the command above yourself and record
the output in docs/evaluation.md.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

_SAMPLE_MESSAGES = [
    "I was charged twice for my subscription and I've already contacted support three times.",
    "I forgot my password and can't log into my account.",
    "Someone used my card without my permission, I think this is fraud.",
    "The app keeps crashing every time I open it on my phone.",
    "I want to cancel my subscription please.",
    "What are your business hours?",
]


class VoiceOpsUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def process_conversation(self) -> None:
        self.client.post(
            "/conversations",
            json={"customer_id": "U123", "text": random.choice(_SAMPLE_MESSAGES)},
        )

    @task(1)
    def search_knowledge(self) -> None:
        self.client.post("/knowledge/search", params={"query": random.choice(_SAMPLE_MESSAGES)})

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health")

from __future__ import annotations

from apps.api.main import app
from fastapi.testclient import TestClient


def test_full_conversation_flow_through_api() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/conversations",
            json={
                "customer_id": "U123",
                "text": "I was charged twice for my subscription and I've already contacted support three times.",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["analysis"]["intent"] == "duplicate_charge"
        assert body["agent"]["decision"]["decision"] == "refund"

        get_resp = client.get(f"/conversations/{body['conversation_id']}")
        assert get_resp.status_code == 200
        assert get_resp.json()["customer_id"] == "U123"


def test_knowledge_search_via_api() -> None:
    with TestClient(app) as client:
        client.post("/knowledge/ingest")
        resp = client.post("/knowledge/search", params={"query": "cancel my subscription"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert any(r["doc_id"] == "subscription_policy" for r in results)


def test_audio_upload_and_transcribe_flow() -> None:
    import io
    import wave

    import numpy as np

    t = np.linspace(0, 1.0, 16000, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    buf.seek(0)

    with TestClient(app) as client:
        upload_resp = client.post("/audio/upload", files={"file": ("test.wav", buf, "audio/wav")})
        assert upload_resp.status_code == 200
        audio_id = upload_resp.json()["audio_id"]
        assert upload_resp.json()["status"] == "accepted"

        transcribe_resp = client.post(f"/transcribe/{audio_id}")
        assert transcribe_resp.status_code == 200
        # No real ASR model available in this sandbox (see apps/asr/service.py) —
        # falls back to NullASR, which must return an explicit empty transcript,
        # never a fabricated one.
        assert transcribe_resp.json()["text"] == ""

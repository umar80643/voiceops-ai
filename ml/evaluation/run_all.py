"""Aggregate evaluation runner (Phase 21). Run: `make evaluate` or
`python -m ml.evaluation.run_all`.

Runs every real, implemented evaluation in this repo and prints a
consolidated summary. Does NOT run ASR evaluation (no HF Hub access in
this build environment — see docs/evaluation.md) and does not fabricate
a result for it.
"""

from __future__ import annotations

import json

from ml.evaluation.rag import evaluate as evaluate_rag
from ml.training.train_emotion import main as train_emotion
from ml.training.train_fusion import main as train_fusion
from ml.training.train_intent import main as train_intent


def main() -> None:
    print("=" * 60)
    print("VoiceOps AI — Evaluation Suite")
    print("=" * 60)

    print("\n[1/4] Intent classifier (synthetic text data)...")
    intent_report = train_intent()

    print("\n[2/4] Emotion classifier (synthetic text data)...")
    emotion_report = train_emotion()

    print("\n[3/4] Multimodal fusion comparison...")
    fusion_report = train_fusion()

    print("\n[4/4] RAG retrieval evaluation...")
    rag_report = evaluate_rag()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = {
        "intent_fine_tuned_macro_f1": intent_report["fine_tuned"]["test_metrics"]["macro_f1"],
        "emotion_macro_f1": emotion_report["metrics"]["macro_f1"],
        "multimodal_improvement_macro_f1": fusion_report["multimodal_improvement_macro_f1"],
        "rag_recall_at_3": rag_report["recall_at_3"],
        "rag_mrr": rag_report["mrr"],
        "note": "ASR evaluation NOT included — no Hugging Face Hub network access "
        "in this build environment. See docs/evaluation.md.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

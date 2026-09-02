"""Multimodal fusion for escalation prediction (Phase 6).

Combines a text representation (TF-IDF over the transcript) with simple
numeric acoustic features (silence_ratio, mean_amplitude, duration_seconds
— the real features Phase 1's `ml/preprocessing/audio.py` actually
produces) via a fusion layer, and predicts whether a conversation should
escalate.

Sandbox scope: full learned audio embeddings (wav2vec2) aren't available
here (no HF/GPU access), so "audio embedding" is honestly limited to the
three numeric signal-level features above rather than a learned encoder.
The comparison below (unimodal text-only vs multimodal text+audio) is
still real and measured — it demonstrates the fusion *mechanism*
(concatenate → classify) even though the audio side is intentionally
simple. `ml/training/train_fusion_audio_embeddings.py` documents the
production wav2vec2-embedding version as unexecuted future work.

Run: python -m ml.training.train_fusion
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ml.datasets.synthetic_emotion import generate_dataset as gen_emotion

ARTIFACT_DIR = Path("models/emotion")  # fusion model lives alongside emotion artifacts


def _synthesize_multimodal(seed: int = 7) -> tuple[list[str], np.ndarray, list[int]]:
    """Build a synthetic multimodal escalation dataset.

    Escalation label is TRUE when emotion is angry/frustrated/urgent AND
    the synthesized "audio" signal (low silence_ratio + higher amplitude,
    simulating raised/continuous speech) crosses a threshold — i.e. the
    label genuinely depends on both modalities, so a text-only model is
    expected to underperform a fused model on the audio-dependent cases.
    """
    rng = random.Random(seed)
    data = gen_emotion(n_per_class=40, seed=seed)
    texts, audio_feats, labels = [], [], []
    escalating_emotions = {"angry", "frustrated", "urgent"}

    for text, emotion in data:
        if emotion in escalating_emotions:
            silence_ratio = rng.uniform(0.0, 0.15)
            mean_amp = rng.uniform(0.4, 0.9)
        else:
            silence_ratio = rng.uniform(0.2, 0.6)
            mean_amp = rng.uniform(0.05, 0.35)
        duration = rng.uniform(10, 120)

        # Label depends on BOTH text-implied emotion and the acoustic proxy —
        # add noise so pure text or pure audio alone can't perfectly solve it.
        acoustic_escalating = mean_amp > 0.5 and silence_ratio < 0.2
        text_escalating = emotion in escalating_emotions
        escalate = int(
            (acoustic_escalating and text_escalating) or (text_escalating and rng.random() < 0.3)
        )

        texts.append(text)
        audio_feats.append([silence_ratio, mean_amp, duration])
        labels.append(escalate)

    return texts, np.array(audio_feats), labels


def main() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    texts, audio_feats, labels = _synthesize_multimodal()

    idx = np.arange(len(texts))
    idx_train, idx_test = train_test_split(idx, test_size=0.25, random_state=42, stratify=labels)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    text_vec_train = vectorizer.fit_transform([texts[i] for i in idx_train])
    text_vec_test = vectorizer.transform([texts[i] for i in idx_test])

    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]

    # --- Unimodal (text only) ---
    text_model = LogisticRegression(max_iter=1000)
    text_model.fit(text_vec_train, y_train)
    text_preds = text_model.predict(text_vec_test)

    # --- Multimodal (text + audio features fused via concatenation) ---
    audio_train = audio_feats[idx_train]
    audio_test = audio_feats[idx_test]
    fused_train = hstack([text_vec_train, audio_train])
    fused_test = hstack([text_vec_test, audio_test])

    fusion_model = LogisticRegression(max_iter=1000)
    fusion_model.fit(fused_train, y_train)
    fusion_preds = fusion_model.predict(fused_test)

    report: dict[str, Any] = {
        "task": "escalation_prediction",
        "dataset": {"source": "synthetic", "is_real_customer_data": False, "n_test": len(y_test)},
        "unimodal_text_only": {
            "accuracy": round(accuracy_score(y_test, text_preds), 4),
            "macro_f1": round(f1_score(y_test, text_preds, average="macro"), 4),
        },
        "multimodal_text_plus_audio_features": {
            "accuracy": round(accuracy_score(y_test, fusion_preds), 4),
            "macro_f1": round(f1_score(y_test, fusion_preds, average="macro"), 4),
        },
    }
    report["multimodal_improvement_macro_f1"] = round(
        report["multimodal_text_plus_audio_features"]["macro_f1"]
        - report["unimodal_text_only"]["macro_f1"],
        4,
    )

    (ARTIFACT_DIR / "fusion_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

"""Inference for the trained intent and emotion classifiers.

Loads the pickled TF-IDF vectorizer + MLP classifier produced by
`ml/training/train_intent.py` / `train_emotion.py`. Raises a clear error
if artifacts don't exist yet rather than silently returning junk.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class ModelNotTrainedError(Exception):
    pass


def _load(artifact_dir: Path, model_filename: str) -> tuple[Any, Any]:
    vec_path = artifact_dir / "vectorizer.pkl"
    model_path = artifact_dir / model_filename
    if not vec_path.exists() or not model_path.exists():
        raise ModelNotTrainedError(
            f"No trained artifacts found in {artifact_dir}. Run the corresponding "
            f"`ml/training/train_*.py` script first."
        )
    with vec_path.open("rb") as f:
        vectorizer = pickle.load(f)  # noqa: S301
    with model_path.open("rb") as f:
        model = pickle.load(f)  # noqa: S301
    return vectorizer, model


def predict_intent(text: str, top_k: int = 3) -> dict:
    vectorizer, model = _load(Path("models/intent"), "fine_tuned_model.pkl")
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs, strict=True), key=lambda x: x[1], reverse=True)
    return {
        "intent": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 4),
        "top_k": [
            {"intent": label, "confidence": round(float(p), 4)} for label, p in ranked[:top_k]
        ],
    }


def predict_emotion(text: str) -> dict:
    vectorizer, model = _load(Path("models/emotion"), "model.pkl")
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs, strict=True), key=lambda x: x[1], reverse=True)
    return {"emotion": ranked[0][0], "confidence": round(float(ranked[0][1]), 4)}

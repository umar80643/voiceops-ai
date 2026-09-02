"""Emotion classifier training (Phase 5). See module docstring in
`ml/datasets/synthetic_emotion.py` for scope/provenance notes.

Run: python -m ml.training.train_emotion
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ml.datasets.synthetic_emotion import EMOTION_LABELS, generate_dataset

ARTIFACT_DIR = Path("models/emotion")


def main() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    texts = [t for t, _ in data]
    labels = [label for _, label in data]

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # class imbalance analysis (real, on this synthetic set it's balanced by construction)
    class_counts = {label: labels.count(label) for label in EMOTION_LABELS}

    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    model.fit(x_train_vec, y_train)
    preds = model.predict(x_test_vec)

    report = {
        "dataset": {
            "source": "synthetic (ml/datasets/synthetic_emotion.py)",
            "is_real_customer_data": False,
            "class_counts": class_counts,
            "labels": EMOTION_LABELS,
        },
        "task": "emotion (NOT sentiment; sentiment is computed separately)",
        "model": "tfidf+mlp(64)",
        "metrics": {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "macro_f1": round(f1_score(y_test, preds, average="macro"), 4),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=EMOTION_LABELS).tolist(),
        },
    }

    (ARTIFACT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    with (ARTIFACT_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    with (ARTIFACT_DIR / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

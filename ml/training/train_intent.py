"""Intent classifier training (Phase 4).

Trains and evaluates two real models on the synthetic intent dataset
(`ml/datasets/synthetic_intent.py`):

  baseline        — TF-IDF + majority-class-aware Logistic Regression
  fine_tuned      — TF-IDF + small MLP (nonlinear), analogous in role to
                    "fine-tuning a pretrained encoder" for this sandbox's
                    text-only substitute pipeline (see module docstring
                    in synthetic_intent.py for why this stands in for a
                    wav2vec2/HuBERT speech classifier here).

Metrics (accuracy, macro F1, per-class F1, confusion matrix) are computed
for real on a held-out test split — nothing here is fabricated. Run:

    python -m ml.training.train_intent
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ml.datasets.synthetic_intent import INTENT_LABELS, generate_dataset

ARTIFACT_DIR = Path("models/intent")


def _split_dataset() -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    data = generate_dataset()
    texts = [t for t, _ in data]
    labels = [label for _, label in data]

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_class_f1": {
            label: round(score, 4)
            for label, score in zip(
                INTENT_LABELS,
                f1_score(y_true, y_pred, average=None, labels=INTENT_LABELS, zero_division=0),
                strict=True,
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=INTENT_LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=INTENT_LABELS, zero_division=0, output_dict=True
        ),
    }


def main() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_dataset()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    # Baseline
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(x_train_vec, y_train)
    baseline_preds = baseline.predict(x_test_vec)
    baseline_metrics = _evaluate(y_test, list(baseline_preds))

    # "Fine-tuned" nonlinear model (text-only stand-in — see module docstring)
    fine_tuned = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    fine_tuned.fit(x_train_vec, y_train)
    val_preds = fine_tuned.predict(x_val_vec)
    _ = _evaluate(y_val, list(val_preds))  # validation checkpoint metrics
    fine_tuned_preds = fine_tuned.predict(x_test_vec)
    fine_tuned_metrics = _evaluate(y_test, list(fine_tuned_preds))

    report = {
        "dataset": {
            "source": "synthetic (ml/datasets/synthetic_intent.py)",
            "is_real_customer_data": False,
            "n_train": len(x_train),
            "n_val": len(x_val),
            "n_test": len(x_test),
            "labels": INTENT_LABELS,
        },
        "baseline": {"model": "tfidf+logistic_regression", "test_metrics": baseline_metrics},
        "fine_tuned": {"model": "tfidf+mlp(64)", "test_metrics": fine_tuned_metrics},
        "improvement_macro_f1": round(
            fine_tuned_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4
        ),
    }

    (ARTIFACT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))

    import pickle  # noqa: PLC0415

    with (ARTIFACT_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    with (ARTIFACT_DIR / "fine_tuned_model.pkl").open("wb") as f:
        pickle.dump(fine_tuned, f)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

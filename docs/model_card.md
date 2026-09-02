# Model Cards

## Intent classifier (`models/intent/`)

- **Purpose:** Classify a customer support utterance into one of 10 intents (billing_issue, duplicate_charge, refund_request, fraud_report, account_access, password_reset, subscription_cancel, technical_issue, delivery_issue, general_question).
- **Architecture:** TF-IDF (1-2 grams) + a 1-hidden-layer (64 units) MLP classifier (scikit-learn `MLPClassifier`). This is a text-based substitute for the spec's wav2vec2/HuBERT audio classifier — see ADR-002 for why.
- **Training data:** Synthetic, template-generated (`ml/datasets/synthetic_intent.py`), 600 examples total, 60/class. **Not real customer data — must never be represented as such.**
- **Training procedure:** 70/15/15 stratified train/val/test split, `sklearn.model_selection.train_test_split`, fixed seed (42).
- **Metrics:** Accuracy 1.0, Macro F1 1.0 on the synthetic test split (see `training_report.json` for the full per-class breakdown and confusion matrix).
- **Known limitations:** Synthetic templates are cleanly separable, so these metrics say nothing about real-world (noisy transcript) performance. No audio signal is used at all in this substitute pipeline.
- **Intended use:** Demo/portfolio pipeline demonstration only. Not fit for production customer-support decisions without retraining on real, representative, consented data and re-evaluating.

## Emotion classifier (`models/emotion/`)

- **Purpose:** Classify an utterance into one of 7 emotions (neutral, happy, sad, angry, frustrated, fearful, urgent) — distinct from sentiment polarity, computed separately.
- **Architecture:** Same TF-IDF + MLP(64) approach as the intent classifier, for the same sandbox-constraint reasons (no GPU/HF access for acoustic wav2vec2 embeddings — see ADR-002).
- **Training data:** Synthetic, template-generated (`ml/datasets/synthetic_emotion.py`), 420 examples, balanced by construction. **Not real customer data.**
- **Metrics:** See `training_report.json`.
- **Known limitations / failure cases:** No acoustic signal (tone, pace, volume) is used — a real angry-sounding calm sentence or a sarcastic calm-sounding angry sentence would likely be misclassified, since the model only sees text.
- **Intended use:** Demo/portfolio pipeline demonstration only.

## Multimodal fusion model (`models/emotion/fusion_report.json`)

- **Purpose:** Predict whether a conversation should escalate, combining text (TF-IDF) with 3 numeric audio features (silence_ratio, mean_amplitude, duration_seconds).
- **Architecture:** Late/concatenation fusion + Logistic Regression (see ADR-010).
- **Result:** Did not outperform the text-only model on the synthetic evaluation set (0.8937 vs 0.9251 macro F1) — reported honestly, not hidden.
- **Intended use:** Demonstrates the fusion *mechanism*; the specific audio features used are too simple to draw conclusions about multimodal fusion's general value for this task.

"""Synthetic intent-classification dataset (Phase 4).

IMPORTANT — data provenance: every example generated here is SYNTHETIC,
built from templates + slot-filling. It is NOT real customer data and
must never be represented as such (see README limitations). It exists so
the training/evaluation pipeline can be exercised end-to-end with real,
measurable metrics in an environment with no access to a licensed speech-
intent corpus.

Sandbox note: the build spec calls for an audio-based wav2vec2/HuBERT
classifier fine-tuned on speech. This sandbox has no network access to
huggingface.co and no GPU, so raw audio cannot be fetched/fine-tuned here.
This module instead generates *text* utterances for the same intent
taxonomy, and `ml/training/train_intent.py` trains a real, evaluated
text classifier on them. `ml/training/train_intent_wav2vec2.py` contains
the real, correct speech fine-tuning code for when GPU + HF Hub access
is available — it is not executed in this sandbox.
"""

from __future__ import annotations

import random

INTENT_LABELS: list[str] = [
    "billing_issue",
    "duplicate_charge",
    "refund_request",
    "fraud_report",
    "account_access",
    "password_reset",
    "subscription_cancel",
    "technical_issue",
    "delivery_issue",
    "general_question",
]

_TEMPLATES: dict[str, list[str]] = {
    "billing_issue": [
        "My bill this month looks wrong, can you check it?",
        "I think I was overcharged on my last invoice.",
        "The amount charged doesn't match what I signed up for.",
        "Why is my billing amount different from last month?",
    ],
    "duplicate_charge": [
        "I was charged twice for my subscription this month.",
        "There are two identical charges on my card from you.",
        "I see a duplicate transaction of {amount} on my statement.",
        "You billed me two times for the same order.",
    ],
    "refund_request": [
        "I would like a refund for my last order.",
        "Can you refund the payment I made yesterday?",
        "I want my money back for this purchase.",
        "Please process a refund, the product didn't arrive.",
    ],
    "fraud_report": [
        "I think someone used my card without my permission.",
        "There's a charge I never authorized on my account.",
        "My account seems to have been hacked and used for purchases.",
        "I did not make this transaction, I believe it's fraud.",
    ],
    "account_access": [
        "I can't log into my account anymore.",
        "It says my account is locked, can you help?",
        "I'm getting an error every time I try to sign in.",
        "My login keeps failing even with the right details.",
    ],
    "password_reset": [
        "I forgot my password and need to reset it.",
        "Can you send me a password reset link?",
        "The reset email never arrived, can you resend it?",
        "How do I change my password?",
    ],
    "subscription_cancel": [
        "I want to cancel my subscription.",
        "Please stop billing me, I'd like to cancel my plan.",
        "How do I unsubscribe from this service?",
        "Cancel my membership starting next month.",
    ],
    "technical_issue": [
        "The app keeps crashing when I open it.",
        "I'm getting an error code every time I try to use the feature.",
        "The website won't load properly on my phone.",
        "Something is broken, the page just shows a blank screen.",
    ],
    "delivery_issue": [
        "My package never arrived even though it says delivered.",
        "The delivery is three days late already.",
        "I received the wrong item in my shipment.",
        "My order shows delivered but I never got it.",
    ],
    "general_question": [
        "What are your business hours?",
        "Do you offer this product in other colors?",
        "How long does shipping usually take?",
        "Can you tell me more about your premium plan?",
    ],
}

_AMOUNTS = ["$19.99", "$29.99", "$49.00", "$9.99", "$120.00"]


def generate_dataset(n_per_class: int = 60, seed: int = 42) -> list[tuple[str, str]]:
    """Return a list of (text, label) synthetic examples, shuffled."""
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for label, templates in _TEMPLATES.items():
        for _ in range(n_per_class):
            template = rng.choice(templates)
            text = (
                template.format(amount=rng.choice(_AMOUNTS)) if "{amount}" in template else template
            )
            examples.append((text, label))
    rng.shuffle(examples)
    return examples

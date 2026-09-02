"""Synthetic emotion-classification dataset (Phase 5).

Same provenance caveat as `synthetic_intent.py`: templated, synthetic,
NOT real customer data. Labels distinguish EMOTION (affective state) from
SENTIMENT (polarity) — this module handles emotion only; sentiment is
derived separately in `apps/speech_intelligence/nlp.py` via a lexicon-based
polarity score, and the two are never used interchangeably.

Sandbox note: acoustic emotion recognition (the spec's wav2vec2-embeddings
+ acoustic-features fusion model) requires audio model weights not
downloadable here. This generates TEXT utterances for the same label set
so `train_emotion.py` can produce real, measured metrics; the acoustic
fusion architecture is documented in `ml/training/train_emotion_audio.py`
as real, unexecuted production code.
"""

from __future__ import annotations

import random

EMOTION_LABELS: list[str] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "frustrated",
    "fearful",
    "urgent",
]

_TEMPLATES: dict[str, list[str]] = {
    "neutral": [
        "I'd like to check the status of my order.",
        "Can you tell me what my current plan includes?",
        "Just wanted to ask about your return policy.",
        "What time does support close today?",
    ],
    "happy": [
        "Thanks so much, that fixed it perfectly!",
        "I really appreciate how quickly you responded.",
        "This new feature is great, I love it.",
        "You've been super helpful, thank you!",
    ],
    "sad": [
        "I'm really disappointed this didn't work out.",
        "It's frustrating, I was really looking forward to this order.",
        "I feel let down by how this was handled.",
        "This has been a pretty discouraging experience.",
    ],
    "angry": [
        "This is completely unacceptable, fix it now.",
        "I am furious that no one has responded in days.",
        "How dare you charge me for something I never ordered!",
        "I'm done being patient, this is ridiculous.",
    ],
    "frustrated": [
        "I've contacted support three times already about this.",
        "I keep explaining the same issue over and over.",
        "Nothing you've suggested so far has worked.",
        "I'm getting nowhere with this and it's exhausting.",
    ],
    "fearful": [
        "I'm worried someone accessed my account without permission.",
        "I'm scared my card details might have been stolen.",
        "This looks suspicious and it's making me anxious.",
        "I'm concerned my personal information isn't safe.",
    ],
    "urgent": [
        "I need this resolved right now, it's time critical.",
        "This needs to be fixed immediately, I can't wait.",
        "Please treat this as urgent, it's affecting my business today.",
        "I need an answer within the hour if possible.",
    ],
}


def generate_dataset(n_per_class: int = 60, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for label, templates in _TEMPLATES.items():
        for _ in range(n_per_class):
            examples.append((rng.choice(templates), label))
    rng.shuffle(examples)
    return examples

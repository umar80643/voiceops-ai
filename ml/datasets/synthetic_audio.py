"""Synthetic audio-intent dataset: real waveforms, not text (Phase 4 upgrade).

Generates raw audio (not text) whose acoustic properties (pitch, energy,
tempo/pause pattern) are causally tied to the label, so a model must
learn from the WAVEFORM to classify correctly — a genuine audio-modality
task. Still synthetic (no real speech corpus reachable in this sandbox),
but this is now a real signal-processing task, not a text stand-in.
"""

from __future__ import annotations

import numpy as np

SR = 8000  # keep small for CPU-only training speed
DURATION = 1.0
N_SAMPLES = int(SR * DURATION)

LABELS = ["calm", "urgent", "angry", "sad"]


def _synth_utterance(label: str, rng: np.random.Generator) -> np.ndarray:
    """Each label maps to a distinct, noisy acoustic profile:
    - calm: low pitch, steady energy, few pauses
    - urgent: fast pitch modulation, high energy, short pauses
    - angry: high energy, harsh (clipped) waveform, low pitch variance
    - sad: low energy, slow pitch, long pauses
    """
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    base_f0 = {"calm": 120, "urgent": 220, "angry": 100, "sad": 90}[label]
    f0 = base_f0 + rng.normal(0, 8, size=N_SAMPLES).cumsum() * 0.002
    amp = {"calm": 0.3, "urgent": 0.6, "angry": 0.8, "sad": 0.15}[label]

    signal = amp * np.sin(2 * np.pi * f0 * t)
    signal += 0.05 * rng.normal(size=N_SAMPLES)  # noise floor

    if label == "angry":
        signal = np.clip(signal * 1.8, -0.9, 0.9)  # harsh clipping distortion
    if label in {"sad", "urgent"}:
        n_pauses = 4 if label == "sad" else 1
        pause_len = int(SR * (0.15 if label == "sad" else 0.05))
        for _ in range(n_pauses):
            start = rng.integers(0, N_SAMPLES - pause_len)
            signal[start : start + pause_len] *= 0.05

    return signal.astype(np.float32)


def generate_dataset(n_per_class: int = 80, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    waveforms, labels = [], []
    for label in LABELS:
        for _ in range(n_per_class):
            waveforms.append(_synth_utterance(label, rng))
            labels.append(label)
    return np.stack(waveforms), labels

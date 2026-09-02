"""Speaker diarization (Phase 3).

Production path: `pyannote.audio` speaker-diarization pipeline (requires
HF Hub access + accepting its gated model license) — not available in
this sandbox (no network access to huggingface.co). Documented as TODO.

To keep the pipeline honestly runnable end-to-end, this module implements
a real, dependency-free *silence-gap turn-taking heuristic*: it splits
audio into speech segments using energy-based VAD, then alternates
speaker labels across segments separated by a pause longer than
`turn_gap_seconds`. This is a legitimate, testable technique used in
simple two-party call analysis — but it is NOT deep-learning diarization
and does not use voice-print/speaker-embedding similarity. It is labeled
"speaker_heuristic_v1" throughout so it's never confused with a trained
diarization model, and role (customer/agent) is never guessed — only
SPEAKER_00 / SPEAKER_01, per the build spec.
"""

from __future__ import annotations

import numpy as np
from shared.schemas.diarization import DiarizedSegment

FRAME_MS = 20.0
ENERGY_THRESHOLD = 0.01
MIN_SEGMENT_SECONDS = 0.3
TURN_GAP_SECONDS = 0.6


def _voiced_frames(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = max(1, int(sample_rate * FRAME_MS / 1000))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    return rms >= ENERGY_THRESHOLD


def diarize(samples: np.ndarray, sample_rate: int) -> list[DiarizedSegment]:
    """Segment audio into alternating-speaker turns using a VAD + gap heuristic.

    Returns an empty list for silent/empty audio rather than fabricating
    a speaker turn.
    """
    voiced = _voiced_frames(samples, sample_rate)
    if voiced.size == 0 or not voiced.any():
        return []

    frame_len = max(1, int(sample_rate * FRAME_MS / 1000))
    frame_seconds = frame_len / sample_rate

    # Find contiguous voiced runs (candidate utterances).
    runs: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, is_voiced in enumerate(voiced):
        if is_voiced and run_start is None:
            run_start = i
        elif not is_voiced and run_start is not None:
            runs.append((run_start * frame_seconds, i * frame_seconds))
            run_start = None
    if run_start is not None:
        runs.append((run_start * frame_seconds, len(voiced) * frame_seconds))

    runs = [(s, e) for s, e in runs if (e - s) >= MIN_SEGMENT_SECONDS]
    if not runs:
        return []

    # Alternate speaker label whenever the gap since the previous run
    # exceeds TURN_GAP_SECONDS (a proxy for a conversational turn change).
    segments: list[DiarizedSegment] = []
    current_speaker = 0
    prev_end = runs[0][0]
    for start, end in runs:
        if start - prev_end > TURN_GAP_SECONDS and segments:
            current_speaker = 1 - current_speaker
        segments.append(
            DiarizedSegment(speaker=f"SPEAKER_{current_speaker:02d}", start=start, end=end)
        )
        prev_end = end

    return segments

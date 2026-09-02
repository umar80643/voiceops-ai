from __future__ import annotations

import numpy as np
from apps.diarization.service import diarize


def _tone(duration_s: float, sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_silent_audio_returns_no_segments() -> None:
    silence = np.zeros(16000, dtype=np.float32)
    assert diarize(silence, 16000) == []


def test_single_continuous_utterance_is_one_segment() -> None:
    audio = _tone(2.0)
    segments = diarize(audio, 16000)
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"


def test_two_utterances_with_long_gap_alternate_speakers() -> None:
    sr = 16000
    utter1 = _tone(1.0, sr)
    gap = np.zeros(int(1.0 * sr), dtype=np.float32)  # > TURN_GAP_SECONDS
    utter2 = _tone(1.0, sr)
    audio = np.concatenate([utter1, gap, utter2])

    segments = diarize(audio, sr)
    assert len(segments) == 2
    assert segments[0].speaker != segments[1].speaker


def test_never_invents_role_labels() -> None:
    audio = _tone(1.5)
    segments = diarize(audio, 16000)
    for seg in segments:
        assert seg.speaker in {"SPEAKER_00", "SPEAKER_01"}

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from ml.preprocessing.audio import (
    AudioValidationError,
    UnsupportedAudioFormatError,
    compute_silence_ratio,
    process_audio_upload,
)


def _make_wav_bytes(
    duration_s: float = 1.0, sr: int = 16000, freq: float = 440.0, amplitude: float = 0.5
) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def test_process_valid_wav() -> None:
    raw = _make_wav_bytes(duration_s=2.0)
    result = process_audio_upload(raw, "test.wav")
    assert result.sample_rate == 16000
    assert 1.9 < result.duration_seconds < 2.1
    assert result.audio_format == "wav"


def test_rejects_mp3() -> None:
    with pytest.raises(UnsupportedAudioFormatError):
        process_audio_upload(b"fake", "test.mp3")


def test_rejects_too_short_audio() -> None:
    raw = _make_wav_bytes(duration_s=0.05)
    with pytest.raises(AudioValidationError):
        process_audio_upload(raw, "short.wav")


def test_silence_ratio_all_silent() -> None:
    silent = np.zeros(16000, dtype=np.float32)
    ratio = compute_silence_ratio(silent, 16000)
    assert ratio == 1.0


def test_silence_ratio_all_loud() -> None:
    loud = np.ones(16000, dtype=np.float32) * 0.9
    ratio = compute_silence_ratio(loud, 16000)
    assert ratio == 0.0


def test_malformed_wav_raises() -> None:
    with pytest.raises(UnsupportedAudioFormatError):
        process_audio_upload(b"not a real wav file", "bad.wav")

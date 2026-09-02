from __future__ import annotations

import io
import wave

from apps.tts.service import SilentTTS


def test_synthesize_returns_valid_wav_bytes() -> None:
    result = SilentTTS().synthesize("Hello, this is a test response.")
    assert result.provider == "silent_dev"
    assert result.sample_rate == 16000
    assert result.latency_seconds >= 0.0

    # Must be a parseable, valid WAV file, not fake placeholder bytes.
    with wave.open(io.BytesIO(result.audio_bytes), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1


def test_longer_text_produces_longer_audio() -> None:
    short = SilentTTS().synthesize("Hi.")
    long = SilentTTS().synthesize("This is a much longer response with many more words in it.")
    assert len(long.audio_bytes) > len(short.audio_bytes)


def test_empty_text_still_produces_valid_wav() -> None:
    result = SilentTTS().synthesize("")
    with wave.open(io.BytesIO(result.audio_bytes), "rb") as wf:
        assert wf.getnframes() >= 0

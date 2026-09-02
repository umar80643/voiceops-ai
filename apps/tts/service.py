"""Text-to-speech (Phase 13) — pluggable provider interface.

Real, tested interface + a `SilentTTS` local dev implementation that
generates valid (silent) WAV bytes so the pipeline (LLM response -> TTS ->
audio bytes -> client) is exercisable end-to-end without any external
API key. `ElevenLabsTTS`/`PollyTTS`-style production providers would
implement the same `TTSProvider` protocol — swappable via config, per
spec ("do not tightly couple to one provider").
"""

from __future__ import annotations

import io
import struct
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    latency_seconds: float
    provider: str


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> TTSResult: ...


class SilentTTS:
    """Local dev provider: emits a valid, silent WAV of duration
    proportional to text length (~150 words/minute), so downstream
    clients/tests get real, playable (if silent) audio bytes rather than
    a fake placeholder string."""

    provider_name = "silent_dev"
    sample_rate = 16000

    def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()
        words = max(1, len(text.split()))
        duration_seconds = words / (150 / 60)
        n_samples = int(duration_seconds * self.sample_rate)

        buf = io.BytesIO()
        n_channels, sampwidth = 1, 2
        byte_rate = self.sample_rate * n_channels * sampwidth
        block_align = n_channels * sampwidth
        data = b"\x00\x00" * n_samples

        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + len(data)))
        buf.write(b"WAVEfmt ")
        buf.write(
            struct.pack("<IHHIIHH", 16, 1, n_channels, self.sample_rate, byte_rate, block_align, 16)
        )
        buf.write(b"data")
        buf.write(struct.pack("<I", len(data)))
        buf.write(data)

        elapsed = time.perf_counter() - start
        return TTSResult(
            audio_bytes=buf.getvalue(),
            sample_rate=self.sample_rate,
            latency_seconds=elapsed,
            provider=self.provider_name,
        )


# TODO(production): implement a real provider (e.g. ElevenLabs, AWS Polly,
# or a local Coqui/Bark model) behind this same Protocol. Not implemented
# here — all require either an API key or a model download not available
# in this sandbox.

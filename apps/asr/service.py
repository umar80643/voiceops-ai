"""Automatic Speech Recognition (Phase 2).

Production path: Hugging Face Transformers Whisper pipeline
(`transformers.pipeline("automatic-speech-recognition", model=...)`),
loaded once per process, GPU if available else CPU.

IMPORTANT — sandbox limitation: this container has no network access to
huggingface.co (only pypi/npm/github are allowlisted), so the real Whisper
weights cannot be downloaded or evaluated here. `WhisperASR` below is the
real, correct integration code and will work as soon as the process has
HF Hub access (or a local model cache) — it is not run in this sandbox.

To keep the system honestly runnable end-to-end without network access,
`NullASR` is provided: it does no speech recognition and returns an empty
transcript with a clearly-flagged `engine="null"`, so callers can never
mistake it for a real transcription. Tests exercise `NullASR` and the
Whisper wrapper's parameter/plumbing logic (not its accuracy), per Rule 1
(no fake implementations claiming to work).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from shared.schemas.transcription import TranscriptionResponse, TranscriptSegment


class ASRModelUnavailableError(Exception):
    """Raised when a real ASR backend cannot be loaded (e.g. no model weights)."""


@dataclass
class ASRResult:
    text: str
    segments: list[TranscriptSegment]
    language: str
    real_time_factor: float
    engine: str


class ASREngine(Protocol):
    def transcribe(self, samples: Any, sample_rate: int) -> ASRResult: ...


@dataclass
class NullASR:
    """Explicit no-op ASR backend used when no real model is loaded.

    Never fabricates transcript text. Exists so the API can be exercised
    end-to-end (upload -> transcribe -> downstream NLP) without silently
    pretending an untrained/unavailable model produced real output.
    """

    engine_name: str = field(default="null", init=False)

    def transcribe(self, samples: Any, sample_rate: int) -> ASRResult:
        return ASRResult(
            text="", segments=[], language="unknown", real_time_factor=0.0, engine="null"
        )


class WhisperASR:
    """Real Whisper integration via Hugging Face Transformers.

    Not exercised in this sandbox (no HF Hub network access). Correctness
    of the plumbing (device selection, chunking, model-load-once) is
    reviewed but WER/latency are NOT claimed here — see docs/evaluation.md,
    which will only be filled in once this has actually been run.
    """

    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None  # loaded lazily, once per process

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # noqa: PLC0415 (lazy import: heavy dep)
        except ImportError as exc:
            raise ASRModelUnavailableError(
                "transformers is not installed in this environment. "
                "Install the 'ml' extra to enable WhisperASR."
            ) from exc

        try:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_timestamps=True,
            )
        except Exception as exc:  # network/model-not-cached failure
            raise ASRModelUnavailableError(
                f"Could not load Whisper model '{self.model_name}': {exc}"
            ) from exc

    def transcribe(self, samples: Any, sample_rate: int) -> ASRResult:
        self._load()
        start = time.perf_counter()
        assert self._pipeline is not None
        output = self._pipeline({"array": samples, "sampling_rate": sample_rate})
        elapsed = time.perf_counter() - start

        chunks = output.get("chunks", [])
        segments = [
            TranscriptSegment(
                start=c["timestamp"][0] or 0.0, end=c["timestamp"][1] or 0.0, text=c["text"]
            )
            for c in chunks
        ]
        duration = len(samples) / sample_rate if sample_rate else 1.0
        rtf = elapsed / duration if duration else 0.0
        return ASRResult(
            text=output.get("text", ""),
            segments=segments,
            language="en",
            real_time_factor=rtf,
            engine=self.model_name,
        )


def to_response(result: ASRResult) -> TranscriptionResponse:
    return TranscriptionResponse(
        text=result.text,
        segments=result.segments,
        language=result.language,
        real_time_factor=result.real_time_factor,
    )

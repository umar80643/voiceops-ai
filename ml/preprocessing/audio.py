"""Audio ingestion & preprocessing (Phase 1).

Implements real WAV parsing/validation using the stdlib `wave` module and
numpy for signal analysis (mono-mixing, silence detection, chunking).

MP3/M4A support requires an external decoder (ffmpeg via pydub/torchaudio)
which is not installed in this environment (no network access to those
codecs' shared libs here) — see `TODO` below. WAV is fully implemented and
tested end-to-end.
"""

from __future__ import annotations

import io
import uuid
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 600.0
TARGET_SAMPLE_RATE = 16000
SILENCE_AMPLITUDE_THRESHOLD = 0.01  # normalized [-1, 1] amplitude
CHUNK_SECONDS = 30.0


class UnsupportedAudioFormatError(Exception):
    pass


class AudioValidationError(Exception):
    pass


@dataclass
class ProcessedAudio:
    audio_id: str
    samples: np.ndarray  # float32, mono, normalized [-1, 1]
    sample_rate: int
    original_channels: int
    duration_seconds: float
    silence_ratio: float
    num_chunks: int
    audio_format: str


def _read_wav(raw_bytes: bytes) -> tuple[np.ndarray, int, int]:
    """Parse WAV bytes into (samples[float32, shape=(n_channels,n_frames) or (n,)], sample_rate, channels)."""
    try:
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise UnsupportedAudioFormatError(f"Could not parse WAV file: {exc}") from exc

    dtype_map: dict[int, type[np.signedinteger[Any]]] = {1: np.int8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise UnsupportedAudioFormatError(f"Unsupported sample width: {sample_width} bytes")

    dtype = dtype_map[sample_width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    max_val = float(np.iinfo(dtype).max)
    audio = audio / max_val  # normalize to [-1, 1]

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    return audio, frame_rate, n_channels


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1)


def _resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampler (dependency-free).

    Adequate for validation/demo purposes. Production ASR services should
    use a proper polyphase resampler (torchaudio.transforms.Resample);
    documented as a TODO for the ASR phase.
    """
    if orig_sr == target_sr:
        return samples
    duration = len(samples) / orig_sr
    n_target = int(round(duration * target_sr))
    if n_target <= 0:
        return np.zeros(0, dtype=np.float32)
    orig_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    target_times = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(target_times, orig_times, samples).astype(np.float32)


def compute_silence_ratio(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    threshold: float = SILENCE_AMPLITUDE_THRESHOLD,
) -> float:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return 1.0 if np.abs(samples).max(initial=0.0) < threshold else 0.0
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(trimmed**2, axis=1))
    silent_frames = int(np.sum(frame_rms < threshold))
    return silent_frames / n_frames


def chunk_audio(samples: np.ndarray, sample_rate: int, chunk_seconds: float = CHUNK_SECONDS) -> int:
    chunk_len = int(chunk_seconds * sample_rate)
    if chunk_len <= 0:
        return 1
    return max(1, int(np.ceil(len(samples) / chunk_len)))


def process_audio_upload(raw_bytes: bytes, filename: str) -> ProcessedAudio:
    """Validate + normalize an uploaded audio file. Currently supports WAV.

    Raises `UnsupportedAudioFormatError` or `AudioValidationError` on
    invalid input rather than silently accepting bad audio.
    """
    lower_name = filename.lower()
    if lower_name.endswith((".mp3", ".m4a")):
        raise UnsupportedAudioFormatError(
            "MP3/M4A require an external codec (ffmpeg) not installed in this "
            "environment. TODO: add torchaudio/ffmpeg-backed decoding. WAV is "
            "fully supported."
        )
    if not lower_name.endswith(".wav"):
        raise UnsupportedAudioFormatError(f"Unrecognized audio extension for '{filename}'")

    samples, orig_sr, channels = _read_wav(raw_bytes)
    mono = _to_mono(samples)
    duration = len(mono) / orig_sr if orig_sr else 0.0

    if duration < MIN_DURATION_SECONDS:
        raise AudioValidationError(f"Audio too short: {duration:.3f}s < {MIN_DURATION_SECONDS}s")
    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(f"Audio too long: {duration:.1f}s > {MAX_DURATION_SECONDS}s")

    resampled = _resample_linear(mono, orig_sr, TARGET_SAMPLE_RATE)
    silence_ratio = compute_silence_ratio(resampled, TARGET_SAMPLE_RATE)
    num_chunks = chunk_audio(resampled, TARGET_SAMPLE_RATE)

    return ProcessedAudio(
        audio_id=str(uuid.uuid4()),
        samples=resampled,
        sample_rate=TARGET_SAMPLE_RATE,
        original_channels=channels,
        duration_seconds=round(len(resampled) / TARGET_SAMPLE_RATE, 3),
        silence_ratio=round(silence_ratio, 4),
        num_chunks=num_chunks,
        audio_format="wav",
    )

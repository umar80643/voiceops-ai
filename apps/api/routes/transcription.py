from __future__ import annotations

from apps.asr.service import ASRModelUnavailableError, NullASR, WhisperASR, to_response
from fastapi import APIRouter, HTTPException
from ml.preprocessing.audio import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    _read_wav,
    _resample_linear,
    _to_mono,
)
from shared.config import get_settings
from shared.logging import get_logger
from shared.schemas.transcription import TranscriptionResponse
from shared.utils.object_store import LocalDiskObjectStore

router = APIRouter(prefix="/transcribe", tags=["asr"])
logger = get_logger(__name__)
_store = LocalDiskObjectStore()
settings = get_settings()


@router.post("/{audio_id}", response_model=TranscriptionResponse)
async def transcribe(audio_id: str) -> TranscriptionResponse:
    try:
        raw_bytes = _store.get(f"{audio_id}.wav")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"audio_id '{audio_id}' not found") from exc

    samples, orig_sr, _ = _read_wav(raw_bytes)
    mono = _resample_linear(_to_mono(samples), orig_sr, TARGET_SAMPLE_RATE)

    try:
        engine = WhisperASR(model_name=settings.asr_model_name, device=settings.asr_device)
        result = engine.transcribe(mono, TARGET_SAMPLE_RATE)
    except ASRModelUnavailableError as exc:
        logger.warning("asr_falling_back_to_null", reason=str(exc))
        result = NullASR().transcribe(mono, TARGET_SAMPLE_RATE)

    return to_response(result)

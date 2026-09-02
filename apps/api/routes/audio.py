from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile
from ml.preprocessing.audio import (
    AudioValidationError,
    UnsupportedAudioFormatError,
    process_audio_upload,
)
from shared.logging import get_logger
from shared.schemas.audio import AudioStatus, AudioUploadResponse
from shared.utils.object_store import LocalDiskObjectStore

router = APIRouter(prefix="/audio", tags=["audio"])
logger = get_logger(__name__)
_store = LocalDiskObjectStore()


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile) -> AudioUploadResponse:
    raw_bytes = await file.read()
    filename = file.filename or "upload.wav"

    try:
        processed = process_audio_upload(raw_bytes, filename)
    except UnsupportedAudioFormatError as exc:
        logger.warning("audio_upload_rejected", filename=filename, reason=str(exc))
        return AudioUploadResponse(
            audio_id="",
            duration_seconds=0.0,
            sample_rate=0,
            channels=0,
            status=AudioStatus.REJECTED,
            reason=str(exc),
        )
    except AudioValidationError as exc:
        logger.warning("audio_upload_invalid", filename=filename, reason=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _store.put(f"{processed.audio_id}.wav", raw_bytes)
    logger.info(
        "audio_upload_accepted",
        audio_id=processed.audio_id,
        duration=processed.duration_seconds,
        silence_ratio=processed.silence_ratio,
    )

    return AudioUploadResponse(
        audio_id=processed.audio_id,
        duration_seconds=processed.duration_seconds,
        sample_rate=processed.sample_rate,
        channels=1,
        status=AudioStatus.ACCEPTED,
    )

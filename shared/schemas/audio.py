from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AudioStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class AudioUploadResponse(BaseModel):
    audio_id: str
    duration_seconds: float
    sample_rate: int
    channels: int
    status: AudioStatus
    reason: str | None = None


class AudioMetadata(BaseModel):
    audio_id: str
    original_filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    silence_ratio: float = Field(ge=0.0, le=1.0)
    num_chunks: int

from __future__ import annotations

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    segments: list[TranscriptSegment]
    language: str
    real_time_factor: float | None = None

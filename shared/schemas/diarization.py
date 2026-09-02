from __future__ import annotations

from pydantic import BaseModel


class DiarizedSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str = ""


class DiarizationResponse(BaseModel):
    conversation: list[DiarizedSegment]
    num_speakers: int

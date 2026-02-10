"""
Pydantic models for AI-1 Audio Intelligence pipeline output.

Output contract designed for backend normalization and syncing to Backboard.io.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class PipelineError(BaseModel):
    """Error response when pipeline fails gracefully."""

    success: bool = False
    error: str
    error_code: str
    call_id: Optional[str] = None


class Segment(BaseModel):
    """A single timestamped transcript segment."""

    start: float
    end: float
    speaker: Optional[str] = None
    text: str

    model_config = ConfigDict(frozen=False)


class AudioMetadata(BaseModel):
    """Metadata describing the processed audio."""

    duration_sec: float
    sample_rate: int
    chunks: int
    cleaned: bool

    model_config = ConfigDict(frozen=False)


class PipelineOutput(BaseModel):
    """Complete pipeline output suitable for Backboard.io sync."""

    call_id: str
    audio_metadata: AudioMetadata
    segments: list[Segment]
    full_transcript: str
    model: str
    detected_language: str | None = None
    language_probability: float | None = None
    avg_logprob: float | None = None
    processing_time_sec: float

    model_config = ConfigDict(frozen=False)

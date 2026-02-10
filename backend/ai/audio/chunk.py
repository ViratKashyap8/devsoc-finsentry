"""
Chunk module: split audio into non-overlapping segments of ≤60 seconds.

Keeps ASR input within model-friendly length.
"""

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_DURATION_SEC = 60


class ChunkInfo(NamedTuple):
    """Metadata for a single audio chunk."""

    audio: np.ndarray
    start_sec: float
    end_sec: float


def chunk(y: np.ndarray, sr: int, max_duration_sec: float = CHUNK_DURATION_SEC) -> list[ChunkInfo]:
    """
    Split audio into non-overlapping chunks of at most max_duration_sec.

    Args:
        y: Input audio (float32).
        sr: Sample rate (Hz).
        max_duration_sec: Maximum chunk duration in seconds.

    Returns:
        List of ChunkInfo(audio, start_sec, end_sec).
    """
    if len(y) == 0:
        return []

    total_samples = len(y)
    total_duration_sec = total_samples / sr
    samples_per_chunk = int(max_duration_sec * sr)

    chunks: list[ChunkInfo] = []
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + samples_per_chunk, total_samples)
        chunk_audio = y[start_sample:end_sample]
        start_sec = start_sample / sr
        end_sec = end_sample / sr
        chunks.append(ChunkInfo(audio=chunk_audio, start_sec=start_sec, end_sec=end_sec))
        start_sample = end_sample

    logger.info("Split %.2f s audio into %d chunks", total_duration_sec, len(chunks))
    return chunks

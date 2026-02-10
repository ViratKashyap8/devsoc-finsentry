"""
Silence module: trim leading and trailing silence.

Reduces processing and improves transcript alignment.
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Uses librosa.effects.trim with an RMS-based threshold (top_db).
    Keeps only the non-silent portion.

    Args:
        y: Input audio (float32, mono).
        top_db: Threshold below peak to consider silence (dB).

    Returns:
        Trimmed audio (float32).
    """
    if len(y) == 0:
        return y

    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    trimmed_samples = len(y) - len(y_trimmed)
    logger.info("Silence trim: removed %d samples (top_db=%d)", trimmed_samples, top_db)

    return y_trimmed.astype(np.float32)

"""
Preprocess module: resample audio to mono 16 kHz WAV.

16 kHz mono is the standard for phone-band ASR and matches telephony specs.
"""

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def preprocess(input_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load and resample audio to mono 16 kHz.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Tuple of (audio_array, sample_rate). Audio is float32 in [-1, 1].
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    try:
        y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception as e:
        err_msg = str(e).lower()
        if "ffmpeg" in err_msg or "audioread" in err_msg or "decoder" in err_msg:
            raise ValueError(
                f"FFmpeg failed to decode audio. Ensure FFmpeg is installed and on PATH: {e}"
            ) from e
        raise ValueError(f"Failed to load audio: {e}") from e

    if len(y) == 0:
        raise ValueError("Audio file is empty after loading")

    # mono=True ensures stereo -> mono conversion; log if original was multi-channel
    logger.info("Preprocessed %s: %d samples at %d Hz (mono)", path.name, len(y), sr)
    return y, sr

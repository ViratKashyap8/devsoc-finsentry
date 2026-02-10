"""
DSP module: band-pass filter (300–3400 Hz), noise reduction, RMS normalization.

Telephony band limits and cleanup for improved ASR performance.

Noise reduction is intentionally conservative to avoid destroying speech,
and can be tuned via environment variables for experimentation:

- FINSENTRY_DSP_REDUCE_NOISE: \"1\"/\"0\" (default: \"1\")
- FINSENTRY_DSP_PROP_DECREASE: float, e.g. \"0.6\" (default: 0.6)
"""

import logging
import os

import noisereduce as nr
import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

LOWCUT = 300
HIGHCUT = 3400
ORDER = 4
TARGET_RMS = 0.05

_REDUCE_NOISE_DEFAULT = os.getenv("FINSENTRY_DSP_REDUCE_NOISE", "1").lower() not in {
    "0",
    "false",
    "no",
}
try:
    _PROP_DECREASE_DEFAULT = float(os.getenv("FINSENTRY_DSP_PROP_DECREASE", "0.6"))
except ValueError:
    _PROP_DECREASE_DEFAULT = 0.6


def _bandpass_filter(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply zero-phase band-pass filter (300–3400 Hz).

    Telephony band limits; attenuates out-of-band noise and DC offset.
    """
    nyq = sr / 2
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    sos = butter(ORDER, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, y)


def _rms_normalize(y: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """
    Scale waveform to target RMS; clip to [-1, 1].

    Improves ASR consistency across varying recording levels.
    """
    rms = np.sqrt(np.mean(y**2) + 1e-10)
    if rms < 1e-10:
        return y
    scale = target_rms / rms
    y_norm = y * scale
    return np.clip(y_norm, -1.0, 1.0).astype(np.float32)


def dsp(
    y: np.ndarray,
    sr: int,
    reduce_noise: bool = _REDUCE_NOISE_DEFAULT,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply DSP chain: band-pass filter, noise reduction, RMS normalization.

    Args:
        y: Input audio (float32, mono).
        sr: Sample rate (Hz).
        reduce_noise: Whether to run noisereduce.
        normalize: Whether to apply RMS normalization.

    Returns:
        Processed audio (float32).
    """
    if len(y) == 0:
        raise ValueError("Empty audio")

    y_out = _bandpass_filter(y, sr)
    logger.info("DSP: band-pass %d–%d Hz applied", LOWCUT, HIGHCUT)

    if reduce_noise:
        y_out = nr.reduce_noise(
            y=y_out,
            sr=sr,
            stationary=True,
            prop_decrease=_PROP_DECREASE_DEFAULT,
        )
        logger.info(
            "DSP: noise reduction applied (stationary=%s, prop_decrease=%.2f)",
            True,
            _PROP_DECREASE_DEFAULT,
        )
    else:
        logger.info("DSP: noise reduction disabled (FINSENTRY_DSP_REDUCE_NOISE)")

    if normalize:
        y_out = _rms_normalize(y_out)
        logger.info("DSP: RMS normalization to %.2f", TARGET_RMS)

    return y_out.astype(np.float32)

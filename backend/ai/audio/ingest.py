"""
Ingest module: validates uploads, generates call_id, saves to data/raw/.

Fails gracefully on corrupted or invalid audio files.
Supports stereo; converts to mono in preprocess. Uses librosa for MP3/M4A.
"""

import logging
import shutil
import uuid
from pathlib import Path

import librosa

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
MIN_DURATION_SEC = 2.0


def _read_audio_for_validation(path: Path) -> tuple[int, int]:
    """Read audio to validate and get duration. Uses librosa (handles MP3, M4A, stereo)."""
    try:
        y, sr = librosa.load(str(path), sr=None, mono=False)
    except Exception as e:
        err_msg = str(e).lower()
        if "ffmpeg" in err_msg or "audioread" in err_msg or "decoder" in err_msg:
            raise ValueError(
                f"FFmpeg failed to decode audio. Ensure FFmpeg is installed and on PATH: {e}"
            ) from e
        raise ValueError(f"Corrupted or invalid audio file: {e}") from e

    if y.size == 0:
        raise ValueError("Audio file is empty")

    n_samples = y.shape[0] if y.ndim >= 1 else len(y)
    if sr <= 0:
        raise ValueError("Invalid sample rate in audio file")
    return n_samples, sr


def ingest(
    input_path: str | Path,
    *,
    call_id_override: str | None = None,
) -> tuple[str, Path]:
    """
    Validate an audio file, generate a call_id, and save it to data/raw/.

    Args:
        input_path: Path to the input audio file.
        call_id_override: If provided, use this stable call_id (for deterministic output).

    Returns:
        Tuple of (call_id, path_to_saved_file).

    Raises:
        ValueError: If file is missing, has invalid extension, corrupted, or too short.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    logger.info("Validating audio: %s", path.name)
    try:
        n_samples, sr = _read_audio_for_validation(path)
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Failed to read audio file")
        raise ValueError(f"Corrupted or invalid audio file: {e}") from e

    duration_sec = n_samples / sr
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short: {duration_sec:.2f}s. Minimum {MIN_DURATION_SEC}s required."
        )

    call_id = call_id_override if call_id_override else uuid.uuid4().hex[:12]
    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{call_id}{ext}"
    out_path = raw_dir / out_name
    shutil.copy2(str(path), str(out_path))
    logger.info("Ingested %s -> %s", path.name, out_path)

    return call_id, out_path

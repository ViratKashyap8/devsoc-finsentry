"""
STT module: faster-whisper wrapper with timestamps.

Defaults to the Whisper **medium** model for better robustness on noisy,
code-switched (Hindi + English) calls. No diarization.
"""

import logging
from typing import Any, Dict, List, Tuple

from .chunk import ChunkInfo
from .schema import Segment

logger = logging.getLogger(__name__)

MODEL_NAME = "faster-whisper-medium"


def _load_whisper(model_size: str = "medium"):
    from faster_whisper import WhisperModel

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
    except ImportError:
        device = "cpu"
        compute_type = "int8"

    logger.info("Loading Whisper model: %s on %s (%s)", model_size, device, compute_type)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_chunks(
    chunks: List[ChunkInfo],
    sr: int,
    model_size: str = "medium",
) -> Tuple[List[Segment], str, Dict[str, Any]]:
    """
    Transcribe audio chunks with faster-whisper; return segments and full transcript.

    Offsets segment timestamps by each chunk's start time. No speaker diarization.

    Args:
        chunks: List of ChunkInfo from chunk module.
        sr: Sample rate (Hz); must be 16000 for Whisper.
        model_size: Whisper model size (base, small, medium, large-v2, etc.).

    Returns:
        Tuple of (segments, full_transcript, stt_info) where stt_info contains:
        {"detected_language": str | None, "language_probability": float | None,
         "avg_logprob": float | None}
    """
    if not chunks:
        return [], "", {"detected_language": None, "language_probability": None, "avg_logprob": None}

    model = _load_whisper(model_size=model_size)
    all_segments: List[Segment] = []
    full_parts: List[str] = []
    all_logprobs: List[float] = []
    detected_language: str | None = None
    language_probability: float | None = None

    for ch in chunks:
        try:
            seg_iter, info = model.transcribe(
                ch.audio,
                language=None,  # let Whisper auto-detect language (bilingual-friendly)
                word_timestamps=False,
                vad_filter=True,  # filter non-speech with VAD
                beam_size=5,  # beam search for more robust decoding
            )

            # Capture language metadata once (same model for all chunks)
            if info is not None and detected_language is None:
                detected_language = getattr(info, "language", None)
                language_probability = getattr(info, "language_probability", None)
                logger.info(
                    "Whisper detected language=%s (prob=%.3f)",
                    detected_language,
                    language_probability or 0.0,
                )

            for s in seg_iter:
                text = (getattr(s, "text", "") or "").strip()
                if not text:
                    continue
                all_segments.append(
                    Segment(
                        start=ch.start_sec + float(getattr(s, "start", 0.0)),
                        end=ch.start_sec + float(getattr(s, "end", 0.0)),
                        speaker=None,
                        text=text,
                    )
                )
                full_parts.append(text)
                avg_lp = getattr(s, "avg_logprob", None)
                if isinstance(avg_lp, (int, float)):
                    all_logprobs.append(float(avg_lp))
        except Exception as e:
            logger.warning("STT failed for chunk %.2f–%.2f: %s", ch.start_sec, ch.end_sec, e)

    full_transcript = " ".join(full_parts).strip()
    avg_logprob = sum(all_logprobs) / len(all_logprobs) if all_logprobs else None

    stt_info: Dict[str, Any] = {
        "detected_language": detected_language,
        "language_probability": language_probability,
        "avg_logprob": avg_logprob,
    }
    return all_segments, full_transcript, stt_info

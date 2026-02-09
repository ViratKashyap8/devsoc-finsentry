"""
Preprocessing for financial call transcripts.

Normalizes text for model input, handles common STT artifacts.
"""

import re
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize transcript text for NLP models.

    - Collapse repeated spaces
    - Normalize common STT artifacts (um, uh, etc.)
    - Fix common number/currency patterns
    """
    if not text:
        return ""
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Optional: remove filler words (can hurt emotion detection - keep for now)
    # text = re.sub(r'\b(um|uh|er|ah)\b', '', text, flags=re.IGNORECASE)
    return text.strip()


def preprocess_transcript(
    full_transcript: str,
    segments: Optional[list[dict]] = None,
    max_segment_length: int = 512,
) -> tuple[str, list[dict]]:
    """
    Preprocess full transcript and optional segments.

    Args:
        full_transcript: Raw transcript text
        segments: Optional list of {start, end, text} dicts
        max_segment_length: Truncate segments longer than this (characters)

    Returns:
        (normalized_full_transcript, normalized_segments)
    """
    full = normalize_text(full_transcript)
    out_segments = []
    if segments:
        for seg in segments:
            text = normalize_text(seg.get("text", ""))
            if len(text) > max_segment_length:
                text = text[: max_segment_length - 3] + "..."
            out_segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            })
    return full, out_segments


def extract_sentences_for_analysis(
    full_transcript: str, max_length: int = 512
) -> list[str]:
    """
    Split transcript into sentences/utterances for per-segment analysis.

    Uses simple sentence splitting (period, question mark, exclamation).
    For finer control, use pre-segmented chunks from STT.
    """
    full = normalize_text(full_transcript)
    sentences = re.split(r"(?<=[.!?])\s+", full)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            if len(s) > max_length:
                # Split long utterances by comma
                parts = re.split(r",\s+", s)
                buf = ""
                for p in parts:
                    if len(buf) + len(p) + 2 <= max_length:
                        buf = buf + ", " + p if buf else p
                    else:
                        if buf:
                            result.append(buf.strip())
                        buf = p
                if buf:
                    result.append(buf.strip())
            else:
                result.append(s)
    return result if result else [full]

"""
Pipeline module: orchestrates ingest, preprocess, DSP, silence trim, chunking, and STT.

CLI: python ai/audio/pipeline.py path/to/audio.wav [--benchmark] [--save-intermediate]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running as script: python ai/audio/pipeline.py
# Remove script dir from path to avoid "chunk" resolving to ai/audio/chunk.py (name clash)
if __package__ is None:
    _script_dir = Path(__file__).resolve().parent
    _root = _script_dir.parent.parent
    if str(_script_dir) in sys.path:
        sys.path.remove(str(_script_dir))
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ai.audio.chunk import chunk
from ai.audio.dsp import dsp
from ai.audio.ingest import ingest
from ai.audio.preprocess import preprocess
from ai.audio.schema import AudioMetadata, PipelineError, PipelineOutput, Segment
from ai.audio.silence import trim_silence
from ai.audio.stt import transcribe_chunks

FLOAT_PRECISION = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MIN_DURATION_SEC = 2.0


def _save_intermediate(
    debug_dir: Path,
    call_id: str,
    y_resampled: bytes | None = None,
    y_post_dsp: bytes | None = None,
    chunks_data: list[tuple[bytes, float, float]] | None = None,
    sr: int = 16000,
) -> None:
    """Save intermediate audio to data/debug/ for inspection."""
    import numpy as np
    import soundfile as sf

    debug_dir.mkdir(parents=True, exist_ok=True)
    if y_resampled is not None:
        arr = np.frombuffer(y_resampled, dtype=np.float32)
        sf.write(debug_dir / f"{call_id}_01_resampled.wav", arr, sr)
        logger.info("Saved resampled audio to %s", debug_dir / f"{call_id}_01_resampled.wav")
    if y_post_dsp is not None:
        arr = np.frombuffer(y_post_dsp, dtype=np.float32)
        sf.write(debug_dir / f"{call_id}_02_post_dsp.wav", arr, sr)
        logger.info("Saved post-DSP audio to %s", debug_dir / f"{call_id}_02_post_dsp.wav")
    if chunks_data:
        for i, (chunk_bytes, start, end) in enumerate(chunks_data):
            arr = np.frombuffer(chunk_bytes, dtype=np.float32)
            sf.write(
                debug_dir / f"{call_id}_03_chunk_{i:02d}_{start:.1f}s_{end:.1f}s.wav",
                arr,
                sr,
            )
        logger.info("Saved %d chunks to %s", len(chunks_data), debug_dir)


def _normalize_output(output: PipelineOutput, model_name: str) -> PipelineOutput:
    """Deterministic output: sort segments, round floats."""
    segments = sorted(
        output.segments,
        key=lambda s: (round(s.start, FLOAT_PRECISION), round(s.end, FLOAT_PRECISION)),
    )
    normalized = [
        Segment(
            start=round(s.start, FLOAT_PRECISION),
            end=round(s.end, FLOAT_PRECISION),
            speaker=s.speaker,
            text=s.text,
        )
        for s in segments
    ]
    return PipelineOutput(
        call_id=output.call_id,
        audio_metadata=AudioMetadata(
            duration_sec=round(output.audio_metadata.duration_sec, FLOAT_PRECISION),
            sample_rate=output.audio_metadata.sample_rate,
            chunks=output.audio_metadata.chunks,
            cleaned=output.audio_metadata.cleaned,
        ),
        segments=normalized,
        full_transcript=output.full_transcript,
        model=model_name,
        detected_language=output.detected_language,
        language_probability=output.language_probability,
        avg_logprob=output.avg_logprob,
        processing_time_sec=round(output.processing_time_sec, FLOAT_PRECISION),
    )


def run_pipeline(
    input_path: str | Path,
    *,
    save_intermediate: bool = False,
    model_size: str = "medium",
    call_id_override: str | None = None,
) -> tuple[PipelineOutput, dict[str, float]]:
    """
    Run the full audio intelligence pipeline.

    Args:
        input_path: Path to input audio file.
        save_intermediate: If True, save resampled/post-DSP/chunked audio to data/debug/.
        model_size: Whisper model size (base, small, medium, etc.).
        call_id_override: Optional stable call_id for deterministic runs.

    Returns:
        Tuple of (PipelineOutput, timings_dict). timings_dict has per-stage seconds.

    Raises:
        ValueError: On invalid or corrupted input.
    """
    path = Path(input_path).resolve()
    timings: dict[str, float] = {}
    model_name = f"faster-whisper-{model_size}"

    # --- Ingest ---
    t0 = time.perf_counter()
    call_id, raw_path = ingest(path, call_id_override=call_id_override)
    timings["ingest"] = time.perf_counter() - t0
    logger.info("[ingest] %.3fs -> call_id=%s", timings["ingest"], call_id)

    # --- Preprocess ---
    t0 = time.perf_counter()
    y, sr = preprocess(raw_path)
    timings["preprocess"] = time.perf_counter() - t0
    logger.info("[preprocess] %.3fs -> %d samples @ %d Hz", timings["preprocess"], len(y), sr)

    duration_sec = len(y) / sr
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short after load: {duration_sec:.2f}s. Minimum {MIN_DURATION_SEC}s."
        )

    _project_root = Path(__file__).resolve().parent.parent.parent
    debug_dir = _project_root / "data" / "debug"
    if save_intermediate:
        _save_intermediate(debug_dir, call_id, y.tobytes(), None, None, sr)

    # --- DSP ---
    t0 = time.perf_counter()
    y_cleaned = dsp(y, sr)
    timings["dsp"] = time.perf_counter() - t0
    logger.info("[dsp] %.3fs", timings["dsp"])

    if save_intermediate:
        _save_intermediate(debug_dir, call_id, None, y_cleaned.tobytes(), None, sr)

    # --- Silence trim ---
    t0 = time.perf_counter()
    y_trimmed = trim_silence(y_cleaned)
    timings["silence"] = time.perf_counter() - t0
    logger.info("[silence] %.3fs -> %d samples", timings["silence"], len(y_trimmed))

    # If trimming removed everything, use cleaned audio
    if len(y_trimmed) == 0:
        logger.warning("Silence trim removed all audio; using pre-trim audio")
        y_trimmed = y_cleaned

    duration_sec = len(y_trimmed) / sr

    # --- Chunk ---
    t0 = time.perf_counter()
    chunks = chunk(y_trimmed, sr)
    timings["chunk"] = time.perf_counter() - t0
    logger.info("[chunk] %.3fs -> %d chunks", timings["chunk"], len(chunks))

    if save_intermediate:
        chunks_data = [(ch.audio.tobytes(), ch.start_sec, ch.end_sec) for ch in chunks]
        _save_intermediate(debug_dir, call_id, None, None, chunks_data, sr)

    # --- STT ---
    t0 = time.perf_counter()
    segments, full_transcript, stt_info = transcribe_chunks(chunks, sr, model_size=model_size)
    timings["stt"] = time.perf_counter() - t0
    logger.info(
        "[stt] %.3fs -> %d segments (lang=%s, prob=%s, avg_logprob=%s)",
        timings["stt"],
        len(segments),
        stt_info.get("detected_language"),
        stt_info.get("language_probability"),
        stt_info.get("avg_logprob"),
    )

    total_time = sum(timings.values())

    raw_output = PipelineOutput(
        call_id=call_id,
        audio_metadata=AudioMetadata(
            duration_sec=round(duration_sec, FLOAT_PRECISION),
            sample_rate=sr,
            chunks=len(chunks),
            cleaned=True,
        ),
        segments=segments,
        full_transcript=full_transcript,
        model=model_name,
        detected_language=stt_info.get("detected_language"),
        language_probability=stt_info.get("language_probability"),
        avg_logprob=stt_info.get("avg_logprob"),
        processing_time_sec=round(total_time, FLOAT_PRECISION),
    )
    output = _normalize_output(raw_output, model_name)

    # Schema validation (Pydantic validates on construction; explicit check for clarity)
    try:
        output.model_validate(output.model_dump())
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}") from e

    return output, timings


def _print_benchmark(
    output: PipelineOutput, input_path: str, timings: dict[str, float] | None = None
) -> None:
    """Print benchmark metrics."""
    dur = output.audio_metadata.duration_sec
    proc = output.processing_time_sec
    rtf = proc / dur if dur > 0 else float("inf")
    print("\n--- Benchmark ---", file=sys.stderr)
    print(f"  Input:        {input_path}", file=sys.stderr)
    print(f"  Duration:     {dur:.2f} s", file=sys.stderr)
    print(f"  Process time: {proc:.2f} s", file=sys.stderr)
    print(f"  Realtime:     {rtf:.2f}x (1x = realtime)", file=sys.stderr)
    if timings:
        print("  Stage breakdown:", file=sys.stderr)
        for stage, t in timings.items():
            pct = 100 * t / proc if proc > 0 else 0
            print(f"    {stage}: {t:.3f}s ({pct:.0f}%)", file=sys.stderr)
    print("-----------------\n", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-1 Audio Intelligence: transform phone-call audio into cleaned transcripts."
    )
    parser.add_argument("audio_path", help="Path to input audio file (.wav, .mp3, .m4a, .ogg, .flac)")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print benchmark stats (duration, processing time, realtime factor)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save resampled, post-DSP, and chunked audio to data/debug/",
    )
    parser.add_argument(
        "--model-size",
        default="medium",
        help="Whisper model size: base, small, medium, large-v2 (default: medium)",
    )
    parser.add_argument(
        "--call-id",
        default=None,
        help="Stable call_id for deterministic output (optional)",
    )
    args = parser.parse_args()

    try:
        output, timings = run_pipeline(
            args.audio_path,
            save_intermediate=args.save_intermediate,
            model_size=args.model_size,
            call_id_override=args.call_id,
        )
        if args.benchmark:
            _print_benchmark(output, args.audio_path, timings)
        print(json.dumps(output.model_dump(), indent=2))
    except ValueError as e:
        err = PipelineError(
            success=False,
            error=str(e),
            error_code="VALIDATION_ERROR",
            call_id=None,
        )
        logger.error("%s", e)
        print(json.dumps(err.model_dump(), indent=2))
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        err = PipelineError(
            success=False,
            error=f"Pipeline failed: {e}",
            error_code="PIPELINE_ERROR",
            call_id=None,
        )
        print(json.dumps(err.model_dump(), indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch-transcribe all .wav files under an input directory using AI-1 pipeline.
Writes one JSON per file to an output directory. Use before retraining AI-2.

Run from repo root: python scripts/transcribe_new_calls.py [--input-dir ...] [--output-dir ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
DEFAULT_INPUT = BACKEND / "data" / "new_calls" / "wav"
DEFAULT_OUTPUT = BACKEND / "data" / "new_calls" / "transcripts"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def discover_wavs(input_dir: Path) -> list[Path]:
    """Return sorted list of .wav files under input_dir (recursive)."""
    if not input_dir.is_dir():
        return []
    return sorted(input_dir.rglob("*.wav"))


def run_one(wav_path: Path, output_dir: Path, overwrite: bool, model_size: str) -> bool:
    """
    Run AI-1 pipeline on one .wav; write <stem>.json to output_dir.
    Returns True on success, False on failure (logs error).
    """
    from ai.audio.pipeline import run_pipeline

    stem = wav_path.stem
    out_path = output_dir / f"{stem}.json"
    if out_path.exists() and not overwrite:
        logger.info("Skip (exists): %s -> %s", wav_path.name, out_path.name)
        return True

    try:
        output, _timings = run_pipeline(wav_path, model_size=model_size)
        try:
            data = output.model_dump(mode="json")
        except TypeError:
            data = output.model_dump()
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("OK: %s -> %s", wav_path.name, out_path.name)
        return True
    except Exception as e:
        logger.exception("Failed %s: %s", wav_path.name, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-transcribe .wav files with AI-1 pipeline; write JSON transcripts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory to search for .wav files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory to write <stem>.json transcripts (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript JSON files",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        help="Whisper model size: base, small, medium (default: base)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    wavs = discover_wavs(input_dir)
    if not wavs:
        logger.warning("No .wav files under %s", input_dir)
        print("Summary: 0 files, 0 succeeded, 0 failed", flush=True)
        return 0

    logger.info("Found %d .wav file(s) under %s", len(wavs), input_dir)
    succeeded = 0
    failed = 0
    for wav_path in wavs:
        if run_one(wav_path, output_dir, args.overwrite, args.model_size):
            succeeded += 1
        else:
            failed += 1

    total = len(wavs)
    print(flush=True)
    print("Summary:", flush=True)
    print(f"  total:     {total}", flush=True)
    print(f"  succeeded: {succeeded}", flush=True)
    print(f"  failed:    {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

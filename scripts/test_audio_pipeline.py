#!/usr/bin/env python3
"""
Run AI-1 audio pipeline on a sample file.
Run from repo root: python scripts/test_audio_pipeline.py
  (or: make test-audio / backend/.venv/bin/python scripts/test_audio_pipeline.py)
Requires backend deps installed (e.g. make setup). Script injects backend into sys.path so the pipeline runs from repo root.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Resolve repo root from this script's location (works from any cwd)
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"

# Set Hugging Face cache to workspace directory (avoids sandbox permission issues)
hf_cache = BACKEND / ".cache" / "huggingface"
hf_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(hf_cache))
# Disable Xet downloader (avoids Rust panic on macOS)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Ensure backend is on path so "ai.audio" resolves when run from repo root
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _find_sample_audio() -> Path | None:
    # Prefer backend/ai/audio sample, then first file under backend/data/audio
    candidates = [
        BACKEND / "ai" / "audio" / "kings_road_2.wav",
        BACKEND / "ai" / "audio" / "Kings Road 2.m4a",
    ]
    for c in candidates:
        if c.exists():
            return c
    data_audio = BACKEND / "data" / "audio"
    if data_audio.exists():
        for ext in (".wav", ".mp3", ".m4a"):
            for f in data_audio.rglob(f"*{ext}"):
                return f
    return None


def main() -> int:
    sample = _find_sample_audio()
    if not sample:
        print("No sample audio found. Put a .wav/.mp3/.m4a in backend/data/audio or backend/ai/audio.", file=sys.stderr)
        return 1
    print(f"Running AI-1 audio pipeline on: {sample}", flush=True)
    print("(First run may download the Whisper 'medium' model from Hugging Face; ensure network access.)", flush=True)
    try:
        from ai.audio.pipeline import run_pipeline
    except ModuleNotFoundError as e:
        print(
            f"Missing dependency: {e}. Activate backend/.venv or run from repo root: make test-audio",
            file=sys.stderr,
        )
        return 1
    try:
        output, timings = run_pipeline(sample, model_size="medium")
        print("\n--- Pipeline output (summary) ---")
        print(f"  call_id: {output.call_id}")
        print(f"  duration_sec: {output.audio_metadata.duration_sec}")
        print(f"  segments: {len(output.segments)}")
        transcript_preview = output.full_transcript[:200]
        if len(output.full_transcript) > 200:
            transcript_preview += "..."
        print(f"  transcript (preview): {transcript_preview}")
        print("\n--- Timings ---")
        for k, v in timings.items():
            print(f"  {k}: {v:.3f}s")
        print("\n--- Full JSON ---")
        try:
            out_dict = output.model_dump(mode="json")
        except TypeError:
            out_dict = output.model_dump()
        json_str = json.dumps(out_dict, indent=2, default=str)
        print(json_str, flush=True)
        return 0
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

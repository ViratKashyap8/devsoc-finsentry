#!/usr/bin/env python3
"""Pre-download Whisper base model using a more reliable method."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"

# Set environment before importing
os.environ["HF_HOME"] = str(BACKEND / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable HF transfer too

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

def main():
    print("Downloading Whisper base model (this may take 2-3 minutes)...")
    print("Cache location:", os.environ["HF_HOME"])

    try:
        from faster_whisper import WhisperModel
        print("Loading model (will download if not cached)...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✓ Model downloaded and loaded successfully!")
        print(f"Model location: {model.model_path}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

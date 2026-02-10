#!/usr/bin/env python3
"""
Test audio file upload to the API.
Run from repo root: python scripts/test_audio_api_upload.py <audio_file>
  (e.g. python scripts/test_audio_api_upload.py backend/ai/audio/kings_road_2.wav)
Requires the API server to be running (make run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing dependency: httpx. Install with: pip install httpx", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file>", file=sys.stderr)
        print("Example: python scripts/test_audio_api_upload.py backend/ai/audio/kings_road_2.wav", file=sys.stderr)
        return 1

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        return 1

    api_url = "http://localhost:8000/api/audio/upload"
    print(f"Uploading {audio_path} to {api_url}...", flush=True)

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            with httpx.Client(timeout=300.0) as client:
                response = client.post(api_url, files=files)
                response.raise_for_status()
                result = response.json()
                print("\n--- Upload successful ---")
                print(json.dumps(result, indent=2))
                file_id = result.get("file_id")
                if file_id:
                    print(f"\nRetrieve with: curl http://localhost:8000/api/audio/{file_id}")
                return 0
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return 1
    except httpx.ConnectError:
        print(f"Connection error: Could not reach {api_url}. Is the server running? (make run)", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Upload failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

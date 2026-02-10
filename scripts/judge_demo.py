#!/usr/bin/env python3
"""
FinSentry Hackathon Demo Script
Uploads audio, processes transcript, and displays finance analysis.

Usage:
  # Use venv Python (recommended):
  backend/.venv/bin/python scripts/judge_demo.py <audio_file.wav>
  
  # Or if venv is activated:
  python scripts/judge_demo.py <audio_file.wav>
  
  # Or from repo root with make:
  make demo ARGS="<audio_file.wav>"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: requests. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

# Resolve repo root from this script's location
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
VENV_PYTHON = BACKEND / ".venv" / "bin" / "python"

# If a backend venv exists and we're not already using it, re-exec into it
try:
    if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
except Exception:
    # Fallback: continue with current interpreter; import errors will be shown later
    pass

# Ensure backend is on path for local audio pipeline
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/api/health"
UPLOAD_URL = f"{BASE_URL}/api/audio/upload"
ANALYZE_URL = f"{BASE_URL}/api/finance/analyze"
# End-to-end calls can be slow on first run (model downloads, cold start), so allow a longer timeout.
TIMEOUT = 600


def check_backend_health() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def upload_audio(audio_path: Path) -> str:
    """Upload audio file and return file_id."""
    print(f"📤 Uploading {audio_path.name}...", flush=True)
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        response = requests.post(UPLOAD_URL, files=files, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        file_id = result.get("file_id")
        if not file_id:
            raise ValueError("Upload response missing file_id")
        print(f"✓ Upload successful (file_id: {file_id})", flush=True)
        return file_id


def get_transcript(audio_path: Path, model_size: str = "medium") -> tuple[str, list[dict], dict]:
    """Get transcript using local audio pipeline."""
    print("🎙️  Processing audio to transcript...", flush=True)
    try:
        from ai.audio.pipeline import run_pipeline
    except ImportError as e:
        print(f"Error: Could not import audio pipeline: {e}", file=sys.stderr)
        if VENV_PYTHON.exists():
            print(
                f"\n💡 Solution: Use the venv Python instead:\n"
                f"   {VENV_PYTHON} scripts/judge_demo.py <audio_file>\n"
                f"   Or: make setup (to install dependencies)\n",
                file=sys.stderr,
            )
        else:
            print("Make sure backend dependencies are installed (make setup)", file=sys.stderr)
        sys.exit(1)
    
    try:
        output, _ = run_pipeline(audio_path, model_size=model_size)
        transcript = output.full_transcript
        segments = [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
            }
            for seg in output.segments
        ]
        stt_meta = {
            "detected_language": getattr(output, "detected_language", None),
            "language_probability": getattr(output, "language_probability", None),
            "avg_logprob": getattr(output, "avg_logprob", None),
        }
        print(f"✓ Transcript generated ({len(segments)} segments)", flush=True)
        return transcript, segments, stt_meta
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_transcript(transcript: str, segments: list[dict], call_id: str) -> dict:
    """Call finance analysis API."""
    print("🔍 Running finance analysis...", flush=True)
    payload = {
        "full_transcript": transcript,
        "segments": segments,
        "call_id": call_id,
        # Use rule-based / light path by default for faster, CPU-friendly demos.
        "use_llm_extraction": False,
    }
    response = requests.post(ANALYZE_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    result = response.json()
    print("✓ Analysis complete", flush=True)
    return result


def extract_summary(analysis: dict, stt_meta: dict | None = None) -> dict:
    """Extract key fields for display."""
    summary = {
        "transcript": analysis.get("full_transcript", ""),
        "intent": "N/A",
        "risk_level": "N/A",
        "amount": "N/A",
        "merchant": "N/A",
        "payment_method": "N/A",
        "language": "N/A",
        "avg_logprob": "N/A",
        "flags": [],
    }

    # Get call-level metrics
    call_metrics = analysis.get("call_metrics", {})
    if call_metrics:
        summary["intent"] = call_metrics.get("dominant_intent", "N/A")
        summary["risk_level"] = call_metrics.get("overall_risk_level", "N/A")

    if stt_meta:
        lang = stt_meta.get("detected_language")
        lang_prob = stt_meta.get("language_probability")
        avg_lp = stt_meta.get("avg_logprob")
        if lang:
            if isinstance(lang_prob, (int, float)):
                summary["language"] = f"{lang} ({lang_prob:.2f})"
            else:
                summary["language"] = lang
        if isinstance(avg_lp, (int, float)):
            summary["avg_logprob"] = f"{avg_lp:.3f}"
    
    # Extract entities (amounts, merchants)
    segments = analysis.get("segments", [])
    amounts: list[str] = []
    merchants: list[str] = []
    payment_methods: list[str] = []
    flags: list[str] = []
    
    for seg in segments:
        entities = seg.get("entities", [])
        for ent in entities:
            # FinancialEntity schema uses entity_type/text keys
            label = (ent.get("entity_type") or ent.get("label") or "").upper()
            text = ent.get("text", "")
            if "AMOUNT" in label or "MONEY" in label or "CURRENCY" in label:
                amounts.append(text)
            elif "MERCHANT" in label or "COMPANY" in label or "MERCHANT_NAME" in label:
                merchants.append(text)
            elif "CARD" in label or "UPI" in label or "PAYMENT_METHOD" in label or "WALLET" in label:
                payment_methods.append(text)
        
        # Check for risk flags
        risk = seg.get("risk_level", "").lower()
        if risk in ("high", "critical"):
            flags.append(f"High risk segment: {seg.get('text', '')[:50]}...")
    
    if amounts:
        summary["amount"] = ", ".join(sorted(set(amounts)))
    if merchants:
        summary["merchant"] = ", ".join(sorted(set(merchants)))
    if payment_methods:
        summary["payment_method"] = ", ".join(sorted(set(payment_methods)))
    if flags:
        summary["flags"] = flags
    
    return summary


def print_summary(summary: dict):
    """Print formatted summary."""
    print("\n" + "=" * 40)
    print("🎧 FinSentry Live Analysis")
    print("=" * 40)
    print(f"\nTranscript:")
    print(f"{summary['transcript'][:500]}{'...' if len(summary['transcript']) > 500 else ''}")
    print(f"\nLanguage: {summary['language']}")
    print(f"Intent: {summary['intent']}")
    print(f"Risk Level: {summary['risk_level']}")
    print(f"Amount: {summary['amount']}")
    print(f"Merchant: {summary['merchant']}")
    print(f"Payment Method: {summary['payment_method']}")
    print(f"STT avg logprob: {summary['avg_logprob']}")
    if summary['flags']:
        print(f"\nFlags:")
        for flag in summary['flags']:
            print(f"  ⚠️  {flag}")
    print("=" * 40 + "\n")


def save_json_report(analysis: dict, output_path: Path):
    """Save full JSON response."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"💾 Full report saved to: {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinSentry Hackathon Demo - Upload audio and get finance analysis"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file (.wav, .mp3, .m4a)",
    )
    parser.add_argument(
        "--model-size",
        default="medium",
        help="Whisper model size for STT (default: medium)",
    )
    args = parser.parse_args()
    
    audio_path = args.audio_file
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        return 1

    # Optionally convert to 16k mono WAV via ffmpeg for consistent ingest
    working_path = audio_path
    if audio_path.suffix.lower() not in {".wav"}:
        tmp_dir = REPO_ROOT / "demo_outputs" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"{audio_path.stem}_16k.wav"
        print(f"🎛️  Converting to 16k mono WAV via ffmpeg -> {tmp_wav}", flush=True)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(tmp_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            working_path = tmp_wav
        except FileNotFoundError:
            print("Error: ffmpeg not found on PATH. Install ffmpeg to enable format conversion.", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as e:
            print(f"Error: ffmpeg conversion failed: {e}", file=sys.stderr)
            return 1
    
    # Check backend health
    print("🔌 Checking backend connection...", flush=True)
    if not check_backend_health():
        print(f"Error: Backend not reachable at {BASE_URL}", file=sys.stderr)
        print("Make sure the server is running: make run", file=sys.stderr)
        return 1
    print("✓ Backend is running", flush=True)
    
    try:
        # Upload audio
        file_id = upload_audio(working_path)

        # Get transcript (using local pipeline)
        transcript, segments, stt_meta = get_transcript(working_path, model_size=args.model_size)

        # Analyze transcript
        analysis = analyze_transcript(transcript, segments, call_id=file_id)

        # Extract and display summary
        summary = extract_summary(analysis, stt_meta=stt_meta)
        print_summary(summary)

        # Save full JSON report
        output_dir = REPO_ROOT / "demo_outputs"
        output_path = output_dir / f"{audio_path.stem}_report.json"
        save_json_report(analysis, output_path)
        
        return 0
        
    except requests.HTTPError as e:
        print(f"HTTP error: {e.response.status_code}", file=sys.stderr)
        if e.response.text:
            print(f"Response: {e.response.text}", file=sys.stderr)
        # Save error report
        output_dir = REPO_ROOT / "demo_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / f"{audio_path.stem}_report.json"
        save_json_report({"error": str(e), "stage": "http", "status_code": e.response.status_code}, error_path)
        return 1
    except requests.RequestException as e:
        print(f"Request error: {e}", file=sys.stderr)
        output_dir = REPO_ROOT / "demo_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / f"{audio_path.stem}_report.json"
        save_json_report({"error": str(e), "stage": "request"}, error_path)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        output_dir = REPO_ROOT / "demo_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / f"{audio_path.stem}_report.json"
        save_json_report({"error": str(e), "stage": "unknown"}, error_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())

# AI-1 Batch Audio Pipeline

Transforms raw phone-call recordings into cleaned audio and timestamped transcripts for FinSentry (financial-event extraction, Backboard.io sync). Batch-only; no streaming.

## Quick Start

```bash
# From project root (venv activated)
python pipeline.py path/to/audio.wav
```

## Requirements

- **Python**: 3.10+
- **FFmpeg**: Must be on `PATH` (for MP3, M4A, OGG, FLAC)
- **Dependencies**: `pip install -r requirements.txt`

## CLI Usage

```bash
python pipeline.py AUDIO_PATH [--benchmark] [--save-intermediate] [--model-size base] [--call-id ID]
```

| Flag | Description |
|------|-------------|
| `AUDIO_PATH` | Input audio (required). `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac` |
| `--benchmark` | Print duration, processing time, realtime factor to stderr |
| `--save-intermediate` | Save resampled, post-DSP, chunked audio to `data/debug/` |
| `--model-size` | Whisper model: `base` (default), `small`, `medium`, `large-v2` |
| `--call-id` | Stable call_id for deterministic/reproducible runs |

### Examples

```bash
# Basic run
python pipeline.py sample.wav

# Benchmark + debug artifacts
python pipeline.py sample.wav --benchmark --save-intermediate

# Smaller/faster model
python pipeline.py sample.wav --model-size base

# Deterministic output (same call_id each run)
python pipeline.py sample.wav --call-id my_call_001
```

Alternative entry point:

```bash
python ai/audio/pipeline.py sample.wav --benchmark
```

## Model Sizes (CPU)

| Model | Disk | RAM | Speed |
|-------|------|-----|-------|
| base | ~150 MB | ~1 GB | ~0.1–0.3x realtime |
| small | ~500 MB | ~1.5 GB | slower |
| medium | ~1.5 GB | ~3 GB | slower |

First run downloads to `~/.cache/huggingface` (or `HF_HOME`).

## CPU vs GPU

- **CPU**: Default. `int8` compute. Works everywhere.
- **GPU**: Auto-detected (PyTorch + CUDA). `float16` for faster inference.

Force CPU: `export CUDA_VISIBLE_DEVICES=""`

## Output

**Success** — JSON to stdout:
```json
{
  "call_id": "a1b2c3d4e5f6",
  "audio_metadata": {
    "duration_sec": 45.2,
    "sample_rate": 16000,
    "chunks": 1,
    "cleaned": true
  },
  "segments": [
    {"start": 0.0, "end": 3.1, "speaker": null, "text": "Hello."}
  ],
  "full_transcript": "Hello.",
  "model": "faster-whisper-base",
  "processing_time_sec": 12.34
}
```

**Error** — Structured JSON (exit 1):
```json
{
  "success": false,
  "error": "Audio too short: 1.20s. Minimum 2.0s required.",
  "error_code": "VALIDATION_ERROR",
  "call_id": null
}
```

## Common Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `File not found` | Wrong path | Use absolute or correct relative path |
| `Unsupported format` | Bad extension | Use .wav, .mp3, .m4a, .ogg, .flac |
| `FFmpeg failed to decode` | Missing FFmpeg / corrupted file | `brew install ffmpeg`; check file integrity |
| `Audio too short` | Clip < 2 s | Minimum 2 seconds required |
| `Corrupted or invalid audio` | Damaged file, wrong format | Re-encode or use different file |
| Hugging Face Xet panic | HF downloader bug | `export HF_HUB_DISABLE_XET=1` |
| `ModuleNotFoundError` | Wrong cwd or path | Run from project root |

## Pipeline Stages

1. **Ingest** — Validate, generate call_id, copy to `data/raw/`
2. **Preprocess** — Resample to mono 16 kHz (stereo auto-converted)
3. **DSP** — Band-pass 300–3400 Hz, noise reduction, RMS normalization
4. **Silence** — Trim leading/trailing silence
5. **Chunk** — Split into ≤60 s segments
6. **STT** — faster-whisper transcription with timestamps

## Benchmark Output

```
--- Benchmark ---
  Input:        sample.wav
  Duration:     45.20 s
  Process time: 12.34 s
  Realtime:     0.27x (1x = realtime)
  Stage breakdown:
    ingest: 0.005s (0%)
    preprocess: 1.173s (10%)
    dsp: 0.029s (0%)
    silence: 0.598s (5%)
    chunk: 0.000s (0%)
    stt: 10.534s (85%)
-----------------
```

Realtime &lt; 1 = faster than realtime.

## Deterministic Output

- Segments sorted by `start` time
- Float precision: 2 decimal places
- `--call-id` for stable call identification across runs

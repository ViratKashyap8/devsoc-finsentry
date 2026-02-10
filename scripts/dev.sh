#!/usr/bin/env bash
# FinSentry dev runner. Usage: ./scripts/dev.sh <setup|run|test-audio|test-finance|transcribe|bootstrap>
# Run from repo root. Uses backend/.venv when present (see README).

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BACKEND="$REPO_ROOT/backend"
SCRIPTS="$REPO_ROOT/scripts"
VENV="$BACKEND/.venv"
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
  UVICORN="$VENV/bin/uvicorn"
else
  PY="${PY:-python3}"
  UVICORN=uvicorn
fi

cmd="${1:-help}"
case "$cmd" in
  setup)
    "$PY" "$SCRIPTS/bootstrap.py" || true
    if command -v uv >/dev/null 2>&1; then
      (cd "$BACKEND" && uv sync) && echo "Deps installed (uv)."
    else
      if [ ! -d "$VENV" ]; then
        "${PY:-python3}" -m venv "$VENV"
        PY="$VENV/bin/python"
      fi
      "$PY" -m pip install -r "$REPO_ROOT/requirements.txt" && echo "Deps installed (pip into $VENV)."
    fi
    "$PY" "$SCRIPTS/bootstrap.py"
    ;;
  run)
    (cd "$BACKEND" && "$UVICORN" main:app --reload --host 0.0.0.0 --port 8000)
    ;;
  test-audio)
    "$PY" "$SCRIPTS/test_audio_pipeline.py"
    ;;
  test-finance)
    "$PY" "$SCRIPTS/test_finance_pipeline.py"
    ;;
  transcribe)
    shift
    "$PY" "$SCRIPTS/transcribe_new_calls.py" "$@"
    ;;
  bootstrap)
    "$PY" "$SCRIPTS/bootstrap.py"
    ;;
  help|*)
    echo "Usage: $0 <setup|run|test-audio|test-finance|transcribe|bootstrap>"
    echo "  setup       - Install deps and run bootstrap"
    echo "  run         - Start API (uvicorn)"
    echo "  test-audio  - Run AI-1 audio pipeline demo"
    echo "  test-finance - Run AI-2 finance pipeline demo"
    echo "  transcribe - Batch-transcribe .wav (pass --overwrite etc. as extra args)"
    echo "  bootstrap   - Check Python, ffmpeg, env vars"
    exit 0
    ;;
esac

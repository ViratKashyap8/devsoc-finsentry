#!/usr/bin/env python3
"""
Bootstrap script: verify Python version, ffmpeg, and required environment variables.
Run from repo root: python scripts/bootstrap.py
Exits 0 if all checks pass; 1 with friendly error messages otherwise.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 10)
REQUIRED_ENV = [
    "BACKBOARD_API_KEY",
    "BACKBOARD_BASE_URL",
]


def _python_ok() -> tuple[bool, str]:
    if sys.version_info < REQUIRED_PYTHON:
        return False, (
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required; "
            f"current: {sys.version_info.major}.{sys.version_info.minor}"
        )
    return True, f"Python {sys.version_info.major}.{sys.version_info.minor} OK"


def _ffmpeg_ok() -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, (
            "ffmpeg not found on PATH. Audio pipeline needs it for MP3/M4A decoding.\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  Windows:  choco install ffmpeg  or download from https://ffmpeg.org"
        )
    try:
        out = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return False, "ffmpeg -version failed"
        first_line = (out.stdout or out.stderr or "").split("\n")[0].strip()
        return True, first_line or "ffmpeg found"
    except Exception as e:
        return False, f"ffmpeg check failed: {e}"


def _env_ok() -> tuple[bool, list[str]]:
    missing = []
    for key in REQUIRED_ENV:
        val = os.environ.get(key)
        if not val or not str(val).strip():
            missing.append(key)
    if missing:
        return False, missing
    return True, []


def _load_dotenv_simple(path: Path) -> None:
    """Parse backend/.env and set os.environ (stdlib-only, no python-dotenv required)."""
    if not path.exists():
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    key = k.strip()
                    val = v.strip().strip("'\"").strip()
                    if key:
                        os.environ.setdefault(key, val)
    except OSError:
        pass


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    _load_dotenv_simple(repo_root / "backend" / ".env")

    checks: list[tuple[str, bool, str]] = []
    # Python
    ok, msg = _python_ok()
    checks.append(("Python", ok, msg))
    # ffmpeg
    ok, msg = _ffmpeg_ok()
    checks.append(("ffmpeg", ok, msg))
    # Environment
    ok, missing = _env_ok()
    if ok:
        checks.append(("Environment", True, "BACKBOARD_API_KEY, BACKBOARD_BASE_URL set"))
    else:
        checks.append((
            "Environment",
            False,
            "Missing: " + ", ".join(missing)
            + ". Create backend/.env (copy from backend/.env.example) and set these keys, then run make setup again.",
        ))

    # Print what passed first (stdout), then any failures (stderr)
    for name, ok, msg in checks:
        if ok:
            print(f"  {name}: {msg}")
    sys.stdout.flush()

    errors = [(n, m) for n, o, m in checks if not o]
    if errors:
        for name, msg in errors:
            print(f"  [{name}] {msg}", file=sys.stderr)
        print(file=sys.stderr)
        print("Bootstrap failed. Fix the above and run make setup again.", file=sys.stderr)
        print("  See README for OS-level deps (ffmpeg) and backend/.env.example for env vars.", file=sys.stderr)
        return 1
    print("\nBootstrap OK. Run: make run")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# Load .env from backend/ (parent of core/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# Required for API startup (validated in validate_required_env()). Only Backboard credentials.
BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY")
BACKBOARD_BASE_URL = os.getenv("BACKBOARD_BASE_URL")

# Legacy alias
BACKBOARD_KEY = BACKBOARD_API_KEY

REQUIRED_ENV_KEYS = ("BACKBOARD_API_KEY", "BACKBOARD_BASE_URL")


def validate_required_env() -> None:
    """Raise RuntimeError if any required env var is missing. Call at app startup."""
    missing = [k for k in REQUIRED_ENV_KEYS if not os.getenv(k) or not str(os.getenv(k)).strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables. Set them in backend/.env or export before starting:\n  "
            + ", ".join(missing)
            + "\nCreate backend/.env from backend/.env.example and fill in the keys."
        )

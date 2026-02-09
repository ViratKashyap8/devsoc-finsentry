import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# Load .env from backend/ (parent of core/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

BACKBOARD_KEY = os.getenv("BACKBOARD_API_KEY")

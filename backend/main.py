import core.config  # noqa: F401 - loads .env and BACKBOARD_KEY before other imports
core.config.validate_required_env()

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import audio, finance, general  # pyright: ignore[reportImplicitRelativeImport]

logger = logging.getLogger(__name__)

# Set Hugging Face cache to workspace directory (avoids permission issues)
_backend_dir = Path(__file__).resolve().parent
_hf_cache = _backend_dir / ".cache" / "huggingface"
_hf_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf_cache))
# Disable Xet downloader (avoids Rust panic on macOS)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Application startup")
    yield
    logger.info("Application shutdown - waiting for background tasks to complete")
    # Give background tasks time to complete during reload
    await asyncio.sleep(1)


app = FastAPI(lifespan=lifespan)

api = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://frontend:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api.include_router(general.router)
api.include_router(audio.router)
api.include_router(finance.router)
app.include_router(api)

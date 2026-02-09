import core.config  # noqa: F401 - loads .env and BACKBOARD_KEY before other imports

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import audio, finance, general  # pyright: ignore[reportImplicitRelativeImport]

app = FastAPI()

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

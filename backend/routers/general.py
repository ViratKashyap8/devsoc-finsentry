from fastapi import APIRouter
from schemas.message import Message

router = APIRouter()


@router.get("/health", response_model=Message)
def health():
    return Message(message="ok va!")

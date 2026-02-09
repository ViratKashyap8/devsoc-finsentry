from pydantic import BaseModel


class Audio(BaseModel):
    file_id: str


class AudioResponse(Audio):
    status: str

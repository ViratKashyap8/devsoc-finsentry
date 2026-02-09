from typing import final

from repositories.audio_repository import (  # pyright: ignore[reportImplicitRelativeImport]
    AudioRepository,
)


@final
class AudioService:
    def __init__(self, repo: AudioRepository):
        self.repo = repo

    async def create_audio_file(self, file: bytes):
        return await self.repo.create_audio_file(file)

    async def retrieve_audio_file(self, file_id: str):
        return await self.repo.retrieve_audio_file(file_id)


_file_repository = AudioRepository()
_file_service = AudioService(_file_repository)


def get_file_service() -> AudioService:
    return _file_service

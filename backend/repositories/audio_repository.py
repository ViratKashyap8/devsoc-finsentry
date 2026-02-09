from typing import final
from uuid import uuid4


@final
class AudioRepository:
    def __init__(self):
        self.files: dict[str, str] = {}
        pass

    async def create_audio_file(self, file: bytes):
        new_uuid = str(uuid4())
        self.files[new_uuid] = f"file registered with uuid: {new_uuid}"
        return new_uuid

    async def retrieve_audio_file(self, file_id: str):
        print(self.files)
        return self.files.get(file_id)

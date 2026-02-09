from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from schemas.audio import Audio, AudioResponse
from services.audio_service import AudioService, get_file_service

router = APIRouter(prefix="/audio")


@router.post("/upload", response_model=Audio)
async def upload_audio_file(
    file: UploadFile, service: Annotated[AudioService, Depends(get_file_service)]
):
    content = await file.read()
    file_id = await service.create_audio_file(content)
    return Audio(file_id=file_id)


@router.get("/{file_id}", response_model=AudioResponse)
async def get_audio_file_details(
    file_id: str, service: Annotated[AudioService, Depends(get_file_service)]
):
    file = await service.retrieve_audio_file(file_id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"wrong file_id: {file_id} va!",
        )

    return AudioResponse(file_id="something", status=file)

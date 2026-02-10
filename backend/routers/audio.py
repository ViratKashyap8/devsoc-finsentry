from typing import Annotated

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ai.audio.pipeline import run_pipeline  # pyright: ignore[reportImplicitRelativeImport]
from ai.finance.pipeline import (  # pyright: ignore[reportImplicitRelativeImport]
    run_finance_analysis,
)
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
            detail=f"File not found: {file_id}",
        )

    return AudioResponse(file_id=file_id, status=file)


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    model_size: str = "medium",
    use_llm_extraction: bool = False,
):
    """
    End-to-end audio analysis for demo:
    - Accepts an uploaded audio file
    - Runs AI-1 audio pipeline (ingest → preprocess → DSP → silence → chunk → STT)
    - Runs AI-2 finance analysis on the resulting transcript
    - Returns a FinanceAnalysisOutput JSON payload
    """
    try:
        suffix = Path(file.filename or "audio").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store uploaded file: {exc}",
        ) from exc

    loop = asyncio.get_running_loop()

    try:
        # Run audio pipeline (CPU heavy) in executor
        pipeline_output, _ = await loop.run_in_executor(
            None, lambda: run_pipeline(tmp_path, model_size=model_size)
        )

        # Prepare segments for finance pipeline
        segments = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in pipeline_output.segments
        ]

        # Run finance analysis in executor as well
        finance_output = await loop.run_in_executor(
            None,
            lambda: run_finance_analysis(
                pipeline_output.full_transcript,
                segments,
                call_id=pipeline_output.call_id,
                use_llm_extraction=use_llm_extraction,
                detected_language=getattr(pipeline_output, "detected_language", None),
                language_probability=getattr(
                    pipeline_output, "language_probability", None
                ),
                avg_logprob=getattr(pipeline_output, "avg_logprob", None),
            ),
        )

        return finance_output.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


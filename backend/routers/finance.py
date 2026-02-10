"""
Finance Intelligence API routes.

Endpoints for transcript analysis, RAG Q&A.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/finance")


class AnalyzeRequest(BaseModel):
    """Request for transcript analysis."""

    full_transcript: str = Field(..., min_length=1)
    segments: Optional[list[dict]] = None
    call_id: str = "unknown"
    use_llm_extraction: bool = True


class RAGQueryRequest(BaseModel):
    """Request for RAG Q&A."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


_rag_generator = None


def _get_rag():
    global _rag_generator
    if _rag_generator is None:
        from ai.finance.rag import FinanceRetriever, RAGGenerator
        # Default: in-memory with sample docs. Replace with your corpus.
        docs = [
            {"id": "1", "text": "Refunds are processed within 5-7 business days."},
            {"id": "2", "text": "Late fees may be waived once per year upon request."},
            {"id": "3", "text": "Payment plans allow up to 12 monthly installments."},
        ]
        retriever = FinanceRetriever(documents=docs)
        _rag_generator = RAGGenerator(retriever)
    return _rag_generator


@router.post("/analyze")
async def analyze_transcript(req: AnalyzeRequest):
    """
    Run Finance Intelligence analysis on transcript.

    Returns intent, entities, obligations, risk, emotion per segment and call-level metrics.
    """
    loop = asyncio.get_running_loop()
    try:
        from ai.finance.pipeline import FinancePipeline
        pipeline = FinancePipeline(use_llm_extraction=req.use_llm_extraction)
        output = await loop.run_in_executor(
            None,
            lambda: pipeline.analyze(
                req.full_transcript,
                segments=req.segments,
                call_id=req.call_id,
            ),
        )
        return output.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    """RAG Q&A over finance documents."""
    loop = asyncio.get_running_loop()
    try:
        gen = _get_rag()
        result = await loop.run_in_executor(
            None,
            lambda: gen.generate(req.query, top_k=req.top_k),
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

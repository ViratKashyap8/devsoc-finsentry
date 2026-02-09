"""
Finance Intelligence & RAG module for fintech audio analytics.

Tasks: intent classification, entity extraction, obligation detection,
risk scoring, regulatory phrase detection, emotion analysis, RAG.

All models run locally (no external LLM APIs).
"""

from .schema import (
    FinanceAnalysisOutput,
    IntentLabel,
    RiskLevel,
    ObligationType,
    EmotionLabel,
)
from .pipeline import FinancePipeline, run_finance_analysis

__all__ = [
    "FinancePipeline",
    "run_finance_analysis",
    "FinanceAnalysisOutput",
    "IntentLabel",
    "RiskLevel",
    "ObligationType",
    "EmotionLabel",
]

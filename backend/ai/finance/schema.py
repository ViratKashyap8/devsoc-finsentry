"""
Pydantic schemas for Finance Intelligence outputs.

Compatible with audio pipeline Segment format for per-utterance analysis.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntentLabel(str, Enum):
    """Call intent categories for financial conversations."""

    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    DISPUTE = "dispute"
    PAYMENT_ARRANGEMENT = "payment_arrangement"
    ACCOUNT_INFO = "account_info"
    CLOSURE_REQUEST = "closure_request"
    FRAUD_REPORT = "fraud_report"
    GENERAL = "general"


class RiskLevel(str, Enum):
    """Risk severity for detected issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    NONE = "none"


class ObligationType(str, Enum):
    """Types of promises/obligations made during calls."""

    PAYMENT_PROMISE = "payment_promise"
    FOLLOW_UP = "follow_up"
    DOCUMENT_SEND = "document_send"
    REFUND = "refund"
    FEE_WAIVER = "fee_waiver"
    ESCALATION = "escalation"
    OTHER = "other"


class EmotionLabel(str, Enum):
    """Emotion/stress indicators."""

    NEUTRAL = "neutral"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    CALM = "calm"
    UPSET = "upset"
    STRESSED = "stressed"


class FinancialEntity(BaseModel):
    """Extracted financial entity from transcript."""

    text: str
    entity_type: str  # e.g. "AMOUNT", "ACCOUNT", "DATE", "PRODUCT", "FEE"
    span_start: int = 0
    span_end: int = 0
    normalized_value: str | None = None  # e.g. "$500" -> "500.00 USD"
    model_config = ConfigDict(frozen=False)


class RegulatoryPhrase(BaseModel):
    """Detected regulatory/compliance phrase."""

    text: str
    category: str  # e.g. "DISCLAIMER", "CONSENT", "REQUIRED_DISCLOSURE"
    span_start: int = 0
    span_end: int = 0
    model_config = ConfigDict(frozen=False)


class Obligation(BaseModel):
    """Detected obligation or promise."""

    text: str
    obligation_type: ObligationType
    due_date: str | None = None
    amount: str | None = None
    span_start: int = 0
    span_end: int = 0
    model_config = ConfigDict(frozen=False)


class SegmentAnalysis(BaseModel):
    """Per-segment analysis (maps to audio pipeline Segment)."""

    start: float = 0.0
    end: float = 0.0
    text: str
    intent: IntentLabel | None = None
    intent_confidence: float = 0.0
    entities: list[FinancialEntity] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    regulatory_phrases: list[RegulatoryPhrase] = Field(default_factory=list)
    emotion: EmotionLabel | None = None
    emotion_confidence: float = 0.0
    stress_score: float = 0.0  # 0.0 = calm, 1.0 = highly stressed
    model_config = ConfigDict(frozen=False)


class CallLevelMetrics(BaseModel):
    """Aggregated call-level metrics."""

    dominant_intent: IntentLabel | None = None
    overall_risk_level: RiskLevel = RiskLevel.NONE
    risk_score: float = 0.0  # 0.0–1.0
    risk_factors: list[str] = Field(default_factory=list)
    total_obligations: int = 0
    obligation_summary: list[dict[str, Any]] = Field(default_factory=list)
    stress_trend: str | None = None  # "increasing", "decreasing", "stable"
    regulatory_compliance_score: float = 1.0  # 0.0–1.0
    model_config = ConfigDict(frozen=False)


class RAGResponse(BaseModel):
    """RAG retrieval + generation response."""

    query: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    answer: str
    sources: list[str] = Field(default_factory=list)
    model_config = ConfigDict(frozen=False)


class FinanceAnalysisOutput(BaseModel):
    """Complete Finance Intelligence output for a call transcript."""

    call_id: str
    full_transcript: str
    segments: list[SegmentAnalysis]
    call_metrics: CallLevelMetrics
    all_entities: list[FinancialEntity] = Field(default_factory=list)
    all_obligations: list[Obligation] = Field(default_factory=list)
    all_regulatory: list[RegulatoryPhrase] = Field(default_factory=list)
    processing_time_sec: float = 0.0
    # Optional STT metadata for end-to-end audio + finance flows
    detected_language: str | None = None
    language_probability: float | None = None
    avg_logprob: float | None = None
    model_config = ConfigDict(frozen=False)

"""
Finance Intelligence pipeline: full analysis of call transcripts.

Orchestrates intent, emotion, entity, obligation, regulatory, risk.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from .dataset.preprocess import normalize_text, preprocess_transcript
from .models.classifier import classify_emotion, classify_intent, stress_score_from_emotion
from .models.extractor import (
    extract_entities,
    extract_obligations,
    extract_regulatory_phrases,
)
from .models.risk import aggregate_call_risk, score_risk_from_text
from .schema import (
    CallLevelMetrics,
    EmotionLabel,
    FinanceAnalysisOutput,
    FinancialEntity,
    IntentLabel,
    Obligation,
    ObligationType,
    RegulatoryPhrase,
    RiskLevel,
    SegmentAnalysis,
)

logger = logging.getLogger(__name__)


class FinancePipeline:
    """
    Finance Intelligence analysis pipeline.

    Processes transcripts into structured analysis. Uses hybrid:
    - Zero-shot classification (intent, emotion)
    - LLM extraction (entities, obligations)
    - Rule-based (regulatory, risk aggregation)
    """

    def __init__(
        self,
        use_llm_extraction: bool = True,
        batch_classify: bool = True,
    ):
        self.use_llm_extraction = use_llm_extraction
        self.batch_classify = batch_classify

    def analyze_segment(
        self,
        text: str,
        start: float = 0.0,
        end: float = 0.0,
        intent: Optional[str] = None,
        intent_conf: float = 0.0,
        emotion: Optional[str] = None,
        emotion_conf: float = 0.0,
    ) -> SegmentAnalysis:
        """Analyze single transcript segment."""
        text = normalize_text(text)
        if not text:
            return SegmentAnalysis(start=start, end=end, text="")

        # Classification (can be pre-computed in batch)
        if intent is None:
            intents = classify_intent([text])
            intent, intent_conf = intents[0] if intents else ("general", 0.0)
        if emotion is None:
            emotions = classify_emotion([text])
            emotion, emotion_conf = emotions[0] if emotions else ("neutral", 0.0)

        stress = stress_score_from_emotion(emotion) if emotion else 0.5
        _, _, risk_factors = score_risk_from_text(text)

        # Extraction
        entities = []
        obligations = []
        regulatory = []
        if self.use_llm_extraction:
            entities = extract_entities(text)
            obligations = extract_obligations(text)
            regulatory = extract_regulatory_phrases(text)
        else:
            from .models.extractor import _fallback_entity_extraction, _rule_based_regulatory
            entities = _fallback_entity_extraction(text)
            regulatory = _rule_based_regulatory(text)

        try:
            intent_enum = IntentLabel(intent) if intent else None
        except ValueError:
            intent_enum = IntentLabel.GENERAL
        try:
            emotion_enum = EmotionLabel(emotion) if emotion else None
        except ValueError:
            emotion_enum = EmotionLabel.NEUTRAL

        return SegmentAnalysis(
            start=start,
            end=end,
            text=text,
            intent=intent_enum,
            intent_confidence=intent_conf,
            entities=entities,
            obligations=obligations,
            regulatory_phrases=regulatory,
            emotion=emotion_enum,
            emotion_confidence=emotion_conf,
            stress_score=stress,
        )

    def analyze(
        self,
        full_transcript: str,
        segments: Optional[list[dict]] = None,
        call_id: str = "unknown",
    ) -> FinanceAnalysisOutput:
        """
        Run full pipeline on transcript.

        Args:
            full_transcript: Full call transcript
            segments: Optional list of {start, end, text} from STT
            call_id: Call identifier

        Returns:
            FinanceAnalysisOutput with all analysis
        """
        t0 = time.perf_counter()
        full, seg_list = preprocess_transcript(full_transcript, segments)

        if not seg_list:
            # Create single segment from full transcript
            seg_list = [{"start": 0.0, "end": 0.0, "text": full}]

        # Batch classify for speed
        texts = [s["text"] for s in seg_list]
        intents = classify_intent(texts) if texts else []
        emotions = classify_emotion(texts) if texts else []

        segment_analyses = []
        all_entities = []
        all_obligations = []
        all_regulatory = []
        intent_labels = []
        risk_factors_per_seg = []

        for i, seg in enumerate(seg_list):
            intent, intent_conf = intents[i] if i < len(intents) else ("general", 0.0)
            emotion, emotion_conf = emotions[i] if i < len(emotions) else ("neutral", 0.0)
            sa = self.analyze_segment(
                seg["text"],
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                intent=intent,
                intent_conf=intent_conf,
                emotion=emotion,
                emotion_conf=emotion_conf,
            )
            segment_analyses.append(sa)
            all_entities.extend(sa.entities)
            all_obligations.extend(sa.obligations)
            all_regulatory.extend(sa.regulatory_phrases)
            intent_labels.append(sa.intent.value if sa.intent else None)
            risk_factors_per_seg.append(
                [f for f in score_risk_from_text(seg["text"])[2]]
            )

        # Call-level metrics
        risk_level, risk_score, risk_factors = aggregate_call_risk(
            intent_labels, risk_factors_per_seg
        )
        dominant_intent = _mode([il for il in intent_labels if il])
        stress_scores = [s.stress_score for s in segment_analyses]
        stress_trend = _trend(stress_scores) if len(stress_scores) > 2 else None
        reg_score = 1.0 - (0.1 * len(all_regulatory)) if all_regulatory else 1.0
        reg_score = max(0.0, reg_score)

        call_metrics = CallLevelMetrics(
            dominant_intent=IntentLabel(dominant_intent) if dominant_intent else None,
            overall_risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors[:15],
            total_obligations=len(all_obligations),
            obligation_summary=[
                {"type": o.obligation_type.value, "text": o.text[:80]}
                for o in all_obligations[:10]
            ],
            stress_trend=stress_trend,
            regulatory_compliance_score=round(reg_score, 2),
        )

        elapsed = time.perf_counter() - t0
        return FinanceAnalysisOutput(
            call_id=call_id,
            full_transcript=full,
            segments=segment_analyses,
            call_metrics=call_metrics,
            all_entities=all_entities,
            all_obligations=all_obligations,
            all_regulatory=all_regulatory,
            processing_time_sec=round(elapsed, 2),
        )


def _mode(items: list) -> str | None:
    """Return most common item."""
    if not items:
        return None
    from collections import Counter
    return Counter(items).most_common(1)[0][0]


def _trend(values: list[float]) -> str:
    """Simple trend: increasing, decreasing, stable."""
    if len(values) < 2:
        return "stable"
    first_half = sum(values[: len(values) // 2]) / max(1, len(values) // 2)
    second_half = sum(values[len(values) // 2 :]) / max(1, len(values) - len(values) // 2)
    diff = second_half - first_half
    if diff > 0.1:
        return "increasing"
    if diff < -0.1:
        return "decreasing"
    return "stable"


def run_finance_analysis(
    full_transcript: str,
    segments: Optional[list[dict]] = None,
    call_id: str = "unknown",
    use_llm_extraction: bool = True,
    detected_language: str | None = None,
    language_probability: float | None = None,
    avg_logprob: float | None = None,
) -> FinanceAnalysisOutput:
    """
    Convenience function to run full finance analysis.

    Args:
        full_transcript: Raw transcript text
        segments: Optional [{start, end, text}] from audio pipeline
        call_id: Call ID
        use_llm_extraction: If False, use rule-based extraction only (faster)

    Returns:
        FinanceAnalysisOutput
    """
    pipeline = FinancePipeline(use_llm_extraction=use_llm_extraction)
    output = pipeline.analyze(full_transcript, segments, call_id)
    # attach STT meta when available
    output.detected_language = detected_language
    output.language_probability = language_probability
    output.avg_logprob = avg_logprob

    # Light-weight retry: if transcript looks financial but intent is too generic,
    # normalize currency formats and Hindi numerals, then re-run once.
    def _has_financial_signals(text: str) -> bool:
        if not text:
            return False
        if any(ch in text for ch in ["₹", "रु", "Rs", "rs", "rupees", "rupay"]):
            return True
        # Hindi digits ०१२३४५६७८९
        if re.search(r"[०१२३४५६७८९]", text):
            return True
        # Amount-like patterns
        return bool(re.search(r"\d[\d,\.]*\s*(INR|rs|rupees)", text, flags=re.IGNORECASE))

    dominant_intent = output.call_metrics.dominant_intent
    if (
        (dominant_intent is None or dominant_intent == IntentLabel.GENERAL)
        and _has_financial_signals(full_transcript)
    ):
        logger.info("Retrying finance analysis with cleaned numeric/currency text")

        def _normalize_hindi_digits(s: str) -> str:
            mapping = {
                "०": "0",
                "१": "1",
                "२": "2",
                "३": "3",
                "४": "4",
                "५": "5",
                "६": "6",
                "७": "7",
                "८": "8",
                "९": "9",
            }
            return "".join(mapping.get(ch, ch) for ch in s)

        def _clean_currency(s: str) -> str:
            # Normalize rupee symbols/words and strip commas inside numbers.
            s = s.replace("₹", " Rs ")
            s = re.sub(r"\bरु\.?", " Rs ", s)
            s = re.sub(r"\brupees?\b", " Rs ", s, flags=re.IGNORECASE)
            # Remove thousands separators inside numbers (e.g., 1,23,456 -> 123456)
            s = re.sub(r"(?<=\d),(?=\d)", "", s)
            s = _normalize_hindi_digits(s)
            return s

        cleaned_full = _clean_currency(full_transcript)
        cleaned_segments = None
        if segments:
            cleaned_segments = []
            for seg in segments:
                seg_copy = dict(seg)
                seg_copy["text"] = _clean_currency(seg_copy.get("text", ""))
                cleaned_segments.append(seg_copy)

        retry_output = pipeline.analyze(cleaned_full, cleaned_segments, call_id)
        retry_output.detected_language = detected_language
        retry_output.language_probability = language_probability
        retry_output.avg_logprob = avg_logprob
        if retry_output.call_metrics.dominant_intent and retry_output.call_metrics.dominant_intent != IntentLabel.GENERAL:
            return retry_output

    return output

"""
Risk scoring from intent, entities, and keywords.

Aggregates segment-level signals into call-level risk.
"""

from __future__ import annotations

import re
from collections import Counter

from ..schema import IntentLabel, RiskLevel


RISK_KEYWORDS = {
    RiskLevel.CRITICAL: [
        "fraud", "stolen", "identity theft", "unauthorized", "lawsuit",
        "attorney", "regulatory", "complaint to", "cfpb", "bbb",
    ],
    RiskLevel.HIGH: [
        "dispute", "chargeback", "escalate", "supervisor", "manager",
        "cancel", "closure", "refund", "never received", "billing error",
    ],
    RiskLevel.MEDIUM: [
        "frustrated", "angry", "unacceptable", "wrong", "error",
        "mistake", "late fee", "overdraft", "collection",
    ],
    RiskLevel.LOW: [
        "payment plan", "arrangement", "extension", "waive",
    ],
}

INTENT_RISK_MAP = {
    IntentLabel.FRAUD_REPORT: RiskLevel.CRITICAL,
    IntentLabel.COMPLAINT: RiskLevel.HIGH,
    IntentLabel.DISPUTE: RiskLevel.HIGH,
    IntentLabel.CLOSURE_REQUEST: RiskLevel.MEDIUM,
    IntentLabel.PAYMENT_ARRANGEMENT: RiskLevel.LOW,
    IntentLabel.INQUIRY: RiskLevel.NONE,
    IntentLabel.ACCOUNT_INFO: RiskLevel.NONE,
    IntentLabel.GENERAL: RiskLevel.NONE,
}


def score_risk_from_text(text: str) -> tuple[RiskLevel, float, list[str]]:
    """
    Compute risk level from raw text using keyword matching.

    Returns (risk_level, score_0_to_1, risk_factors).
    """
    text_lower = text.lower()
    factors = []
    max_level = RiskLevel.NONE
    level_scores = {RiskLevel.NONE: 0, RiskLevel.LOW: 0.2, RiskLevel.MEDIUM: 0.5, RiskLevel.HIGH: 0.8, RiskLevel.CRITICAL: 1.0}
    score = 0.0

    for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
        for kw in RISK_KEYWORDS[level]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                factors.append(kw)
                if level_scores[level] > level_scores[max_level]:
                    max_level = level

    if factors:
        score = min(0.95, 0.3 + 0.1 * len(factors))
        score = max(score, level_scores[max_level])

    return max_level, score, factors


def aggregate_call_risk(
    segment_intents: list[str | None],
    segment_risk_factors: list[list[str]],
) -> tuple[RiskLevel, float, list[str]]:
    """
    Aggregate segment-level risk into call-level.

    Args:
        segment_intents: Intent label per segment
        segment_risk_factors: Risk factors per segment

    Returns:
        (overall_risk_level, risk_score, combined_factors)
    """
    all_factors = []
    for f in segment_risk_factors:
        all_factors.extend(f)

    intent_risks = []
    for intent_str in segment_intents:
        if intent_str:
            try:
                intent = IntentLabel(intent_str)
                intent_risks.append(INTENT_RISK_MAP.get(intent, RiskLevel.NONE))
            except ValueError:
                pass

    level_order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    max_intent = (
        max(intent_risks, key=lambda r: level_order.index(r))
        if intent_risks
        else RiskLevel.NONE
    )
    _, kw_score, kw_factors = score_risk_from_text(" ".join(all_factors))
    combined = list(dict.fromkeys(all_factors + kw_factors))

    # Start from intent-based risk, elevate if keywords indicate higher
    final_level = max_intent
    for level in reversed(level_order):
        for kw in RISK_KEYWORDS.get(level, []):
            if any(kw in c.lower() for c in combined):
                if level_order.index(level) > level_order.index(final_level):
                    final_level = level
                break

    score = max(
        kw_score,
        level_order.index(final_level) / (len(level_order) - 1),
    )
    return final_level, min(1.0, score), combined[:20]

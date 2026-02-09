"""
Intent and emotion classification using sentence-transformers zero-shot.

No training required - uses semantic similarity to candidate labels.
Fast, runs on CPU, ~80MB model.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schema import EmotionLabel, IntentLabel

logger = logging.getLogger(__name__)

INTENT_LABELS = [e.value for e in IntentLabel]
EMOTION_LABELS = [e.value for e in EmotionLabel]

_model_cache: dict[str, Any] = {}


def _get_model(model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load sentence-transformers model."""
    if model_id not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
            _model_cache[model_id] = SentenceTransformer(model_id)
        except ImportError as e:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from e
    return _model_cache[model_id]


def _fallback_intent(text: str) -> tuple[str, float]:
    """Rule-based intent when sentence-transformers unavailable."""
    t = text.lower()
    if "dispute" in t or "chargeback" in t:
        return ("dispute", 0.7)
    if "refund" in t or "reimburse" in t:
        return ("inquiry", 0.6)
    if "payment plan" in t or "installment" in t:
        return ("payment_arrangement", 0.7)
    if "close" in t and "account" in t:
        return ("closure_request", 0.7)
    if "fraud" in t or "stolen" in t:
        return ("fraud_report", 0.8)
    if "balance" in t or "due date" in t:
        return ("account_info", 0.6)
    if "frustrated" in t or "angry" in t:
        return ("complaint", 0.6)
    return ("general", 0.5)


def _fallback_emotion(text: str) -> tuple[str, float]:
    """Rule-based emotion when sentence-transformers unavailable."""
    t = text.lower()
    if any(w in t for w in ["angry", "furious"]):
        return ("angry", 0.7)
    if any(w in t for w in ["anxious", "worried"]):
        return ("anxious", 0.6)
    if any(w in t for w in ["frustrated", "upset"]):
        return ("frustrated", 0.6)
    if any(w in t for w in ["calm", "thank you"]):
        return ("calm", 0.5)
    return ("neutral", 0.5)


def classify_intent(
    texts: list[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_fallback: bool = True,
) -> list[tuple[str, float]]:
    """
    Zero-shot intent classification.

    Returns list of (intent_label, confidence) per text.
    If sentence-transformers unavailable and use_fallback=True, uses rule-based.
    """
    if not texts:
        return []
    try:
        model = _get_model(model_id)
    except ImportError:
        if use_fallback:
            return [_fallback_intent(t) for t in texts]
        raise
    # Encode candidate labels
    label_embs = model.encode(INTENT_LABELS, convert_to_tensor=False)
    text_embs = model.encode(texts, convert_to_tensor=False)

    import numpy as np
    scores = np.dot(text_embs, label_embs.T)
    # Softmax over labels
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    results = []
    for i in range(len(texts)):
        best_idx = int(probs[i].argmax())
        results.append((INTENT_LABELS[best_idx], float(probs[i][best_idx])))
    return results


def classify_emotion(
    texts: list[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_fallback: bool = True,
) -> list[tuple[str, float]]:
    """
    Zero-shot emotion classification.

    Returns list of (emotion_label, confidence) per text.
    If sentence-transformers unavailable and use_fallback=True, uses rule-based.
    """
    if not texts:
        return []
    try:
        model = _get_model(model_id)
    except ImportError:
        if use_fallback:
            return [_fallback_emotion(t) for t in texts]
        raise
    label_embs = model.encode(EMOTION_LABELS, convert_to_tensor=False)
    text_embs = model.encode(texts, convert_to_tensor=False)

    import numpy as np
    scores = np.dot(text_embs, label_embs.T)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    results = []
    for i in range(len(texts)):
        best_idx = int(probs[i].argmax())
        results.append((EMOTION_LABELS[best_idx], float(probs[i][best_idx])))
    return results


def stress_score_from_emotion(emotion: str) -> float:
    """Map emotion label to 0-1 stress score."""
    high_stress = {"angry", "anxious", "frustrated", "upset", "stressed"}
    if emotion in high_stress:
        return 0.8
    if emotion == "calm":
        return 0.2
    return 0.5  # neutral

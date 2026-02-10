"""
Entity and obligation extraction using small instruction-tuned LLM.

Uses SmolLM2-360M or Qwen2-0.5B for structured JSON output.
Runs on CPU (slower) or consumer GPU.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..schema import FinancialEntity, Obligation, ObligationType, RegulatoryPhrase

logger = logging.getLogger(__name__)

_extraction_model_cache: Any = None


def _get_extraction_model(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    """Lazy-load extraction LLM."""
    global _extraction_model_cache
    if _extraction_model_cache is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading extraction model %s on %s", model_id, device)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                model = model.to(device)
            _extraction_model_cache = (tokenizer, model, device)
        except ImportError as e:
            raise ImportError(
                "Install transformers, torch, accelerate: pip install transformers torch accelerate"
            ) from e
    return _extraction_model_cache


ENTITY_EXTRACTION_PROMPT = """Extract financial entities from this call transcript. Return ONLY a JSON array, no other text.
Entity types: AMOUNT, ACCOUNT, DATE, PRODUCT, FEE, CARD_NUMBER, REFERENCE_NUMBER
Format: [{"text": "entity span", "entity_type": "TYPE", "normalized_value": "optional"}]

Transcript: {text}

JSON:"""

OBLIGATION_EXTRACTION_PROMPT = """Extract any obligations or promises made in this call. Return ONLY a JSON array, no other text.
Types: payment_promise, follow_up, document_send, refund, fee_waiver, escalation, other
Format: [{"text": "promise span", "obligation_type": "type", "amount": "if applicable", "due_date": "if applicable"}]
If none, return []

Transcript: {text}

JSON:"""

REGULATORY_PROMPT = """Identify regulatory or compliance phrases in this text. Return ONLY a JSON array.
Categories: DISCLAIMER, CONSENT, REQUIRED_DISCLOSURE, ID_VERIFICATION, TERMS_REFERENCE
Format: [{"text": "phrase", "category": "CATEGORY"}]
If none, return []

Text: {text}

JSON:"""


def _generate(
    tokenizer: Any, model: Any, device: str, prompt: str, max_new_tokens: int = 256
) -> str:
    """Run generation and return decoded text."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return decoded.strip()


def _parse_json_array(raw: str) -> list[dict]:
    """Parse JSON array from model output, handling common failures."""
    raw = raw.strip()
    # Find [...]
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def extract_entities(
    text: str,
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
) -> list[FinancialEntity]:
    """Extract financial entities from transcript segment."""
    if not text.strip():
        return []
    try:
        tokenizer, model, device = _get_extraction_model(model_id)
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:800])  # truncate for context
        raw = _generate(tokenizer, model, device, prompt)
        items = _parse_json_array(raw)
        entities = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                entities.append(
                    FinancialEntity(
                        text=str(item["text"]),
                        entity_type=str(item.get("entity_type", "OTHER")),
                        normalized_value=str(item["normalized_value"]) if item.get("normalized_value") else None,
                    )
                )
        return entities
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return _fallback_entity_extraction(text)


def _fallback_entity_extraction(text: str) -> list[FinancialEntity]:
    """Regex fallback when LLM unavailable."""
    entities = []
    for m in re.finditer(r"\$\d+(?:\.\d{2})?", text):
        entities.append(
            FinancialEntity(text=m.group(), entity_type="AMOUNT", span_start=m.start(), span_end=m.end())
        )
    for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        entities.append(
            FinancialEntity(text=m.group(), entity_type="DATE", span_start=m.start(), span_end=m.end())
        )
    return entities


def extract_obligations(
    text: str,
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
) -> list[Obligation]:
    """Extract obligations/promises from transcript segment."""
    if not text.strip():
        return []
    try:
        tokenizer, model, device = _get_extraction_model(model_id)
        prompt = OBLIGATION_EXTRACTION_PROMPT.format(text=text[:800])
        raw = _generate(tokenizer, model, device, prompt)
        items = _parse_json_array(raw)
        obligations = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                obl_type = item.get("obligation_type", "other")
                try:
                    obl_enum = ObligationType(obl_type)
                except ValueError:
                    obl_enum = ObligationType.OTHER
                obligations.append(
                    Obligation(
                        text=str(item["text"]),
                        obligation_type=obl_enum,
                        amount=str(item["amount"]) if item.get("amount") else None,
                        due_date=str(item["due_date"]) if item.get("due_date") else None,
                    )
                )
        return obligations
    except Exception as e:
        logger.warning("Obligation extraction failed: %s", e)
        return []


def extract_regulatory_phrases(
    text: str,
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
) -> list[RegulatoryPhrase]:
    """Extract regulatory/compliance phrases."""
    phrases = _rule_based_regulatory(text)
    if not phrases and text.strip():
        try:
            tokenizer, model, device = _get_extraction_model(model_id)
            prompt = REGULATORY_PROMPT.format(text=text[:600])
            raw = _generate(tokenizer, model, device, prompt, max_new_tokens=128)
            items = _parse_json_array(raw)
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    phrases.append(
                        RegulatoryPhrase(
                            text=str(item["text"]),
                            category=str(item.get("category", "OTHER")),
                        )
                    )
        except Exception as e:
            pass
    return phrases


def _rule_based_regulatory(text: str) -> list[RegulatoryPhrase]:
    """Rule-based regulatory phrase detection (fast, no model)."""
    patterns = [
        (r"call may be recorded", "REQUIRED_DISCLOSURE"),
        (r"quality assurance", "REQUIRED_DISCLOSURE"),
        (r"for security(?: purposes)?", "ID_VERIFICATION"),
        (r"verify your identity", "ID_VERIFICATION"),
        (r"terms and conditions", "TERMS_REFERENCE"),
        (r"do you consent", "CONSENT"),
        (r"agree to", "CONSENT"),
    ]
    phrases = []
    for pattern, cat in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            phrases.append(RegulatoryPhrase(text=m.group(), category=cat, span_start=m.start(), span_end=m.end()))
    return phrases

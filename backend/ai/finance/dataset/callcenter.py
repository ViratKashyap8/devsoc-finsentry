"""
Call center dataset processing for Finance Intelligence training.

Parses .docx transcripts from the Hugging Face call center dataset and converts
them to FinanceTrainExample format for LLM fine-tuning.

Usage:
    python -m ai.finance.dataset.callcenter \
        --input data/audio/Call\ center\ data\ samples \
        --output data/finance/callcenter
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from docx import Document

from .format import FinanceTrainExample, InstructionExample, save_dataset_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intent detection patterns
INTENT_PATTERNS = {
    "dispute": ["dispute", "chargeback", "unauthorized", "didn't make this"],
    "inquiry": ["question", "when will", "how long", "what is", "can you tell me"],
    "payment_arrangement": ["payment plan", "installment", "pay later", "arrange payment"],
    "complaint": ["frustrated", "angry", "terrible", "worst", "unacceptable", "complaint"],
    "fraud_report": ["fraud", "stolen", "scam", "unauthorized"],
    "closure_request": ["close my account", "cancel", "terminate"],
    "account_info": ["balance", "statement", "account number", "due date"],
    "general": [],
}

# Entity extraction patterns
ENTITY_PATTERNS = {
    "AMOUNT": r"\$[\d,]+(?:\.\d{2})?|\d+\s*(?:dollars|USD|euro|EUR)",
    "DATE": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:today|tomorrow|yesterday|next\s+\w+|this\s+\w+)\b",
    "ACCOUNT": r"\b(?:account|card)\s*(?:number|#|ending in)?\s*[\d\-*]+\b",
    "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "REFERENCE": r"\b(?:ref|reference|ticket|case)\s*#?\s*\d+\b",
}

# Emotion keywords
EMOTION_KEYWORDS = {
    "angry": ["angry", "furious", "outraged", "mad"],
    "frustrated": ["frustrated", "annoyed", "irritated", "upset"],
    "anxious": ["worried", "concerned", "anxious", "nervous"],
    "calm": ["calm", "patient", "cooperative", "understanding"],
    "neutral": [],
}


def parse_docx_transcript(docx_path: Path) -> dict[str, Any]:
    """
    Parse a call center .docx file.
    
    Returns dict with:
        - summary: str
        - conversation_type: str
        - key_points: str
        - customer_sentiment: str
        - transcript_lines: list[dict] with timestamp, speaker, text
        - full_transcript: str
    """
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    
    result = {
        "summary": "",
        "conversation_type": "",
        "key_points": "",
        "customer_sentiment": "",
        "transcript_lines": [],
        "full_transcript": "",
        "source_file": str(docx_path),
    }
    
    # Parse metadata sections
    i = 0
    while i < len(paragraphs):
        text = paragraphs[i].lower()
        if text == "summary":
            i += 1
        elif text == "conversation type":
            if i + 1 < len(paragraphs):
                result["conversation_type"] = paragraphs[i + 1]
            i += 2
        elif text == "key points":
            if i + 1 < len(paragraphs):
                result["key_points"] = paragraphs[i + 1]
            i += 2
        elif text == "customer sentiment":
            if i + 1 < len(paragraphs):
                result["customer_sentiment"] = paragraphs[i + 1]
            i += 2
        elif "transcription" in text.lower():
            # Start of transcript section
            i += 1
            break
        else:
            i += 1
    
    # Parse transcript lines (timestamp + text pairs)
    timestamp_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")
    transcript_lines = []
    full_transcript_parts = []
    
    while i < len(paragraphs):
        if timestamp_pattern.match(paragraphs[i]):
            timestamp = paragraphs[i]
            if i + 1 < len(paragraphs) and not timestamp_pattern.match(paragraphs[i + 1]):
                text = paragraphs[i + 1]
                transcript_lines.append({"timestamp": timestamp, "text": text})
                full_transcript_parts.append(text)
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    result["transcript_lines"] = transcript_lines
    result["full_transcript"] = " ".join(full_transcript_parts)
    
    return result


def detect_intent(text: str) -> str:
    """Detect intent from text using keyword matching."""
    text_lower = text.lower()
    for intent, keywords in INTENT_PATTERNS.items():
        if intent == "general":
            continue
        if any(kw in text_lower for kw in keywords):
            return intent
    return "general"


def extract_entities(text: str) -> list[dict]:
    """Extract financial entities from text using regex."""
    entities = []
    for entity_type, pattern in ENTITY_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "label": entity_type,
                "start": match.start(),
                "end": match.end(),
            })
    return entities


def detect_emotion(text: str) -> str:
    """Detect emotion from text."""
    text_lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if emotion == "neutral":
            continue
        if any(kw in text_lower for kw in keywords):
            return emotion
    return "neutral"


def detect_risk(text: str, intent: str) -> str:
    """Determine risk level based on content and intent."""
    high_risk_words = ["fraud", "stolen", "unauthorized", "scam", "legal", "lawsuit", "lawyer"]
    medium_risk_words = ["dispute", "complaint", "unacceptable", "escalate", "supervisor"]
    
    text_lower = text.lower()
    
    if any(w in text_lower for w in high_risk_words):
        return "high"
    if any(w in text_lower for w in medium_risk_words) or intent in ["fraud_report", "dispute"]:
        return "medium"
    if intent in ["complaint"]:
        return "medium"
    return "low"


def transcript_to_training_examples(
    parsed: dict[str, Any],
    chunk_size: int = 3,
) -> list[FinanceTrainExample]:
    """
    Convert parsed transcript to training examples.
    
    Groups transcript lines into chunks for more context.
    """
    examples = []
    lines = parsed["transcript_lines"]
    
    # Create examples from chunks of dialog
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        text = " ".join(line["text"] for line in chunk)
        
        if len(text) < 20:  # Skip very short chunks
            continue
        
        intent = detect_intent(text)
        entities = extract_entities(text)
        emotion = detect_emotion(text)
        risk = detect_risk(text, intent)
        
        # Detect obligations
        obligations = []
        obligation_patterns = [
            (r"(?:will|going to|promise to)\s+([^.!?]+)", "follow_up"),
            (r"(?:refund|reimburse)\s+([^.!?]+)", "refund"),
            (r"(?:payment|pay)\s+(?:by|on|within)\s+([^.!?]+)", "payment_promise"),
            (r"(?:send|email|mail)\s+(?:you|the)\s+([^.!?]+)", "document_send"),
        ]
        for pattern, obl_type in obligation_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                obligations.append({
                    "text": m.group(),
                    "type": obl_type,
                })
        
        examples.append(FinanceTrainExample(
            text=text,
            intent=intent,
            entities=entities,
            obligations=obligations,
            emotion=emotion,
            risk_level=risk,
            id=f"{Path(parsed['source_file']).stem}_{i}",
        ))
    
    # Also create a full-transcript example
    if parsed["full_transcript"]:
        full_text = parsed["full_transcript"][:1500]  # Truncate for LLM context
        examples.append(FinanceTrainExample(
            text=full_text,
            intent=detect_intent(full_text),
            entities=extract_entities(full_text),
            obligations=[],
            emotion=detect_emotion(parsed.get("customer_sentiment", "")),
            risk_level=detect_risk(full_text, detect_intent(full_text)),
            id=f"{Path(parsed['source_file']).stem}_full",
        ))
    
    return examples


def to_instruction_format(example: FinanceTrainExample) -> dict[str, str]:
    """Convert training example to instruction format for LLM fine-tuning."""
    output_parts = []
    
    if example.intent:
        output_parts.append(f"Intent: {example.intent}")
    if example.entities:
        ents = ", ".join(f"{e['text']}({e.get('label', '')})" for e in example.entities)
        output_parts.append(f"Entities: {ents}")
    if example.obligations:
        obls = ", ".join(f"{o['text']}[{o.get('type', '')}]" for o in example.obligations)
        output_parts.append(f"Obligations: {obls}")
    if example.emotion:
        output_parts.append(f"Emotion: {example.emotion}")
    if example.risk_level and example.risk_level != "none":
        output_parts.append(f"Risk: {example.risk_level}")
    
    return {
        "instruction": "Analyze this financial call transcript segment. Extract the customer intent, financial entities, obligations/promises, emotional tone, and risk level.",
        "input": example.text,
        "output": "; ".join(output_parts) if output_parts else "No significant findings.",
    }


def process_callcenter_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Process all call center .docx files and generate training datasets.
    
    Returns paths to created files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_examples: list[FinanceTrainExample] = []
    all_transcripts: list[dict] = []
    
    # Find all docx files
    docx_files = list(input_dir.rglob("*.docx"))
    logger.info("Found %d .docx files", len(docx_files))
    
    for docx_path in docx_files:
        try:
            logger.info("Processing: %s", docx_path.name)
            parsed = parse_docx_transcript(docx_path)
            
            # Store full transcript for RAG
            all_transcripts.append({
                "id": docx_path.stem,
                "text": parsed["full_transcript"],
                "summary": parsed.get("key_points", ""),
                "conversation_type": parsed.get("conversation_type", ""),
                "customer_sentiment": parsed.get("customer_sentiment", ""),
            })
            
            # Generate training examples
            examples = transcript_to_training_examples(parsed)
            all_examples.extend(examples)
            logger.info("  Generated %d training examples", len(examples))
            
        except Exception as e:
            logger.warning("Failed to process %s: %s", docx_path, e)
    
    paths = {}
    
    # Save training examples in JSONL format
    if all_examples:
        train_path = output_dir / "callcenter_train.jsonl"
        save_dataset_jsonl(train_path, [ex.model_dump(exclude_none=True) for ex in all_examples])
        paths["train"] = train_path
        logger.info("Saved %d training examples to %s", len(all_examples), train_path)
        
        # Save instruction-tuning format
        instruction_examples = [to_instruction_format(ex) for ex in all_examples]
        instr_path = output_dir / "callcenter_instruction.jsonl"
        save_dataset_jsonl(instr_path, instruction_examples)
        paths["instruction"] = instr_path
        logger.info("Saved instruction format to %s", instr_path)
    
    # Save transcripts for RAG
    if all_transcripts:
        rag_path = output_dir / "callcenter_rag_corpus.jsonl"
        save_dataset_jsonl(rag_path, all_transcripts)
        paths["rag_corpus"] = rag_path
        logger.info("Saved %d transcripts for RAG to %s", len(all_transcripts), rag_path)
    
    return paths


if __name__ == "__main__":
    import argparse
    
    _root = Path(__file__).resolve().parent.parent.parent.parent
    default_input = _root / "data" / "audio" / "Call center data samples"
    default_output = _root / "data" / "finance" / "callcenter"
    
    parser = argparse.ArgumentParser(description="Process call center dataset for training")
    parser.add_argument("--input", default=str(default_input), help="Input directory with .docx files")
    parser.add_argument("--output", default=str(default_output), help="Output directory for datasets")
    args = parser.parse_args()
    
    paths = process_callcenter_dataset(args.input, args.output)
    print("Generated files:")
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))

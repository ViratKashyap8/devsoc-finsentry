"""
Dataset format for Finance Intelligence training.

JSONL format: one JSON object per line. Supports intent, NER, obligation, and
combined instruction-tuning formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel


class IntentExample(BaseModel):
    """Single example for intent classification."""

    text: str
    intent: str
    id: str | None = None


class NERExample(BaseModel):
    """Single example for named entity extraction (span-based)."""

    text: str
    entities: list[dict[str, Any]]  # [{"text": "...", "label": "AMOUNT", "start": 0, "end": 5}]
    id: str | None = None


class ObligationExample(BaseModel):
    """Single example for obligation/promise detection."""

    text: str
    obligations: list[dict[str, Any]]  # [{"text": "...", "type": "payment_promise", "amount": "500"}]
    id: str | None = None


class InstructionExample(BaseModel):
    """Instruction-tuning format for LLM fine-tuning."""

    instruction: str
    input: str
    output: str
    id: str | None = None


class FinanceTrainExample(BaseModel):
    """Combined training example (multi-task)."""

    text: str
    intent: str | None = None
    entities: list[dict[str, Any]] = []
    obligations: list[dict[str, Any]] = []
    emotion: str | None = None
    risk_level: str | None = None
    regulatory_phrases: list[dict[str, Any]] = []
    id: str | None = None


def load_dataset_jsonl(
    path: str | Path, cls: type[BaseModel] = FinanceTrainExample
) -> list[BaseModel]:
    """Load JSONL dataset into list of Pydantic models."""
    path = Path(path)
    if not path.exists():
        return []
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(cls.model_validate(data))
    return examples


def save_dataset_jsonl(
    path: str | Path, examples: list[BaseModel] | list[dict[str, Any]]
) -> None:
    """Save examples to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            if isinstance(ex, BaseModel):
                obj = ex.model_dump(exclude_none=True)
            else:
                obj = ex
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_dataset_jsonl(
    path: str | Path, cls: type[BaseModel] = FinanceTrainExample
) -> Iterator[BaseModel]:
    """Stream dataset without loading full file."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield cls.model_validate(json.loads(line))

"""
Dataset utilities for LoRA fine-tuning.

Converts JSONL to instruction format for HuggingFace.
"""

from __future__ import annotations

from typing import Any

from ..dataset.format import load_dataset_jsonl, FinanceTrainExample


def format_instruction_example(ex: FinanceTrainExample) -> dict[str, str]:
    """Convert combined example to instruction format."""
    parts = []
    if ex.intent:
        parts.append(f"Intent: {ex.intent}")
    if ex.entities:
        ents = ", ".join(f"{e['text']}({e.get('label','')})" for e in ex.entities)
        parts.append(f"Entities: {ents}")
    if ex.obligations:
        obls = ", ".join(f"{o['text']}[{o.get('type','')}]" for o in ex.obligations)
        parts.append(f"Obligations: {obls}")
    if ex.emotion:
        parts.append(f"Emotion: {ex.emotion}")
    if ex.risk_level and ex.risk_level != "none":
        parts.append(f"Risk: {ex.risk_level}")
    output = "; ".join(parts) if parts else "None"
    return {
        "instruction": "Analyze this financial call utterance. Extract intent, entities, obligations, emotion, and risk.",
        "input": ex.text,
        "output": output,
    }


def build_hf_dataset(
    jsonl_path: str,
    tokenizer: Any = None,
    max_length: int = 512,
    split: float = 0.9,
):
    """
    Build HuggingFace Dataset for training.

    Returns (train_dataset, eval_dataset) with instruction/input/output columns.
    SFTTrainer will handle tokenization via formatting_func.
    """
    from datasets import Dataset

    examples = load_dataset_jsonl(jsonl_path, cls=FinanceTrainExample)
    formatted = [format_instruction_example(ex) for ex in examples]
    ds = Dataset.from_list(formatted)
    ds = ds.train_test_split(test_size=1 - split, seed=42)
    return ds["train"], ds["test"]

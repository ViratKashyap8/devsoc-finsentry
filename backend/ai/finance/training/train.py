"""
LoRA fine-tuning script for finance extraction model.

Usage:
  python -m ai.finance.training.train \
    --data data/finance/synthetic/combined_train.jsonl \
    --output data/finance/models/smollm-lora \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --epochs 2 --batch 2

  # With new-call data (concatenates base + extra, then train/val split):
  python -m ai.finance.training.train \
    --data data/finance/synthetic/combined_train.jsonl \
    --extra-dataset data/new_calls/finance_train.jsonl \
    --output models/finance_v2 --epochs 2

Optimized for 48hr hackathon: 2 epochs, small batch, ~30min on T4.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_extra_dataset(extra_path: str | Path, split: float = 0.9):
    """Load extra JSONL via datasets.load_dataset('json'), convert to instruction format, return (train_ds, eval_ds)."""
    from datasets import load_dataset

    from ..dataset.format import FinanceTrainExample
    from .dataset import format_instruction_example

    path = Path(extra_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Extra dataset not found: {path}")
    ds = load_dataset("json", data_files=str(path), split="train")
    # Convert to instruction/input/output format
    formatted = []
    for i in range(len(ds)):
        row = ds[i]
        ex = FinanceTrainExample.model_validate(
            {
                "text": row["text"],
                "intent": row.get("intent"),
                "risk_level": row.get("risk_level"),
                "entities": row.get("entities", []),
                "obligations": row.get("obligations", []),
                "regulatory_phrases": row.get("regulatory_phrases", []),
                "emotion": row.get("emotion"),
                "id": row.get("id"),
            }
        )
        formatted.append(format_instruction_example(ex))
    from datasets import Dataset

    extra_ds = Dataset.from_list(formatted)
    n = len(extra_ds)
    if n < 2:
        # Single sample (or empty): use all for both train and eval so split doesn't fail
        return extra_ds, extra_ds
    split_ds = extra_ds.train_test_split(test_size=1 - split, seed=42)
    return split_ds["train"], split_ds["test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to base combined_train.jsonl (required unless only using --extra-dataset)")
    parser.add_argument(
        "--extra-dataset",
        default=None,
        help="Optional path to extra JSONL (e.g. data/new_calls/finance_train.jsonl); concatenated with base then shuffled and split",
    )
    parser.add_argument("--output", default="models/finance_v2", help="Output dir for LoRA weights")
    parser.add_argument("--output-dir", dest="output", help="Alias for --output")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--train-split", type=float, default=0.9, help="Fraction for train (rest is val)")
    args = parser.parse_args()

    # Validate paths before loading model (fail fast with clear errors)
    if args.extra_dataset:
        extra_path = Path(args.extra_dataset).resolve()
        if not extra_path.exists():
            logger.error(
                "Extra dataset not found: %s. Create it with: make transcribe then make prepare-finance-dataset (see README).",
                extra_path,
            )
            raise SystemExit(1)
    if args.data:
        data_path = Path(args.data).resolve()
        if not data_path.exists():
            logger.error("Base dataset not found: %s", data_path)
            raise SystemExit(1)
    if not args.data and not args.extra_dataset:
        logger.error("Provide at least one of --data or --extra-dataset")
        raise SystemExit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset, concatenate_datasets
    except ImportError as e:
        logger.error(
            "Install: pip install transformers peft trl datasets torch"
        )
        raise SystemExit(1) from e

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto" if __import__("torch").cuda.is_available() else None,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data: base and/or extra
    from .dataset import build_hf_dataset

    if args.data:
        train_ds, eval_ds = build_hf_dataset(args.data, split=args.train_split)
        if args.extra_dataset:
            extra_train, extra_eval = _load_extra_dataset(args.extra_dataset, split=args.train_split)
            base_all = concatenate_datasets([train_ds, eval_ds])
            extra_all = concatenate_datasets([extra_train, extra_eval])
            combined = concatenate_datasets([base_all, extra_all])
            combined = combined.shuffle(seed=42)
            split = combined.train_test_split(test_size=1 - args.train_split, seed=42)
            train_ds, eval_ds = split["train"], split["test"]
            logger.info(
                "Combined base + extra dataset: train=%d eval=%d",
                len(train_ds),
                len(eval_ds),
            )
    elif args.extra_dataset:
        train_ds, eval_ds = _load_extra_dataset(args.extra_dataset, split=args.train_split)
        logger.info("Using extra dataset only: train=%d eval=%d", len(train_ds), len(eval_ds))
    else:
        raise SystemExit("Provide --data and/or --extra-dataset")
    print(f"Dataset sizes: train={len(train_ds)} eval={len(eval_ds)}", flush=True)

    def format_prompt(example):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )

    # SFTConfig for newer trl versions
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        fp16=__import__("torch").cuda.is_available(),
        max_length=args.max_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=format_prompt,
        processing_class=tokenizer,
    )

    trainer.train()
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Saved LoRA weights to %s", out_path)

    # Print epoch metrics from training state
    if hasattr(trainer, "state") and trainer.state.log_history:
        print("Epoch metrics:", flush=True)
        for entry in trainer.state.log_history:
            if "eval_loss" in entry:
                print(f"  eval_loss={entry.get('eval_loss'):.4f}", flush=True)
            if "loss" in entry and "eval_loss" not in entry:
                print(f"  loss={entry.get('loss'):.4f} (step {entry.get('current_step', '')})", flush=True)


if __name__ == "__main__":
    main()

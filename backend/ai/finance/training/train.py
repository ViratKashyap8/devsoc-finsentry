"""
LoRA fine-tuning script for finance extraction model.

Usage:
  python -m ai.finance.training.train \
    --data data/finance/synthetic/combined_train.jsonl \
    --output data/finance/models/smollm-lora \
    --model HuggingFaceH4/smollm2-360m-instruct \
    --epochs 2 --batch 2

Optimized for 48hr hackathon: 2 epochs, small batch, ~30min on T4.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to combined_train.jsonl")
    parser.add_argument("--output", required=True, help="Output dir for LoRA weights")
    parser.add_argument("--model", default="HuggingFaceH4/smollm2-360m-instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
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

    # Load data
    from .dataset import build_hf_dataset
    train_ds, eval_ds = build_hf_dataset(args.data, split=0.9)

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
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Saved LoRA weights to %s", args.output)


if __name__ == "__main__":
    main()

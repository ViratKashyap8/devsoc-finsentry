"""
Evaluation script for Finance Intelligence models.

Computes metrics on held-out test set:
- Intent: accuracy, macro F1
- Entity: span-level F1 (strict)
- Obligation: precision, recall, F1
- Risk: accuracy (if labels available)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def eval_intent(predicted: list[str], gold: list[str]) -> dict:
    """Intent classification metrics."""
    correct = sum(1 for p, g in zip(predicted, gold) if p == g)
    acc = correct / len(gold) if gold else 0
    # Per-class P/R/F1
    gold_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    tp = defaultdict(int)
    for g, p in zip(gold, predicted):
        gold_counts[g] += 1
        pred_counts[p] += 1
        if g == p:
            tp[g] += 1
    classes = set(gold_counts) | set(pred_counts)
    f1s = []
    for c in classes:
        p = tp[c] / pred_counts[c] if pred_counts[c] else 0
        r = tp[c] / gold_counts[c] if gold_counts[c] else 0
        f1s.append(_f1(p, r))
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    return {"accuracy": acc, "macro_f1": macro_f1}


def eval_obligation(predicted: list[list[dict]], gold: list[list[dict]]) -> dict:
    """Obligation detection: overlap-based P/R/F1."""
    tp = fp = fn = 0
    for pred_obls, gold_obls in zip(predicted, gold):
        pred_texts = {o.get("text", "").lower()[:50] for o in pred_obls}
        gold_texts = {o.get("text", "").lower()[:50] for o in gold_obls}
        tp += len(pred_texts & gold_texts)
        fp += len(pred_texts - gold_texts)
        fn += len(gold_texts - pred_texts)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {"precision": p, "recall": r, "f1": _f1(p, r)}


def run_eval(
    test_path: str,
    output_path: str | None = None,
    pipeline=None,
) -> dict:
    """
    Run evaluation on test JSONL.

    Expects same format as training (FinanceTrainExample).
    """
    from ..dataset.format import load_dataset_jsonl, FinanceTrainExample
    from ..pipeline import FinancePipeline

    examples = load_dataset_jsonl(test_path, cls=FinanceTrainExample)
    if not examples:
        logger.warning("No test examples found at %s", test_path)
        return {}

    pipeline = pipeline or FinancePipeline(use_llm_extraction=False)
    predicted_intents = []
    predicted_obligations = []
    gold_intents = []
    gold_obligations = []

    for ex in examples:
        seg = pipeline.analyze_segment(ex.text)
        predicted_intents.append(seg.intent.value if seg.intent else "general")
        predicted_obligations.append(
            [{"text": o.text, "type": o.obligation_type.value} for o in seg.obligations]
        )
        gold_intents.append(ex.intent or "general")
        gold_obligations.append(ex.obligations)

    results = {
        "intent": eval_intent(predicted_intents, gold_intents),
        "obligation": eval_obligation(predicted_obligations, gold_obligations),
        "n_examples": len(examples),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results to %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test JSONL")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()
    results = run_eval(args.test, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

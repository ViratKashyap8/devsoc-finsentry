"""
Synthetic data generator for Finance Intelligence training.

Generates plausible financial call utterances with labels for hackathon MVP.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .format import (
    FinanceTrainExample,
    IntentExample,
    ObligationExample,
    save_dataset_jsonl,
)

# Templates: (template, intent, entities_expected, obligation_type)
INTENT_TEMPLATES = [
    ("I'm calling to dispute a charge of $450 on my account.", "dispute", ["AMOUNT", "ACCOUNT"], None),
    ("When will my refund of $199 be processed?", "inquiry", ["AMOUNT"], "refund"),
    ("I need to set up a payment plan for the remaining balance.", "payment_arrangement", [], "payment_promise"),
    ("Can you waive the late fee of $35?", "inquiry", ["AMOUNT", "FEE"], "fee_waiver"),
    ("I want to close my account number ending in 4521.", "closure_request", ["ACCOUNT"], None),
    ("Someone used my card without my permission. I need to report fraud.", "fraud_report", [], "escalation"),
    ("What's my current balance and next due date?", "account_info", ["ACCOUNT", "DATE"], None),
    ("This is the third time I've called. Nobody has helped me!", "complaint", [], None),
    ("I promise to pay $200 by Friday next week.", "payment_arrangement", ["AMOUNT", "DATE"], "payment_promise"),
    ("Please send me the transaction history for the last 90 days.", "inquiry", ["DATE"], "document_send"),
    ("I'm extremely frustrated with your service. I'm considering filing a complaint.", "complaint", [], None),
    ("Per your request, I'm confirming this call may be recorded for quality assurance.", "general", [], None),
    ("When can I expect a callback from your supervisor?", "inquiry", [], "follow_up"),
]

# More variations for diversity: (original, replacement) pairs
VARIATIONS = [
    ("account", "card"),
    ("refund", "reimbursement"),
    ("payment", "installment"),
    ("fee", "charge"),
]

EMOTION_MAP = {
    "complaint": "frustrated",
    "dispute": "anxious",
    "inquiry": "neutral",
    "payment_arrangement": "calm",
    "fraud_report": "anxious",
    "closure_request": "neutral",
    "account_info": "neutral",
    "general": "neutral",
}

RISK_MAP = {
    "fraud_report": "high",
    "complaint": "medium",
    "dispute": "medium",
    "closure_request": "low",
    "payment_arrangement": "low",
    "inquiry": "none",
    "account_info": "none",
    "general": "none",
}


def _random_amount() -> str:
    return str(random.choice([35, 50, 99, 199, 200, 450, 500, 1200, 2500]))


def _random_account_suffix() -> str:
    return str(random.randint(1000, 9999))


def _apply_variation(text: str) -> str:
    for orig, repl in random.sample(VARIATIONS, min(2, len(VARIATIONS))):
        if orig in text.lower() and random.random() < 0.5:
            text = text.replace(orig, repl, 1)
    return text


def generate_intent_example() -> IntentExample:
    """Generate single intent classification example."""
    template, intent, _, _ = random.choice(INTENT_TEMPLATES)
    text = template
    text = text.replace("$450", f"${_random_amount()}")
    text = text.replace("4521", _random_account_suffix())
    text = text.replace("$199", f"${_random_amount()}")
    text = text.replace("$35", f"${_random_amount()}")
    text = _apply_variation(text)
    return IntentExample(text=text, intent=intent)


def generate_obligation_example() -> ObligationExample:
    """Generate single obligation detection example."""
    obligation_texts = [
        ("I'll pay $500 by next Friday.", "payment_promise", "500", "next Friday"),
        ("We'll send you the document within 3 business days.", "document_send", None, "3 business days"),
        ("A supervisor will call you back within 24 hours.", "follow_up", None, "24 hours"),
        ("Your refund of $199 will be processed in 5-7 days.", "refund", "199", "5-7 days"),
        ("I can waive the $35 late fee this one time.", "fee_waiver", "35", None),
        ("I'm escalating this to our fraud department.", "escalation", None, None),
    ]
    text, obl_type, amount, due = random.choice(obligation_texts)
    obligations = [{"text": text, "type": obl_type, "amount": amount, "due_date": due}]
    return ObligationExample(text=text, obligations=obligations)


def generate_combined_example() -> FinanceTrainExample:
    """Generate combined multi-task example."""
    intent_ex = generate_intent_example()
    template_entry = next((t for t in INTENT_TEMPLATES if t[1] == intent_ex.intent), INTENT_TEMPLATES[0])
    _, intent, _, obl_type = template_entry

    obligations = []
    if obl_type and random.random() < 0.5:
        obl_ex = generate_obligation_example()
        obligations = obl_ex.obligations

    entities = []
    if "AMOUNT" in template_entry[2]:
        import re
        amounts = re.findall(r"\$(\d+)", intent_ex.text)
        for i, amt in enumerate(amounts):
            entities.append({"text": f"${amt}", "label": "AMOUNT", "start": 0, "end": 0})
    if "ACCOUNT" in template_entry[2]:
        entities.append({"text": "account", "label": "ACCOUNT", "start": 0, "end": 0})

    return FinanceTrainExample(
        text=intent_ex.text,
        intent=intent,
        entities=entities,
        obligations=obligations,
        emotion=EMOTION_MAP.get(intent, "neutral"),
        risk_level=RISK_MAP.get(intent, "none"),
    )


def generate_synthetic_dataset(
    output_dir: str | Path,
    n_intent: int = 500,
    n_obligation: int = 300,
    n_combined: int = 1000,
) -> dict[str, Path]:
    """
    Generate synthetic datasets and save to output_dir.

    Returns paths to created files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Intent
    intent_examples = [generate_intent_example().model_dump(exclude_none=True) for _ in range(n_intent)]
    intent_path = output_dir / "intent_train.jsonl"
    save_dataset_jsonl(intent_path, intent_examples)
    paths["intent"] = intent_path

    # Obligation
    obl_examples = [generate_obligation_example().model_dump(exclude_none=True) for _ in range(n_obligation)]
    obl_path = output_dir / "obligation_train.jsonl"
    save_dataset_jsonl(obl_path, obl_examples)
    paths["obligation"] = obl_path

    # Combined (for instruction tuning)
    combined = [generate_combined_example().model_dump(exclude_none=True) for _ in range(n_combined)]
    combined_path = output_dir / "combined_train.jsonl"
    save_dataset_jsonl(combined_path, combined)
    paths["combined"] = combined_path

    return paths


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    # Default output relative to project root
    _root = Path(__file__).resolve().parent.parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(_root / "data" / "finance" / "synthetic"))
    parser.add_argument("--n-intent", type=int, default=500)
    parser.add_argument("--n-obligation", type=int, default=300)
    parser.add_argument("--n-combined", type=int, default=1000)
    args = parser.parse_args()
    paths = generate_synthetic_dataset(args.output, args.n_intent, args.n_obligation, args.n_combined)
    print("Generated:", json.dumps({k: str(v) for k, v in paths.items()}, indent=2))

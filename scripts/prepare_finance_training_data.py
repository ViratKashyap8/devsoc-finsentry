#!/usr/bin/env python3
"""
Prepare supervised training data for AI-2 from transcript JSONs + human labels.
Reads transcripts (from transcribe_new_calls.py) and a labels file (CSV or JSONL),
validates every transcript has a label, converts to FinanceTrainExample JSONL.
Run from repo root: python scripts/prepare_finance_training_data.py [options]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
DEFAULT_TRANSCRIPTS = BACKEND / "data" / "new_calls" / "transcripts"
DEFAULT_LABELS = BACKEND / "data" / "new_calls" / "labels.csv"
DEFAULT_OUTPUT = BACKEND / "data" / "new_calls" / "finance_train.jsonl"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Expected label columns (CSV/JSONL); intent and risk_level required for training
LABEL_CALL_ID = "call_id"
LABEL_INTENT = "intent"
LABEL_RISK = "risk_level"
OPTIONAL_LABEL_COLS = ["amount", "currency", "merchant", "counterparty", "payment_method", "transaction_date"]
# Allow counterparty or merchant
LABEL_MERCHANT_ALIASES = ("merchant", "counterparty")


def load_transcript_jsons(transcripts_dir: Path) -> dict[str, dict]:
    """Load all .json files under transcripts_dir. Return {call_id: data}."""
    if not transcripts_dir.is_dir():
        return {}
    out = {}
    for p in sorted(transcripts_dir.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            call_id = data.get("call_id")
            if call_id is None:
                # Fallback: use stem as call_id
                call_id = p.stem
                data["call_id"] = call_id
            out[call_id] = data
        except Exception as e:
            logger.warning("Skip %s: %s", p.name, e)
    return out


def load_labels_csv(path: Path) -> list[dict]:
    """Load labels from CSV. Normalize column names to lowercase."""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip().lower(): v.strip() if isinstance(v, str) else v for k, v in row.items()})
    return rows


def load_labels_jsonl(path: Path) -> list[dict]:
    """Load labels from JSONL."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                row = {k.lower(): v for k, v in row.items()}
            rows.append(row)
    return rows


def load_labels(path: Path) -> list[dict]:
    """Load labels from CSV or JSONL based on extension."""
    path = Path(path)
    if not path.exists():
        return []
    suf = path.suffix.lower()
    if suf == ".csv":
        return load_labels_csv(path)
    if suf in (".jsonl", ".json"):
        return load_labels_jsonl(path)
    # Try CSV first, then JSONL
    try:
        return load_labels_csv(path)
    except Exception:
        return load_labels_jsonl(path)


def labels_by_call_id(rows: list[dict]) -> dict[str, dict]:
    """Index label rows by call_id. Use first occurrence if duplicate."""
    by_id = {}
    for row in rows:
        cid = row.get(LABEL_CALL_ID) or row.get("call_id")
        if cid is not None and str(cid).strip():
            by_id[str(cid).strip()] = row
    return by_id


def build_train_example(transcript: dict, label: dict) -> dict:
    """Build one FinanceTrainExample-compatible dict from transcript + label."""
    full_transcript = transcript.get("full_transcript") or ""
    call_id = transcript.get("call_id") or label.get(LABEL_CALL_ID, "")
    intent = (label.get(LABEL_INTENT) or "").strip().lower() or None
    risk_level = (label.get(LABEL_RISK) or label.get("risk") or "").strip().lower() or None
    return {
        "text": full_transcript,
        "intent": intent,
        "risk_level": risk_level,
        "entities": [],
        "obligations": [],
        "regulatory_phrases": [],
        "id": call_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare AI-2 training JSONL from transcript JSONs + labels file.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS,
        help="Directory containing transcript .json files (default: %(default)s)",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to labels CSV or JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()

    transcripts_dir = args.transcripts_dir.resolve()
    labels_path = args.labels_file.resolve()
    output_path = args.output_file.resolve()

    transcripts = load_transcript_jsons(transcripts_dir)
    if not transcripts:
        logger.error("No transcript JSONs found under %s", transcripts_dir)
        return 1

    label_rows = load_labels(labels_path)
    if not label_rows:
        logger.error("No labels loaded from %s", labels_path)
        return 1

    labels_by_id = labels_by_call_id(label_rows)
    missing = [tid for tid in transcripts if tid not in labels_by_id]
    if missing:
        logger.error("Transcripts missing labels (call_id): %s", missing)
        return 1

    # Build training examples in transcript order
    from ai.finance.dataset.format import FinanceTrainExample, save_dataset_jsonl

    examples = []
    for call_id, transcript in sorted(transcripts.items()):
        if call_id not in labels_by_id:
            continue
        ex_dict = build_train_example(transcript, labels_by_id[call_id])
        examples.append(FinanceTrainExample.model_validate(ex_dict))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_jsonl(output_path, examples)

    # Stats
    intents = Counter(ex.intent for ex in examples if ex.intent)
    risks = Counter(ex.risk_level for ex in examples if ex.risk_level)
    print(flush=True)
    print("Stats:", flush=True)
    print(f"  total rows: {len(examples)}", flush=True)
    print("  intents:", flush=True)
    for k, v in intents.most_common():
        print(f"    {k}: {v}", flush=True)
    print("  risk_level:", flush=True)
    for k, v in risks.most_common():
        print(f"    {k}: {v}", flush=True)
    print(f"  output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

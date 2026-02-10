#!/usr/bin/env python3
"""
Run AI-2 finance pipeline on a sample transcript.
Run from repo root: python scripts/test_finance_pipeline.py  (or: make test-finance)
Uses use_llm_extraction=False so no OpenAI/API keys are required; backend/.env is not loaded.
Requires backend deps installed. Script injects backend into sys.path when run from repo root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Sample transcript; no API keys or .env required when use_llm_extraction=False
SAMPLE_TRANSCRIPT = """
Customer: I'm calling about a chargeback on my card. I didn't make that purchase.
Agent: I understand. I'll open a dispute and send you a form within 5 business days.
Customer: Can you waive the late fee? I've never been late before.
Agent: I can waive it this once. Your refund will be processed in 7 to 10 days.
"""


def main() -> int:
    print("Running AI-2 finance pipeline on sample transcript (use_llm_extraction=False, no API keys)...", flush=True)
    try:
        from ai.finance.pipeline import run_finance_analysis
    except ModuleNotFoundError as e:
        print(
            f"Missing dependency: {e}. Activate backend/.venv or run from repo root: make test-finance",
            file=sys.stderr,
        )
        return 1
    try:
        result = run_finance_analysis(
            SAMPLE_TRANSCRIPT.strip(),
            segments=None,
            call_id="demo-call",
            use_llm_extraction=False,
        )
        print("\n--- Finance pipeline output (summary) ---")
        print(f"  call_id: {result.call_id}")
        if result.call_metrics:
            m = result.call_metrics
            print(f"  dominant_intent: {getattr(m.dominant_intent, 'value', m.dominant_intent)}")
            print(f"  overall_risk_level: {getattr(m.overall_risk_level, 'value', m.overall_risk_level)}")
            print(f"  risk_score: {m.risk_score}")
            print(f"  total_obligations: {m.total_obligations}")
        print(f"  segments: {len(result.segments)}")
        print("\n--- Full JSON ---")
        try:
            out_dict = result.model_dump(mode="json")
        except TypeError:
            out_dict = result.model_dump()
        print(json.dumps(out_dict, indent=2, default=str), flush=True)
        return 0
    except Exception as e:
        print(f"Finance pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

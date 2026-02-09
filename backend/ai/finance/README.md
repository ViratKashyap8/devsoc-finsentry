# Finance Intelligence & RAG Module

Local-only NLP for fintech call analytics. No OpenAI/Anthropic/Gemini.

## Quick Start (48hr Hackathon)

### 1. Install Dependencies

```bash
cd backend
uv sync
# Optional for LoRA training:
uv pip install peft trl
```

### 2. Generate Synthetic Data

```bash
python -m ai.finance.dataset.synthetic --output data/finance/synthetic --n-combined 1000
```

### 3. Run Finance Analysis (API)

```bash
uv run uvicorn main:app --reload
# POST http://localhost:8000/api/finance/analyze
```

```json
{
  "full_transcript": "I'm calling to dispute a charge of $450. When will my refund be processed?",
  "call_id": "call-001",
  "use_llm_extraction": true
}
```

### 4. RAG Q&A

```bash
# POST http://localhost:8000/api/finance/rag/query
{"query": "How long do refunds take?", "top_k": 3}
```

### 5. LoRA Fine-Tuning (Optional)

```bash
python -m ai.finance.training.train \
  --data data/finance/synthetic/combined_train.jsonl \
  --output data/finance/models/smollm-lora \
  --epochs 2 --batch 2
```

### 6. Evaluate

```bash
python -m ai.finance.eval.evaluate --test data/finance/synthetic/combined_train.jsonl --output results.json
```

## Architecture

| Component       | Model                          | Notes                    |
|----------------|---------------------------------|--------------------------|
| Intent/Emotion | all-MiniLM-L6-v2 (zero-shot)   | 80MB, fast, CPU          |
| NER/Obligation | SmolLM2-360M-Instruct          | ~2GB VRAM, JSON output   |
| Risk           | Rule-based + intent            | No model load            |
| Regulatory     | Rules + optional LLM           | Fast fallback            |
| RAG Embeddings | all-MiniLM-L6-v2               | Same as classifier       |
| RAG Retriever  | FAISS (flat)                   | Exact search             |

## Dataset Format (JSONL)

Each line is a JSON object:

```json
{
  "text": "I'll pay $500 by next Friday.",
  "intent": "payment_arrangement",
  "entities": [{"text": "$500", "label": "AMOUNT"}],
  "obligations": [{"text": "I'll pay $500 by next Friday", "type": "payment_promise", "amount": "500", "due_date": "next Friday"}],
  "emotion": "calm",
  "risk_level": "low"
}
```

## Integration with Audio Pipeline

```python
from ai.audio.pipeline import run_pipeline
from ai.finance import run_finance_analysis

audio_out, _ = run_pipeline("call.wav")
finance_out = run_finance_analysis(
    audio_out.full_transcript,
    segments=[s.model_dump() for s in audio_out.segments],
    call_id=audio_out.call_id,
)
```

## Warnings

- **SmolLM2-360M** output quality varies; fine-tune with LoRA for production.
- **48hr timeline**: Prioritize rule-based + zero-shot. LoRA is stretch goal.
- **CPU inference**: LLM extraction is slow on CPU (~10s per segment). Use `use_llm_extraction=False` for rule-only fast path.

# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

FinSentry is a fintech call analytics platform that processes phone call recordings into structured financial intelligence. It combines audio transcription with NLP analysis for intent detection, entity extraction, obligation tracking, and risk assessment.

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, uv (package manager)
- **Frontend**: React 19, TypeScript, Vite, pnpm
- **AI/ML**: sentence-transformers (zero-shot classification), SmolLM2-360M (LLM extraction), FAISS (RAG retrieval), faster-whisper (STT)
- **Containerization**: Docker Compose

## Development Commands

### Backend
```bash
cd backend
uv sync                              # Install dependencies
uv run uvicorn main:app --reload     # Start dev server (port 8000)
uv run python -m pytest              # Run tests (if present)
```

### Frontend
```bash
cd frontend
pnpm install                         # Install dependencies
pnpm dev                             # Start dev server (port 5173)
pnpm build                           # Build for production
pnpm lint                            # Run ESLint
```

### Docker
```bash
docker-compose up                    # Start both services
```

### AI-specific Commands
```bash
# Generate synthetic training data
python -m ai.finance.dataset.synthetic --output data/finance/synthetic --n-combined 1000

# LoRA fine-tuning (optional, requires peft/trl)
python -m ai.finance.training.train --data data/finance/synthetic/combined_train.jsonl --output data/finance/models/smollm-lora --epochs 2 --batch 2

# Evaluate model
python -m ai.finance.eval.evaluate --test data/finance/synthetic/combined_train.jsonl --output results.json

# Audio pipeline CLI
python ai/audio/pipeline.py path/to/audio.wav --benchmark
```

## Architecture

### Backend Structure

```
backend/
├── main.py                 # FastAPI app entry, mounts routers under /api
├── routers/                # API endpoints
│   ├── general.py          # Health check (/api/health)
│   ├── audio.py            # Audio upload/retrieval (/api/audio)
│   └── finance.py          # Finance analysis & RAG (/api/finance)
├── ai/
│   ├── audio/              # Audio Intelligence module
│   │   └── pipeline.py     # Ingest → Preprocess → DSP → Silence → Chunk → STT
│   └── finance/            # Finance Intelligence module
│       ├── pipeline.py     # Main analysis orchestrator (FinancePipeline)
│       ├── models/         # ML models (classifier, extractor, risk)
│       ├── rag/            # RAG Q&A (retriever, generator, embeddings)
│       ├── dataset/        # Data generation and preprocessing
│       └── training/       # LoRA fine-tuning scripts
├── services/               # Business logic layer
├── repositories/           # Data access layer
├── schemas/                # Pydantic request/response models
└── core/config.py          # Environment config (.env loading)
```

### Key Data Flow

1. **Audio Pipeline**: Raw audio → Resample (16kHz mono) → DSP (band-pass, noise reduction) → Silence trim → Chunk (≤60s) → faster-whisper STT → Timestamped transcript

2. **Finance Pipeline**: Transcript → Batch intent/emotion classification (zero-shot) → Per-segment analysis (entities, obligations, regulatory via LLM or rules) → Call-level risk aggregation

3. **RAG Pipeline**: Query → Embed with MiniLM → FAISS retrieval → Context + SmolLM2 generation

### Model Configuration

Models are configured in `backend/ai/finance/config.py`:
- `all-MiniLM-L6-v2`: Intent, emotion, and RAG embeddings (~80MB)
- `SmolLM2-360M-Instruct`: Entity/obligation extraction (~2GB VRAM)
- `faster-whisper-base`: Speech-to-text (~150MB)

Use `use_llm_extraction=False` in finance API calls for rule-only fast path (CPU-friendly).

## Key Patterns

- **Lazy model loading**: ML models are loaded on first use and cached (see `_model_cache` in classifier.py, `_rag_generator` in finance router)
- **Fallback strategies**: All classifiers have rule-based fallbacks when ML dependencies are unavailable
- **Async with executors**: CPU-bound ML inference runs in `run_in_executor` to avoid blocking FastAPI
- **Pyright ignore comments**: Use `# pyright: ignore[reportImplicitRelativeImport]` for relative imports from `backend/` root

## Environment Variables

Create `backend/.env` with:
```
BACKBOARD_API_KEY=your_key_here
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/audio/upload` - Upload audio file
- `GET /api/audio/{file_id}` - Get audio file details
- `POST /api/finance/analyze` - Analyze transcript (intent, entities, obligations, risk)
- `POST /api/finance/rag/query` - RAG Q&A over finance documents

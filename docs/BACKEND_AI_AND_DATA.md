# backend/ai and backend/data — What’s required

## backend/ai — **Required for correct functioning**

The whole **backend/ai** folder is needed for the app and demo scripts to work.

| Part | Used by | Purpose |
|------|--------|--------|
| **ai/audio/** | API (audio upload/pipeline), `make test-audio`, CLI pipeline | Ingest, preprocess, DSP, silence, chunk, STT (faster-whisper). |
| **ai/finance/** | API (`/api/finance/analyze`, `/api/finance/rag/query`), `make test-finance` | Intent/emotion, extraction, risk, RAG. |

- **Routers** import `ai.finance.pipeline`, `ai.finance.rag`.
- **Demo scripts** use `ai.audio.pipeline` and `ai.finance.pipeline`.
- Do not remove or rename **backend/ai**; the project depends on it.

---

## backend/data — **Largely optional for core functioning**

For “correct functioning” we mean: API runs, `make test-audio` and `make test-finance` work.

### What the code does with `data/`

| Path | Required? | Notes |
|------|-----------|--------|
| **data/raw/** | **No pre-existing content** | Used by **ingest** (audio pipeline). Code does `raw_dir.mkdir(parents=True, exist_ok=True)`, so the directory is created when needed. Nothing in `data/` has to exist beforehand. |
| **data/debug/** | No | Only used when running the audio pipeline with `--save-intermediate`; created on demand. |
| **data/audio/** | No | Optional. Used as **fallback** by `scripts/test_audio_pipeline.py` (it first looks for samples in **backend/ai/audio/**). Also used by optional CLIs: `callcenter.py`, `preprocess_callcenter.py` (training/preprocessing). |
| **data/finance/callcenter/** | No | Optional. Pre-built JSONL and RAG index for **optional** training and `build_index`. The API RAG uses **in-memory** docs by default (see `routers/finance.py`). |
| **data/finance/models/** | No | Optional. LoRA fine-tuned model (e.g. smollm-lora-callcenter). Default extraction uses the HuggingFace model, not this local path. |

### Conclusion on backend/data

- **No part of backend/data is required to exist before running the app or demo scripts.**
- **data/raw** (and optionally **data/debug**) are **created at runtime** when the audio pipeline runs.
- Everything currently in **data/audio/** and **data/finance/** (call center samples, JSONL, rag_index, LoRA checkpoints) is for:
  - Optional training / preprocessing scripts
  - Optional RAG index building
  - Optional LoRA model
  - Fallback sample for `test_audio_pipeline.py` (which already has samples in **ai/audio/**)

So for core functioning, **backend/data is effectively redundant** as stored content. The project only needs the ability to create **data/raw** (and **data/debug** if you use `--save-intermediate`); the existing files in **data/audio** and **data/finance** are not required.

### If you remove or trim backend/data

- You can delete **data/audio** and **data/finance** (or their contents) and the API plus `make test-audio` / `make test-finance` will still work.
- Keep **backend/ai** unchanged.
- Optionally add **data/raw/** and **data/debug/** to `.gitignore` if you don’t want to commit runtime-generated files (the code will recreate the dirs when needed).

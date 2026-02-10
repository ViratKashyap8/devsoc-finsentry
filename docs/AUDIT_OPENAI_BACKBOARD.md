# Audit: OpenAI and Backboard usage

## Summary

- **OpenAI / third-party LLM:** Not used. No `openai` package, no `ChatCompletion`, no embeddings API calls to OpenAI. `OPENAI_API_KEY` was only referenced in config/bootstrap/env/README; it has been removed from required env.
- **Backboard:** `BACKBOARD_API_KEY` and `BACKBOARD_BASE_URL` are required at startup. They are not used for inference, classification, extraction, or embeddings. No Backboard SDK or HTTP client exists in the repo. Outputs are designed for eventual Backboard sync; ledger writes are intended to go exclusively through Backboard when that integration is implemented.
- **Local AI only:** AI-1 uses faster-whisper; AI-2 uses sentence-transformers (MiniLM) and HuggingFace/transformers (SmolLM2-360M). No migration to Backboard AI was performed because Backboard does not provide AI endpoints in this codebase.

---

## Files that referenced OPENAI_API_KEY (updated)

| File | Change |
|------|--------|
| `backend/core/config.py` | Removed `OPENAI_API_KEY` from env load and from `REQUIRED_ENV_KEYS`. Only Backboard credentials required. |
| `scripts/bootstrap.py` | Removed `OPENAI_API_KEY` from `REQUIRED_ENV`. |
| `backend/.env.example` | Removed `OPENAI_API_KEY`. |
| `README.md` | Env section and checklist now require only Backboard keys; added "AI and third-party providers" section. |

---

## Files and functions: OpenAI / LLM / embeddings

No code imports or calls OpenAI. References were configuration-only (now removed).

| File | What it does (no OpenAI) |
|------|--------------------------|
| `backend/ai/finance/rag/embeddings.py` | `embed_texts()` — uses **sentence-transformers** (all-MiniLM-L6-v2). |
| `backend/ai/finance/rag/retriever.py` | Uses `embed_texts()` for FAISS indexing; no external API. |
| `backend/ai/finance/models/classifier.py` | `classify_intent()`, `classify_emotion()` — **sentence-transformers** zero-shot. |
| `backend/ai/finance/models/extractor.py` | `extract_entities()`, `extract_obligations()`, `extract_regulatory_phrases()` — **transformers** (SmolLM2-360M). |
| `backend/ai/finance/rag/generator.py` | `RAGGenerator.generate()` — **transformers** (SmolLM2-360M). |
| `backend/ai/audio/stt.py` | **faster-whisper** (local STT). |

---

## Files that reference Backboard

| File | Usage |
|------|--------|
| `backend/core/config.py` | Loads `BACKBOARD_API_KEY`, `BACKBOARD_BASE_URL`; validates both at startup. |
| `backend/main.py` | Imports `core.config` (triggers validation). |
| `backend/ai/audio/schema.py` | Docstring: output "suitable for Backboard.io sync". |
| `backend/ai/audio/README.md` | Mentions "Backboard.io sync". |
| `scripts/bootstrap.py` | Validates Backboard env vars. |
| `README.md`, `backend/.env.example` | Document Backboard as required. |

No file performs HTTP requests to Backboard or uses an Backboard SDK.

---

## Deliverables completed

1. **Audit summary** — This file and README section "AI and third-party providers (audit)".
2. **Changed files** — `backend/core/config.py`, `scripts/bootstrap.py`, `backend/.env.example`, `README.md`; new `docs/AUDIT_OPENAI_BACKBOARD.md`.
3. **Backboard AI client** — Not added. Backboard does not expose AI endpoints in this repo; no migration.
4. **README updates** — Env section and checklist use only Backboard credentials; new section documents no OpenAI usage, local-only AI, and that ledger writes are intended for Backboard when implemented.

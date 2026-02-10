# FinSentry

Fintech call analytics: audio → transcript → finance intelligence (intent, entities, obligations, risk, RAG).

## Python virtual environment

The repo expects the **virtual environment inside `backend/`**, at `backend/.venv`. Pyright and tooling are configured for this path.

**macOS / Linux**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
# or, if you use uv:  uv sync
```

**Windows (PowerShell)**

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

**Windows (cmd)**

```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r ..\requirements.txt
```

After activation, run commands from the repo root with the venv active (e.g. `uvicorn main:app --reload` from `backend/`), or use `make run` / `./scripts/dev.sh run` which use `backend/.venv` when present.

## One-command startup (backend + AI pipelines)

From a fresh clone:

```bash
# 1. Install OS-level dependency (required for MP3/M4A audio)
# macOS:
brew install ffmpeg
# Ubuntu/Debian: sudo apt install ffmpeg

# 2. Set required env vars (API won't start without them)
cp backend/.env.example backend/.env
# Edit backend/.env and set: BACKBOARD_API_KEY, BACKBOARD_BASE_URL

# 3. Install Python deps and run backend
make setup && make run
```

Then open **http://localhost:8000** (API) and **http://localhost:8000/docs** (Swagger).

Alternative without Make:

```bash
./scripts/dev.sh setup
./scripts/dev.sh run
```

## Commands

| Command | Description |
|--------|-------------|
| `make setup` | Install dependencies (uv or pip) and run bootstrap checks |
| `make run` | Start FastAPI with uvicorn (port 8000) |
| `make test-audio` | Run AI-1 audio pipeline on a sample file |
| `make test-finance` | Run AI-2 finance pipeline on a sample transcript |
| `make demo` | Run hackathon judge demo (`AUDIO=/path/to/file.(wav|mp3|m4a)`) |
| `make bootstrap` | Check Python ≥3.10, ffmpeg on PATH, and required env vars |

Same via script: `./scripts/dev.sh <setup|run|test-audio|test-finance|bootstrap>`.

## Transcribing new calls (before retraining AI-2)

To turn new call recordings into transcripts for training or evaluation:

1. **Put .wav files** in `backend/data/new_calls/wav/` (create the directory if needed).
2. **Run the batch script** from repo root:
   ```bash
   python scripts/transcribe_new_calls.py
   ```
   This discovers all `.wav` files under `backend/data/new_calls/wav/`, runs the AI-1 pipeline on each, and writes one JSON per file to `backend/data/new_calls/transcripts/<stem>.json`. Existing outputs are skipped unless you pass `--overwrite`.
3. **Options:** `--input-dir`, `--output-dir`, `--overwrite`, `--model-size` (e.g. `base`). Defaults match the `new_calls` paths above.
4. Use the resulting transcript JSONs as input for AI-2 dataset preparation or retraining (e.g. combine into JSONL for `ai.finance.training` or `ai.finance.dataset`).

Run this step before retraining AI-2 so that new audio is transcribed consistently by AI-1.

## Preparing new calls for AI-2 retraining

After you have transcript JSONs in `backend/data/new_calls/transcripts/` (from `transcribe_new_calls.py`), add human-authored labels and build the training JSONL:

1. **Create a labels file** (CSV or JSONL) with at least `call_id`, `intent`, and `risk_level`. Optional columns: `amount`, `currency`, `merchant` (or `counterparty`), `payment_method`, `transaction_date`. Place it at `backend/data/new_calls/labels.csv` (or pass `--labels-file`). You can copy `docs/labels.example.csv` and edit.

2. **Example labels CSV**:

   ```csv
   call_id,amount,currency,merchant,payment_method,transaction_date,intent,risk_level
   call_001,150.00,USD,Acme Corp,card,2025-01-15,dispute,high
   call_002,,,,,inquiry,low
   call_003,25.00,USD,Fee Waiver,,,payment_arrangement,medium
   ```

   Each `call_id` must match the `call_id` inside the corresponding transcript JSON (or the transcript filename stem if `call_id` is missing). `intent` and `risk_level` are required for training; use values consistent with your model (e.g. `dispute`, `inquiry`, `payment_arrangement`; `low`, `medium`, `high`).

3. **Run the preparation script** from repo root:

   ```bash
   make prepare-finance-dataset
   ```

   Or: `python scripts/prepare_finance_training_data.py --transcripts-dir backend/data/new_calls/transcripts --labels-file backend/data/new_calls/labels.csv --output-file backend/data/new_calls/finance_train.jsonl`. The script validates that every transcript has a label, then writes `backend/data/new_calls/finance_train.jsonl` in the canonical schema used by `backend/ai/finance/dataset/format.py` (HF-compatible JSONL).

4. **Example output row** (one line of `finance_train.jsonl`):

   ```json
   {"text": "Customer: I'm calling about a chargeback...", "intent": "dispute", "risk_level": "high", "entities": [], "obligations": [], "regulatory_phrases": [], "id": "call_001"}
   ```

5. Use `backend/data/new_calls/finance_train.jsonl` as input for AI-2 training (see below). The script prints total rows and intent/risk distributions.

## Retraining AI-2 with new call data

After you have `backend/data/new_calls/finance_train.jsonl` (from **Preparing new calls for AI-2 retraining**), you can run LoRA fine-tuning so AI-2 uses both existing HF data and the new-call dataset.

1. **From repo root** (uses `backend/.venv` when present):
   ```bash
   make train-finance
   ```
   This runs the training script with default options: `--extra-dataset data/new_calls/finance_train.jsonl` and `--output models/finance_v2`. It trains only on the new-call JSONL by default (no base `--data`).

2. **Include a base dataset and new-call data** (concatenates both, then shuffles and splits train/val):
   ```bash
   make train-finance ARGS="--data data/finance/synthetic/combined_train.jsonl --extra-dataset data/new_calls/finance_train.jsonl --output models/finance_v2 --epochs 2"
   ```
   Or from the backend directory:
   ```bash
   cd backend
   uv run python -m ai.finance.training.train \
     --data data/finance/synthetic/combined_train.jsonl \
     --extra-dataset data/new_calls/finance_train.jsonl \
     --output models/finance_v2 \
     --epochs 2
   ```

3. **CLI arguments** (run from `backend/` or via `make train-finance ARGS="..."`):
   - `--data` – Base training JSONL (optional if you only use `--extra-dataset`).
   - `--extra-dataset` – Path to extra JSONL (e.g. `data/new_calls/finance_train.jsonl`). Loaded with `datasets.load_dataset("json", data_files=...)`, then concatenated with the base dataset, shuffled, and split into train/val.
   - `--output` / `--output-dir` – Checkpoint directory (default: `models/finance_v2`).
   - `--epochs` – Number of training epochs (default: 2).

   The script prints dataset sizes (train/eval) and epoch metrics (loss, eval_loss). Checkpoints are written to the given output dir under `backend/`. This is training-only; the inference pipeline is unchanged until you point it at the new LoRA weights.

## Environment variables (required for API)

Set in `backend/.env` (or export before starting):

- **BACKBOARD_API_KEY** – Backboard API key
- **BACKBOARD_BASE_URL** – Backboard API base URL

The server **fails startup** if either is missing. Copy `backend/.env.example` to `backend/.env` and fill in the values. No OpenAI or other third-party LLM keys are required (see below).

## Python dependencies (audit)

All Python imports in this repo are satisfied by **`requirements.txt`** (and `backend/pyproject.toml`). There are no missing packages, no duplicate package names, and version specifiers are aligned between the two files. Goal: **`pip install -r requirements.txt` must succeed on a fresh machine** with the venv activated and OS-level deps installed (below). The first install can take several minutes due to `torch` and `transformers`.

## OS-level dependencies

These are **not** installed by pip; install them before or alongside the Python stack.

| Dependency | Used by | Install |
|------------|--------|--------|
| **ffmpeg** | Audio pipeline (librosa: MP3/M4A/OGG/FLAC) | **macOS:** `brew install ffmpeg` · **Ubuntu/Debian:** `sudo apt install ffmpeg` · **Windows:** `choco install ffmpeg` or [ffmpeg.org](https://ffmpeg.org) |
| **libsndfile** (optional on Linux) | `soundfile` (WAV read/write) | If `import soundfile` fails or you build from source: **Ubuntu/Debian:** `sudo apt install libsndfile1` (or `libsndfile1-dev` for build). Many wheels bundle it; macOS/Windows usually need no extra step. |

Run `python scripts/bootstrap.py` to verify Python version, ffmpeg, and env vars.

## 🚀 5-Minute Judge Demo Flow (Mac)

1. **Start the backend (in one terminal)**  
   ```bash
   cd /Users/zion/Documents/Projects/devsoc-finsentry   # repo root
   make setup        # first time only – installs deps and runs bootstrap
   make run          # starts FastAPI on http://localhost:8000
   ```

2. **Run the live judge demo (new terminal)**  
   ```bash
   cd /Users/zion/Documents/Projects/devsoc-finsentry
   make demo AUDIO="/path/to/judge.mp3"   # or .wav / .m4a
   ```

   - The script will:
     - Check `/api/health`
     - Convert the audio to 16k mono WAV via `ffmpeg` (if needed)
     - Run the hardened audio pipeline (Whisper **medium**, VAD, beam search)
     - Run Finance Intelligence (AI-2) in fast, rule-first mode
     - Print a stage-friendly summary (transcript preview, intent, risk, entities)

3. **Inspect the detailed JSON report**  
   - Every run writes a JSON report to:
     ```bash
     demo_outputs/<audio_stem>_report.json
     ```
   - Open this file to see the full pipeline + finance output that was summarized on screen.

## AI and third-party providers (audit)

**No OpenAI or other third-party LLM APIs are used in this repository.** All AI runs locally.

| Component | What it uses | Notes |
|-----------|--------------|--------|
| **AI-1 (audio)** | faster-whisper (local STT) | No external API. |
| **AI-2 (finance)** | sentence-transformers (MiniLM), HuggingFace/transformers (SmolLM2-360M) | Intent/emotion classification and entity/obligation extraction are local-only. No OpenAI, Anthropic, or Gemini. |
| **RAG** | sentence-transformers (embeddings), local SmolLM2 (generation) | Retrieval and generation run on local models. |

**Backboard.io**  
- **BACKBOARD_API_KEY** and **BACKBOARD_BASE_URL** are required at startup for API operability.  
- This codebase does **not** call Backboard for inference, classification, extraction, or embeddings. There is no Backboard AI client or SDK in the repo; AI is entirely local.  
- Outputs (e.g. pipeline results, transcripts) are designed for eventual sync to Backboard (schema is “suitable for Backboard.io sync”). **All ledger writes are intended to go exclusively through Backboard** when that integration is implemented; no other persistence path is used for ledger data.

**Why Backboard cannot replace local AI in this repo yet**  
Backboard does not provide AI endpoints in this implementation. The repo has no Backboard SDK or HTTP client that performs inference/embeddings. Until such an integration exists, AI-2 continues to use local models (sentence-transformers, SmolLM2) with no OpenAI dependency.

## Project layout

- **backend/** – FastAPI app, AI-1 (audio) and AI-2 (finance) modules. Create the Python venv here: **backend/.venv**
- **frontend/** – React/Vite app (separate setup: `cd frontend && pnpm install && pnpm dev`)
- **scripts/** – `bootstrap.py`, `dev.sh`, `test_audio_pipeline.py`, `test_finance_pipeline.py`, `transcribe_new_calls.py`, `prepare_finance_training_data.py`
- **requirements.txt** – Top-level Python deps (mirrors `backend/pyproject.toml`)

Backend can be installed with **uv** (`cd backend && uv sync`, which creates `backend/.venv`) or **pip** (create and activate `backend/.venv` as above, then `pip install -r requirements.txt` from repo root).

---

## Post-Transfer Checklist (frontend integration)

Use this after cloning or moving the repo to a new machine. Verified: **make setup** installs deps and runs bootstrap; **make run** starts the backend with uvicorn on port 8000; uvicorn boots without import errors when env is set; **env validation** blocks startup if Backboard keys (`BACKBOARD_API_KEY`, `BACKBOARD_BASE_URL`) are missing. All steps below from **repo root** unless noted.

1. **Create venv**  
   `cd backend && python3 -m venv .venv && source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows PowerShell).

2. **Install deps**  
   With venv active: `pip install -r ../requirements.txt` (or from repo root: `make setup`, which creates the venv if missing and installs deps).

3. **make setup**  
   From repo root: `make setup`. Installs dependencies (uv or pip) and runs bootstrap (Python ≥3.10, ffmpeg, env vars). Fix any bootstrap errors (e.g. install ffmpeg, create `backend/.env` from `backend/.env.example`).

4. **Test audio**  
   `make test-audio`. Runs AI-1 pipeline on a sample file; first run may download the Whisper model.

5. **Test finance**  
   `make test-finance`. Runs AI-2 pipeline on a sample transcript (no API keys needed).

6. **Run server**  
   `make run`. Starts the backend with uvicorn on **http://localhost:8000**. Requires `backend/.env` with `BACKBOARD_API_KEY` and `BACKBOARD_BASE_URL`; the server will not start if either is missing.

**Quick check:** `make setup && make run` then open http://localhost:8000/docs. Frontend can point at `http://localhost:8000` for the API.

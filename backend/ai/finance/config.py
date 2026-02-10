"""
Configuration for Finance Intelligence models.

All models are open-source and run locally. Tuned for 3B–7B and consumer GPUs.
"""

from pathlib import Path
from typing import Literal

# --- Model choices (hackathon-optimized: fast, small, local) ---

# Intent + Emotion: zero-shot with sentence-transformers (no training needed)
INTENT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, fast

# Entity/Obligation extraction: instruction-tuned small LLM
# Options: "HuggingFaceTB/SmolLM2-360M-Instruct" | "Qwen/Qwen2-0.5B-Instruct" | "microsoft/Phi-2"
EXTRACTION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"  # 360M, ~2GB VRAM
# Fallback for CPU-only: same model, slower
EXTRACTION_MODEL_CPU = "HuggingFaceTB/SmolLM2-360M-Instruct"

# RAG: embeddings + retrieval
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_INDEX_TYPE: Literal["flat", "ivf"] = "flat"  # flat = exact, ivf = approximate for large corpus

# LoRA fine-tuning
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Batch sizes (tune for your GPU)
BATCH_SIZE_INFERENCE = 4
BATCH_SIZE_TRAINING = 2
MAX_SEQ_LENGTH = 512

# Paths
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "finsentry"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "finance"
DEFAULT_MODEL_OUTPUT_DIR = DEFAULT_DATA_DIR / "models"

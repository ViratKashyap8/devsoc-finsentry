"""
Embeddings for RAG retrieval.

Uses sentence-transformers (all-MiniLM-L6-v2) - 80MB, fast, CPU-friendly.
"""

from __future__ import annotations

from typing import Any

_embedding_model: Any = None


def _get_embedding_model(model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_id)
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from None
    return _embedding_model


def embed_texts(
    texts: list[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Embed texts into vectors.

    Returns list of embedding vectors (each is list of floats).
    """
    if not texts:
        return []
    model = _get_embedding_model(model_id)
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return embs.tolist()

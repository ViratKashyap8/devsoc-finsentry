"""
FAISS-based retrieval for finance document corpus.

Indexes policy docs, FAQs, call summaries for RAG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .embeddings import embed_texts

logger = logging.getLogger(__name__)

_faiss_index: Any = None
_index_metadata: list[dict] = []


def build_index_from_documents(
    documents: list[dict],
    text_field: str = "text",
    id_field: str = "id",
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> tuple[Any, list[dict]]:
    """
    Build FAISS index from document list.

    Each document: {text_field: "...", id_field: "...", ...}
    Long texts are chunked for better retrieval.

    Returns (faiss_index, metadata_list) for later retrieval.
    """
    try:
        import numpy as np
        import faiss
    except ImportError:
        raise ImportError(
            "Install faiss: pip install faiss-cpu (or faiss-gpu for CUDA)"
        ) from None

    chunks = []
    metadata = []

    def _chunk(text: str) -> list[str]:
        words = text.split()
        out = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                out.append(chunk)
        return out if out else [text]

    for doc in documents:
        text = doc.get(text_field, "")
        doc_id = doc.get(id_field, str(len(chunks)))
        if not text:
            continue
        for c in _chunk(text):
            chunks.append(c)
            metadata.append({"id": doc_id, "text": c, **{k: v for k, v in doc.items() if k not in (text_field, id_field)}})

    if not chunks:
        return None, []

    embeddings = embed_texts(chunks, model_id=model_id)
    vecs = __import__("numpy").array(embeddings, dtype="float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after norm)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    return index, metadata


class FinanceRetriever:
    """
    FAISS retriever for finance documents.
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        documents: list[dict] | None = None,
    ):
        self.index = None
        self.metadata: list[dict] = []
        if index_path and metadata_path and Path(index_path).exists():
            self._load(index_path, metadata_path)
        elif documents:
            self.index, self.metadata = build_index_from_documents(documents)

    def _load(self, index_path: Path | str, metadata_path: Path | str) -> None:
        try:
            import faiss
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.warning("Failed to load index: %s", e)

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if self.index is None:
            return
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        import faiss
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def search(
        self,
        query: str,
        top_k: int = 5,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> list[tuple[str, float]]:
        """
        Retrieve top_k most similar chunks.

        Returns list of (text, score) where score is cosine similarity.
        """
        if self.index is None or not self.metadata:
            return []
        q_vecs = embed_texts([query], model_id=model_id)
        import numpy as np
        import faiss
        q = np.array(q_vecs, dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.metadata)))
        results = []
        for idx, sc in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.metadata):
                results.append((self.metadata[idx].get("text", ""), float(sc)))
        return results

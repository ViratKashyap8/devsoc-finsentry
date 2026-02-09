"""
Build FAISS index for RAG from call center transcripts.

Usage:
    python -m ai.finance.rag.build_index \
        --corpus data/finance/callcenter/callcenter_rag_corpus.jsonl \
        --output data/finance/callcenter/rag_index
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .retriever import FinanceRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str | Path) -> list[dict]:
    """Load corpus from JSONL file."""
    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus


def build_rag_index(
    corpus_path: str | Path,
    output_dir: str | Path,
    text_field: str = "text",
    id_field: str = "id",
) -> tuple[Path, Path]:
    """
    Build FAISS index from corpus and save.
    
    Returns (index_path, metadata_path).
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load documents
    documents = load_corpus(corpus_path)
    logger.info("Loaded %d documents from %s", len(documents), corpus_path)
    
    if not documents:
        raise ValueError("No documents found in corpus")
    
    # Build index
    retriever = FinanceRetriever(documents=documents)
    
    # Save index
    index_path = output_dir / "faiss.index"
    metadata_path = output_dir / "metadata.json"
    retriever.save(index_path, metadata_path)
    
    logger.info("Saved FAISS index to %s", index_path)
    logger.info("Saved metadata to %s", metadata_path)
    
    # Test retrieval
    test_query = "customer support withdrawal"
    results = retriever.search(test_query, top_k=2)
    logger.info("Test query '%s': %d results", test_query, len(results))
    for text, score in results[:2]:
        logger.info("  Score %.3f: %s...", score, text[:100])
    
    return index_path, metadata_path


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent.parent
    default_corpus = _root / "data" / "finance" / "callcenter" / "callcenter_rag_corpus.jsonl"
    default_output = _root / "data" / "finance" / "callcenter" / "rag_index"
    
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG")
    parser.add_argument("--corpus", default=str(default_corpus), help="Path to corpus JSONL")
    parser.add_argument("--output", default=str(default_output), help="Output directory for index")
    args = parser.parse_args()
    
    index_path, metadata_path = build_rag_index(args.corpus, args.output)
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")

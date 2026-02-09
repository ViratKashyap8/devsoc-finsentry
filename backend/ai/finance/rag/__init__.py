"""RAG module: retrieval + generation for finance Q&A."""

from .retriever import FinanceRetriever, build_index_from_documents
from .generator import RAGGenerator
from .embeddings import embed_texts

__all__ = ["FinanceRetriever", "build_index_from_documents", "RAGGenerator", "embed_texts"]

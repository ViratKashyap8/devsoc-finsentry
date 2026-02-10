"""
RAG generator: retrieve + generate with local LLM.

Uses same extraction model (SmolLM2-360M) for consistency.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional

from ..schema import RAGResponse

logger = logging.getLogger(__name__)

_generator_model_cache: tuple = ()


def _get_generator(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    global _generator_model_cache
    if not _generator_model_cache:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                model = model.to(device)
            _generator_model_cache = (tokenizer, model, device)
        except ImportError as e:
            raise ImportError("Install transformers, torch") from e
    return _generator_model_cache


RAG_PROMPT = """Use the following context to answer the question. If the context does not contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""


class RAGGenerator:
    """Retrieve + generate answers from finance documents."""

    def __init__(
        self,
        retriever: Any,
        model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    ):
        self.retriever = retriever
        self.model_id = model_id

    def generate(
        self,
        query: str,
        top_k: int = 3,
        max_new_tokens: int = 150,
    ) -> RAGResponse:
        """Retrieve context and generate answer."""
        retrieved = self.retriever.search(query, top_k=top_k)
        contexts = [r[0] for r in retrieved]
        context = "\n\n".join(contexts) if contexts else "No relevant context found."

        prompt = RAG_PROMPT.format(context=context[:2000], query=query)

        try:
            tokenizer, model, device = _get_generator(self.model_id)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            import torch
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning("Generation failed: %s", e)
            answer = "I could not generate an answer. Please try rephrasing or check the context."

        return RAGResponse(
            query=query,
            retrieved_contexts=contexts,
            answer=answer,
            sources=[f"doc_{i}" for i in range(len(contexts))],
        )

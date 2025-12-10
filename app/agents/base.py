from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import textwrap

import numpy as np
from crewai import LLM

from app.rag.vector_store import FaissIndex
from app.rag.embeddings import embed_texts


# --- LLM config -------------------------------------------------------------

DEFAULT_MODEL = (
    os.getenv("CREWAI_MODEL")
    or os.getenv("OLLAMA_MODEL")
    or "ollama/gemma3:4b"  # change if your Ollama model name is different
)

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class CrewConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.1
    max_tokens: Optional[int] = None


def make_llm(config: Optional[CrewConfig] = None) -> LLM:
    """
    Build a CrewAI LLM that talks to your local Ollama server.

    We normalize the model name so that LiteLLM knows to use the
    'ollama' provider even if the env var is just 'gemma3:4b', 'llama3', etc.
    """
    if config is None:
        config = CrewConfig()

    # Start from env / default
    model = config.model or "ollama/mistral"

    
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"

    return LLM(
        model=model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )



# RAG-style context retrieval 


def _load_index(index_dir: str) -> FaissIndex:
    """
    Thin wrapper in case we want extra logging later.
    """
    return FaissIndex.load(index_dir)


def retrieve_contexts(
    index_dir: str,
    question: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Use your existing FAISS index + embeddings to get top-k chunks for a question.

    Returns a list of dicts:
    [
        {
          "chunk_id": int,
          "text": str,
          "source_id": str,
          "page_num": int | None,
          "score": float,
        },
        ...
    ]
    """
    index = _load_index(index_dir)

    # Embed the question (reusing your embed_texts function)
    q_emb: np.ndarray = embed_texts([question])[0]

    # Search FAISS
    indices, distances = index.search(q_emb, top_k)

    contexts: List[Dict[str, Any]] = []
    for idx, dist in zip(indices, distances):
        if idx < 0 or idx >= len(index.metadata):
            continue

        meta = index.metadata[idx] or {}
        contexts.append(
            {
                "chunk_id": int(idx),
                "text": meta.get("text", "") or "",
                "source_id": meta.get("source_id", "") or "",
                "page_num": meta.get("page_num", None),
                "score": float(dist),
            }
        )

    return contexts


def format_contexts_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """
    Turn context dicts into a readable string for prompts.
    """
    if not contexts:
        return "NO CONTEXT FOUND."

    parts: List[str] = []
    for i, c in enumerate(contexts, start=1):
        score = c.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (float, int)) else "n/a"
        page = c.get("page_num")
        page_str = str(page) if page is not None else "unknown"

        chunk_text = (c.get("text") or "").strip()

        block = textwrap.dedent(
            f"""
            [Chunk {i} | id={c.get("chunk_id")} | page={page_str} | score={score_str}]
            {chunk_text}
            """
        ).strip()

        parts.append(block)

    return "\n\n".join(parts)

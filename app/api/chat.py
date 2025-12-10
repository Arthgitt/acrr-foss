from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.base import retrieve_contexts, format_contexts_for_prompt

router = APIRouter(prefix="/chat", tags=["chat"])

# Where your vector stores live
BASE_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/vector_stores"))

# Ollama config 
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------- Pydantic models ----------


class ChatRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 5


class ContextChunk(BaseModel):
    chunk_id: int
    text: str
    source_id: Optional[str] = None
    page_num: Optional[int] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    contexts: List[ContextChunk]


# ---------- Internal helpers ----------


def _get_index_dir(doc_id: str) -> Path:
    index_dir = BASE_INDEX_DIR / doc_id
    if not index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index directory not found for doc_id '{doc_id}' at {index_dir}",
        )
    return index_dir


def _build_prompt(question: str, context_text: str) -> str:
    return (
        "You are a helpful assistant for analyzing financial, loan, and compliance "
        "documents. Use ONLY the information from the context below. "
        "If the answer is not clearly supported by the context, say you don't know.\n\n"
        "=== DOCUMENT CONTEXT ===\n"
        f"{context_text}\n\n"
        "=== QUESTION ===\n"
        f"{question}\n\n"
        "=== ANSWER ===\n"
    )


def _call_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM error while generating answer: {exc}",
        ) from exc


def _run_rag(doc_id: str, question: str, top_k: int) -> ChatResponse:
    index_dir = _get_index_dir(doc_id)

    # 1) Retrieve contexts via the same FAISS+embeddings used by CrewAI
    contexts_raw = retrieve_contexts(
        index_dir=str(index_dir),
        question=question,
        top_k=top_k,
    )

    if not contexts_raw:
        return ChatResponse(answer="No relevant context found.", contexts=[])

    # 2) Format for prompt
    context_text = format_contexts_for_prompt(contexts_raw)
    prompt = _build_prompt(question, context_text)

    # 3) Call Ollama
    answer = _call_ollama(prompt)

    # 4) Convert dicts -> Pydantic models
    contexts = [
        ContextChunk(
            chunk_id=c.get("chunk_id", -1),
            text=c.get("text", "") or "",
            source_id=c.get("source_id"),
            page_num=c.get("page_num"),
            score=c.get("score"),
        )
        for c in contexts_raw
    ]

    return ChatResponse(answer=answer, contexts=contexts)


# ---------- Endpoints ----------


@router.post("/rag", response_model=ChatResponse)
def chat_rag(req: ChatRequest) -> ChatResponse:
    """
    Main RAG endpoint: search FAISS, build prompt, ask local Ollama, return answer + contexts.
    """
    return _run_rag(req.doc_id, req.question, req.top_k)



@router.post("/ask", response_model=ChatResponse)
def chat_ask(req: ChatRequest) -> ChatResponse:
    """
    Alias for /chat/rag so the frontend can POST to either /chat/ask or /chat/rag.
    """
    return _run_rag(req.doc_id, req.question, req.top_k)

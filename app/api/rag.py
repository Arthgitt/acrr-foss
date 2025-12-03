# app/api/rag.py

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.rag.rag_pipeline import RAGPipeline, get_rag_pipeline


router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)


# ---------- Request / Response models ----------

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    include_contexts: bool = True


class ContextChunk(BaseModel):
    text: str
    source_id: Optional[str] = None
    page_num: Optional[int] = None
    score: Optional[float] = None


class RAGQueryResponse(BaseModel):
    answer: str
    contexts: Optional[List[ContextChunk]] = None


# ---------- Endpoint ----------

@router.post("/query", response_model=RAGQueryResponse)
def rag_query(
    payload: RAGQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGQueryResponse:
    """
    POST /api/rag/query

    Body:
    {
      "question": "your question here",
      "top_k": 5,
      "include_contexts": true
    }
    """
    try:
        result: Dict[str, Any] = pipeline.query(
            question=payload.question,
            top_k=payload.top_k,
            include_contexts=payload.include_contexts,
        )
    except Exception as e:
        # This error message will show in the JSON if something goes wrong
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    contexts_data = None
    if payload.include_contexts:
        raw_contexts = result.get("contexts") or []
        contexts_data = [
            ContextChunk(
                text=c.get("text", ""),
                source_id=c.get("source_id"),
                page_num=c.get("page_num"),
                score=c.get("score"),
            )
            for c in raw_contexts
        ]

    return RAGQueryResponse(
        answer=result.get("answer", ""),
        contexts=contexts_data,
    )

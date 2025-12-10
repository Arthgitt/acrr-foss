from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from app.extractors.mortgage_key_fields import extract_mortgage_key_fields


from app.agents.orchestrator import (
    run_financial_crew_chat,
    run_financial_analysis,
)

router = APIRouter(prefix="/agents", tags=["agents"])

# Base directory where your FAISS indices live
BASE_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/vector_stores"))

# Base directory where layout-block JSONs are stored
BASE_LAYOUT_DIR = Path("data") / "layout_blocks"


# --------------- Shared models -----------------


class CrewContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chunk_id: int
    text: str
    source_id: str | None = None
    page_num: int | None = None
    score: float | None = None


# --------------- Q&A (crew-chat) -----------------


class CrewChatRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 6


class CrewChatResponse(BaseModel):
    answer: str
    contexts: List[CrewContext]


@router.post("/financial/crew-chat", response_model=CrewChatResponse)
def crew_financial_chat(req: CrewChatRequest) -> CrewChatResponse:
    """
    Run the CrewAI-based multi-agent analysis for a financial document (Q&A).
    """
    index_dir = BASE_INDEX_DIR / req.doc_id

    if not index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index directory not found for doc_id '{req.doc_id}' at {index_dir}",
        )

    try:
        result: Dict[str, Any] = run_financial_crew_chat(
            index_dir=str(index_dir),
            question=req.question,
            top_k=req.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CrewAI error while answering question: {exc}",
        ) from exc

    return CrewChatResponse(**result)


# --------------- Full document analysis -----------------


class AnalysisRequest(BaseModel):
    doc_id: str
    top_k: int = 12


class AgentAnalysisResult(BaseModel):
    agent: str
    title: str
    markdown: str
    contexts: List[CrewContext]


class AnalysisResponse(BaseModel):
    doc_id: str
    results: List[AgentAnalysisResult]


# --------------- Mortgage key-fields extractor -----------------


class KeyFieldsRequest(BaseModel):
    doc_id: str
    top_k: int = 30


class KeyFieldsResponse(BaseModel):
    doc_id: str
    key_fields: Dict[str, Any]
    used_layout: bool = False


@router.post("/financial/key-fields", response_model=KeyFieldsResponse)
def financial_key_fields(req: KeyFieldsRequest) -> KeyFieldsResponse:
    """
    Run the mortgage-specific key-fields extractor for a given doc_id.
    Uses the existing FAISS index + (optionally) layout_blocks JSON.
    """
    index_dir = BASE_INDEX_DIR / req.doc_id

    if not index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index directory not found for doc_id '{req.doc_id}' at {index_dir}",
        )

    try:
        result = extract_mortgage_key_fields(
            index_dir=str(index_dir),
            top_k=req.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error running mortgage key-fields extractor: {exc}",
        ) from exc

    return KeyFieldsResponse(
        doc_id=result.get("doc_id", req.doc_id),
        key_fields=result.get("key_fields", {}),
        used_layout=bool(result.get("used_layout")),
    )


@router.post("/financial/analyze", response_model=AnalysisResponse)
def financial_analyze(req: AnalysisRequest) -> AnalysisResponse:
    """
    Run full multi-agent document analysis (Overview, Numeric, Checklist, Risk, Cross-validation).

    Also tries to attach layout-block JSON stored at:
    data/layout_blocks/<doc_id>/layout_blocks.json
    """
    index_dir = BASE_INDEX_DIR / req.doc_id

    if not index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index directory not found for doc_id '{req.doc_id}' at {index_dir}",
        )

    layout_json_path = BASE_LAYOUT_DIR / req.doc_id / "layout_blocks.json"

    # 1) Call the orchestrator
    try:
        result: Dict[str, Any] = run_financial_analysis(
            index_dir=str(index_dir),
            top_k=req.top_k,
            layout_json_path=layout_json_path if layout_json_path.exists() else None,
        )
    except Exception as exc:
        print("[financial_analyze] run_financial_analysis crashed:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"CrewAI error while running document analysis: {exc}",
        ) from exc

    print("[financial_analyze] result keys:", list(result.keys()))

    # 2) Convert result dict -> Pydantic model with proper error reporting
    try:
        response_obj = AnalysisResponse(**result)
    except Exception as exc:
        print("[financial_analyze] Failed to build AnalysisResponse:", repr(exc))
        print("[financial_analyze] Raw result:", result)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Error converting run_financial_analysis result into AnalysisResponse: {exc}; "
                f"got keys={list(result.keys())}"
            ),
        ) from exc

    return response_obj

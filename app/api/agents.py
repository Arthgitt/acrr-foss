# app/api/agents.py

from __future__ import annotations

from fastapi import APIRouter

from app.agents.orchestrator import DocumentAnalysisOrchestrator

router = APIRouter(
    prefix="/api/agents",
    tags=["agents"],
)


@router.post("/analyze/{doc_id}")
async def analyze_document(doc_id: str):
    """
    Run all agents on the given document and return their outputs.
    """
    orchestrator = DocumentAnalysisOrchestrator()
    result = orchestrator.run(doc_id=doc_id)
    return result

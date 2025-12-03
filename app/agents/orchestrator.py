# app/agents/orchestrator.py

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import HTTPException

from app.rag.vector_store import FaissIndex
from app.rag.rag_pipeline import RAGPipeline
from app.agents.base import BaseAgent
from app.agents.financial_agents import OverviewAgent, RiskAgent, ChecklistAgent


class DocumentAnalysisOrchestrator:
    """
    Orchestrates running multiple agents over a single document.
    """

    def __init__(self, agents: List[BaseAgent] | None = None):
        if agents is None:
            # Default set of agents for now
            agents = [
                OverviewAgent(),
                RiskAgent(),
                ChecklistAgent(),
            ]
        self.agents = agents

    def _build_pipeline_for_doc(self, doc_id: str) -> RAGPipeline:
        """
        Load FAISS index for the given doc_id and build a RAGPipeline.
        Assumes indices are stored under RAG_INDEX_DIR/<doc_id>.
        """
        # Root directory where all per-document indices live
        # e.g. RAG_INDEX_DIR="data/vector_stores"
        base_dir = os.getenv("RAG_INDEX_DIR", "data/vector_stores")
        index_dir = Path(base_dir) / doc_id

        if not index_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No index found for doc_id='{doc_id}' in {index_dir}",
            )

        index = FaissIndex.load(index_dir)

        # Use the SAME env vars as rag_pipeline.get_rag_pipeline
        ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "mistral")

        pipeline = RAGPipeline(
            index=index,
            ollama_url=ollama_url,
            model_name=model_name,
        )
        return pipeline

    def run(self, doc_id: str) -> Dict[str, Any]:
        """
        Run all agents on a given document and collect the results.
        """
        pipeline = self._build_pipeline_for_doc(doc_id)

        results: List[Dict[str, Any]] = []
        for agent in self.agents:
            agent_result = agent.run(pipeline)
            results.append(agent_result)

        return {
            "doc_id": doc_id,
            "agent_count": len(self.agents),
            "results": results,
        }

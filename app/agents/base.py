# app/agents/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

from app.rag.rag_pipeline import RAGPipeline


class BaseAgent(ABC):
    """
    Simple base class for all agents.
    Each agent:
    - has a name and description
    - implements `run()` which uses a RAGPipeline
    """

    name: str = "base-agent"
    description: str = "Base agent"

    @abstractmethod
    def run(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        """
        Execute the agent using a RAG pipeline.
        Returns a dict that can be JSON-serialized.
        """
        raise NotImplementedError

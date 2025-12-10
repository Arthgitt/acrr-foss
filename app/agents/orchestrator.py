from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json

from app.agents.financial_agents import (
    run_financial_crew,
    run_document_analysis_crew,
)


def run_financial_crew_chat(
    index_dir: str,
    question: str,
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    Thin wrapper so other parts of the app only depend on this module for
    Q&A with the financial Crew.
    """
    return run_financial_crew(
        index_dir=index_dir,
        question=question,
        top_k=top_k,
    )


def run_financial_analysis(
    index_dir: str,
    top_k: int = 12,
    layout_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Wrapper for full multi-agent document analysis.

    If layout_json_path is provided and exists, we load the JSON with
    layout blocks (bounding boxes, page numbers, etc.) and pass it
    down to the Crew so agents can optionally use it.

    This keeps the API (/agents/financial/analyze) decoupled from the
    internal Crew implementation in app.agents.financial_agents.
    """
    layout_blocks = None

    # layout_json_path is optional and best-effort: if anything goes wrong
    if layout_json_path is not None and layout_json_path.exists():
        try:
            with layout_json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            # accept either a list of blocks or a dict with "blocks" key
            if isinstance(loaded, list):
                layout_blocks = loaded
            elif isinstance(loaded, dict) and "blocks" in loaded:
                layout_blocks = loaded["blocks"]
        except Exception as e:
            print(f"[run_financial_analysis] Failed to load layout blocks: {e}")
            layout_blocks = None

    # Delegate to the CrewAI implementation
    return run_document_analysis_crew(
        index_dir=index_dir,
        top_k=top_k,
        layout_blocks=layout_blocks,
    )

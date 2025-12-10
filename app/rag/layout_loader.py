from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json


LAYOUT_ROOT = Path("data") / "layout_blocks"


def load_layout_blocks(doc_id: str) -> List[Dict[str, Any]]:
    """
    Load layout blocks JSON saved by the Streamlit app.
    Returns [] if file is missing or invalid.
    """
    path = LAYOUT_ROOT / f"{doc_id}.json"
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def layout_blocks_as_compact_json(doc_id: str, max_chars: int = 4000) -> str:
    """
    Return a compact JSON string for LLM prompts (truncated to max_chars).
    """
    blocks = load_layout_blocks(doc_id)
    if not blocks:
        return ""

    raw = json.dumps(blocks, ensure_ascii=False)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "... (truncated)"

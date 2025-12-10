"""
Simple text chunking utilities

We keep it very basic:
- Split on character length (max_chars)
- Optional overlap between chunks
"""

from typing import List, Dict, Any


def make_char_chunks(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split a long string into overlapping character chunks.

    Returns a list like:
    [
        {"id": 0, "start": 0, "end": 800, "text": "..."},
        {"id": 1, "start": 700, "end": 1500, "text": "..."},
        ...
    ]
    """
    if not text:
        return []

    chunks: List[Dict[str, Any]] = []
    n = len(text)
    start = 0
    chunk_id = 0

    
    max_chars = max(1, max_chars)
    overlap = max(0, overlap)

    while start < n:
        end = min(start + max_chars, n)
        chunk_text = text[start:end]

        chunks.append(
            {
                "id": chunk_id,
                "start": start,
                "end": end,
                "text": chunk_text,
            }
        )

        chunk_id += 1
        # Move start forward, but keep some overlap
        if end == n:
            break
        start = end - overlap

    return chunks

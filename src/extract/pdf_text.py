"""
PDF text extraction utilities using PyMuPDF (fitz).

For Step 2 of the ACRR FOSS project:
- Input: PDF bytes (from Streamlit upload)
- Output: list of pages with page number and plain text.
"""

from typing import List, Dict, Any

import fitz  # PyMuPDF


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract plain text from each page of a PDF given as bytes.

    Returns a list of dicts like:
    [
        {"page_num": 1, "text": "..."},
        {"page_num": 2, "text": "..."},
        ...
    ]
    """
    pages: List[Dict[str, Any]] = []

    # Open the PDF from bytes
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # "text" gives us text in reading order
            text = page.get_text("text")

            pages.append(
                {
                    "page_num": page_index + 1,
                    "text": text,
                }
            )

    return pages


def join_pages_to_single_text(pages: List[Dict[str, Any]]) -> str:
    """
    Helper: join per-page text into one long string,
    with page separators.
    """
    chunks = []
    for p in pages:
        header = f"\n\n--- Page {p['page_num']} ---\n\n"
        chunks.append(header + (p["text"] or ""))

    return "".join(chunks)

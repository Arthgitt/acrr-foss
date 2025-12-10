"""
PDF text extraction utilities using PyMuPDF (fitz).

Step 2 + OCR fallback upgrade:
- If normal text extraction fails (image-only pages),
  fallback to pytesseract OCR to extract text from rendered page image.
"""

from typing import List, Dict, Any

import pymupdf  # PyMuPDF

from src.extract.ocr import ocr_page   # OCR fallback imported here


# ----------------------------------------------------------
# MAIN PDF TEXT EXTRACTOR (with OCR fallback)
# ----------------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from PDF bytes.
    If a page contains no extractable text, OCR is used instead.

    Returns:
        [
            {"page_num": 1, "text": "..."},
            {"page_num": 2, "text": "..."},
            ...
        ]
    """
    pages: List[Dict[str, Any]] = []

    # Open the PDF from bytes
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]

            # Try normal text extraction first
            text: str = page.get_text("text") or ""

            # If empty, run OCR fallback
            if not text.strip():
                try:
                    text = ocr_page(page)
                    print(f"[OCR] Page {page_index + 1} processed via OCR fallback")
                except Exception as e:
                    print(f"[OCR ERROR] Page {page_index + 1}: {e}")
                    text = ""  # fail safe

            pages.append(
                {
                    "page_num": page_index + 1,
                    "text": text,
                }
            )

    return pages


# ----------------------------------------------------------
# Join pages into one text block (used later for embeddings)
# ----------------------------------------------------------
def join_pages_to_single_text(pages: List[Dict[str, Any]]) -> str:
    """
    Join multiple page text sections into a single long text block for RAG.
    """
    combined = []
    for p in pages:
        header = f"\n\n--- Page {p['page_num']} ---\n\n"
        combined.append(header + (p["text"] or ""))

    return "".join(combined)

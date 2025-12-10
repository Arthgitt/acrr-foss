from __future__ import annotations

from typing import Optional

import pytesseract
from PIL import Image
import fitz  # PyMuPDF


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """
    Render a PyMuPDF page to an image and run Tesseract OCR.
    """
    # matrix to control resolution
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text or ""

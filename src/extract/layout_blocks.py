from typing import List, Dict, Any
import pymupdf  # PyMuPDF
import re


def extract_layout_blocks_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract layout-aware text blocks from a PDF, including bounding boxes.

    Returns a list of dicts like:
    [
        {
            "page_num": 1,
            "bbox": [x0, y0, x1, y1],
            "text": "...",
            "block_no": 0,
            "block_type": "text"  # text / image / other
        },
        ...
    ]
    """
    blocks: List[Dict[str, Any]] = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            # Each item: (x0, y0, x1, y1, text, block_no, block_type)
            for b in page.get_text("blocks"):
                if len(b) < 5:
                    continue

                x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                text = b[4] or ""
                text = text.strip()
                if not text:
                    continue

                block_no = int(b[5]) if len(b) > 5 else -1
                raw_type = int(b[6]) if len(b) > 6 else 0

                
                if raw_type == 0:
                    block_type = "text"
                elif raw_type == 1:
                    block_type = "image"
                elif raw_type == 2:
                    block_type = "vector"
                elif raw_type == 3:
                    block_type = "annotation"
                else:
                    block_type = "other"

                blocks.append(
                    {
                        "page_num": page_num,
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "text": text,
                        "block_no": block_no,
                        "block_type": block_type,
                    }
                )

    return blocks

# simple heuristic field-finder using layout blocks
def find_key_values_by_keyword(
    blocks: List[Dict[str, Any]],
    keyword: str,
    max_vertical_distance: float = 20.0,
    max_horizontal_distance: float = 250.0,
) -> List[Dict[str, Any]]:
    """
    Given layout blocks and a keyword/label (e.g. 'Total Loan Amount'),
    return candidate numeric values that appear in the same block or
    in nearby blocks to the right on the same line.

    Returns a list of dicts like:
    [
        {
            "keyword_block": {...},
            "value_candidates": ["380,000.00", "95,641.53"],
        },
        ...
    ]
    """
    keyword_lower = keyword.lower()
    results: List[Dict[str, Any]] = []

    # Regex: simple money/number detector: $ 380,000.00, 950.50, 150, etc.
    num_pattern = re.compile(r"[+-]?\$?\s*\d[\d,]*\.?\d*")

    for i, blk in enumerate(blocks):
        text = blk.get("text", "")
        if keyword_lower not in text.lower():
            continue

        page = blk["page_num"]
        x0, y0, x1, y1 = blk["bbox"]

        # numbers in the same block
        candidates = [m.group(0).strip() for m in num_pattern.finditer(text)]

        # if none, search blocks on the same page, roughly same vertical line, to the right
        if not candidates:
            for j, other in enumerate(blocks):
                if j == i or other.get("page_num") != page:
                    continue

                ox0, oy0, ox1, oy1 = other["bbox"]

                same_row = abs(oy0 - y0) <= max_vertical_distance
                to_the_right = ox0 >= x1 and (ox0 - x1) <= max_horizontal_distance

                if same_row and to_the_right:
                    for m in num_pattern.finditer(other.get("text", "")):
                        val = m.group(0).strip()
                        if val:  # avoid empty strings
                            candidates.append(val)

        results.append(
            {
                "keyword_block": blk,
                "value_candidates": candidates,
            }
        )

    return results
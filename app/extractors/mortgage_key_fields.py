from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

from crewai import Agent, Task, Crew, Process

from app.agents.base import make_llm, retrieve_contexts, format_contexts_for_prompt


def _load_layout_blocks_doc(index_dir: str, max_blocks: int = 80) -> List[dict]:
    """
    Load layout_blocks JSON that Streamlit saved under data/layout_blocks/{doc_id}.json.
    Return a *list* of dicts (possibly trimmed to max_blocks).
    """
    doc_id = Path(index_dir).name
    layout_path = Path("data") / "layout_blocks" / f"{doc_id}.json"

    if not layout_path.exists():
        return []

    try:
        with layout_path.open("r", encoding="utf-8") as f:
            blocks: List[dict] = json.load(f)
    except Exception:
        return []

    if len(blocks) > max_blocks:
        return blocks[:max_blocks]
    return blocks


def _format_layout_blocks_for_prompt(blocks: List[dict]) -> str:
    """
    Turn layout_blocks into a compact markdown table so the LLM can see
    which labels & values are in the same row / near each other.
    """
    if not blocks:
        return ""

    lines: List[str] = []
    lines.append(
        "Layout-based view (each row is one text block with its bounding box):"
    )
    lines.append("")
    lines.append("| # | Page | Type | BBox [x0,y0,x1,y1] | Text snippet |")
    lines.append("|---|------|------|--------------------|--------------|")

    for i, b in enumerate(blocks, start=1):
        page = b.get("page_num") or b.get("page") or "?"
        btype = b.get("block_type") or b.get("type", "text")
        bbox = b.get("bbox", "")
        text = (b.get("text") or "").replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        lines.append(f"| {i} | {page} | {btype} | {bbox} | {text} |")

    return "\n".join(lines)


def extract_mortgage_key_fields(
    index_dir: str,
    top_k: int = 30,
) -> Dict[str, Any]:
    """
    High-level pipeline:

    1. Use FAISS index to retrieve top_k text chunks for a generic 'mortgage overview' question.
    2. Load layout_blocks for this doc_id if available.
    3. Ask a specialized LLM agent to produce **strict JSON** with mortgage key fields.
    4. Save that JSON under data/key_fields/{doc_id}.json and return it.
    """
    doc_id = Path(index_dir).name

    # --- 1) Retrieve text chunks via FAISS ---
    question = (
        "Summarize this mortgage-related document (loan estimate, lender fees "
        "worksheet, closing disclosure, or title report) and extract all key "
        "numeric and party details."
    )
    contexts = retrieve_contexts(index_dir=index_dir, question=question, top_k=top_k)

    if not contexts:
        return {
            "doc_id": doc_id,
            "key_fields": {},
            "raw_text": "",
            "used_layout": False,
        }

    context_text = format_contexts_for_prompt(contexts)

    # --- Load layout_blocks snapshot (if present) ---
    layout_blocks = _load_layout_blocks_doc(index_dir)
    layout_md = _format_layout_blocks_for_prompt(layout_blocks)
    used_layout = bool(layout_blocks)

    if layout_md:
        combined_context = (
            context_text
            + "\n\n---\n\n"
            + "Additional layout-based view (labels & values that sit on the same rows):\n\n"
            + layout_md
        )
    else:
        combined_context = context_text

    # --- One-agent CrewAI run to get **strict JSON** ---
    llm = make_llm()

    extractor_agent = Agent(
        role="Mortgage Key Fields Extractor",
        goal=(
            "Read mortgage-related PDFs (loan estimate, lender fees worksheet, "
            "closing disclosure, bank statement, title report) and extract all "
            "important structured fields in strict JSON."
        ),
        backstory=(
            "You are a senior mortgage underwriter. You know standard US mortgage "
            "documents and typical field names. You never hallucinate: if a field is "
            "missing, you set it to null. You never invent numbers."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # ⚠️ NOTE: double braces {{ }} are needed inside f-strings where we show JSON
    description = f"""
You are given context from a mortgage-related document plus (optionally) a
layout-based view that shows which text blocks are on the same row.

Context (text + optional layout table):

{combined_context}

Your job is to output a **single JSON object only**, no explanations.

The JSON MUST match this schema (values can be null if unknown):

{{
  "document_type": string | null,          // e.g. "loan_estimate", "lender_fees_worksheet", "title_report"
  "loan_terms": {{
    "loan_amount": number | null,
    "interest_rate_percent": number | null,     // e.g. 4.25 for 4.25%
    "term_months": integer | null,
    "rate_type": string | null                 // e.g. "fixed", "ARM", "HELOC", etc.
  }},
  "property": {{
    "address": string | null,
    "city": string | null,
    "state": string | null,
    "zip": string | null
  }},
  "parties": {{
    "borrower_names": [string],    // list, may be empty
    "lender_name": string | null,
    "other_parties": [string]      // e.g. title company, escrow company
  }},
  "payment_info": {{
    "estimated_monthly_payment": number | null,
    "principal_interest": number | null,
    "escrow": number | null,                  // taxes+insurance escrow if given
    "first_payment_date": string | null       // free-text date like "01/01/2026"
  }},
  "fee_items": [
    {{
      "label": string,
      "amount": number | null,
      "category": string | null               // e.g. "origination", "title", "government", "escrow"
    }}
  ],
  "risk_notes": [
    string                                    // short bullet-style comments about anything unusual or missing
  ],
  "other_numeric_fields": [
    {{
      "section": string | null,              // e.g. "Prepaids", "Escrow", "Adjustments"
      "label": string,                       // e.g. "Daily Interest Charges"
      "value": number | null,               // numeric value only
      "unit": string | null,                // e.g. "USD", "%", "days"
      "notes": string | null                // e.g. "calculated over 25 days"
    }}
  ],
  "other_text_fields": [
    {{
      "section": string | null,              // e.g. "Disclosures", "Loan Terms"
      "label": string,                       // e.g. "Rate lock expiration"
      "value": string | null                 // e.g. "10/15/2025"
    }}
  ]
}}

Rules:
- Use ONLY information from the provided context. Do NOT guess or invent values.
- If you cannot find a field, set it to null (or [] for lists).
- All currency amounts should be numbers only, no dollar signs or commas.
- If you see numeric values that do not clearly belong to loan_terms, payment_info, or fee_items
  (for example, per-diem interest, prepaid taxes, HOA dues, reserves), capture them in
  other_numeric_fields with a good section and label.
- If you see extra important textual information (for example, disclosures, rate lock
  expiration, prepayment penalty notes) that does not fit the main fields, capture it
  in other_text_fields.
- Output **ONLY** the JSON object, no markdown, no code fences, no commentary.
"""

    task = Task(
        description=description,
        expected_output=(
            "A single valid JSON object with the exact schema shown above, including "
            "other_numeric_fields and other_text_fields."
        ),
        agent=extractor_agent,
    )

    crew = Crew(
        agents=[extractor_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    crew_result = crew.kickoff()
    raw_output = getattr(crew_result, "output", str(crew_result))

    # --- Parse JSON and save to disk ---
    try:
        key_fields = json.loads(raw_output)
    except json.JSONDecodeError:
        # if the model wrapped it in ```json``` or similar, try to clean
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove possible "json" prefix
            cleaned = cleaned.replace("json", "", 1).strip()
        try:
            key_fields = json.loads(cleaned)
        except Exception:
            key_fields = {
                "parse_error": "Model did not return valid JSON",
                "raw_output": raw_output,
            }

    out_dir = Path("data") / "key_fields"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(key_fields, f, ensure_ascii=False, indent=2)

    # expose combined_context as raw_text for downstream agents
    raw_text = combined_context

    return {
        "doc_id": doc_id,
        "key_fields": key_fields,
        "used_layout": used_layout,
        "raw_text": raw_text,
    }

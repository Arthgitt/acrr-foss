# app/agents/financial_agents.py

from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
import json
import re  # NEW: for small post-processing of markdown

from crewai import Agent, Task, Crew, Process

from app.agents.base import make_llm, retrieve_contexts, format_contexts_for_prompt
from app.extractors.mortgage_key_fields import extract_mortgage_key_fields


# ----------------------------------------------------------------------
# Helper: load layout blocks JSON and build a short “layout snapshot”
# ----------------------------------------------------------------------


def _load_layout_snippet(index_dir: str, max_blocks: int = 60) -> str:
    """
    Load layout_blocks JSON saved by Streamlit and turn it into a short
    markdown-friendly snapshot.

    The goal is NOT to dump the whole JSON, but to give the agents a sense
    of which values sit on the same row/near which labels.
    """
    doc_id = Path(index_dir).name
    layout_path = Path("data") / "layout_blocks" / f"{doc_id}.json"

    if not layout_path.exists():
        return ""

    try:
        with layout_path.open("r", encoding="utf-8") as f:
            blocks = json.load(f)
    except Exception:
        return ""

    if not isinstance(blocks, list):
        return ""

    lines: List[str] = []
    for b in blocks[:max_blocks]:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        page = b.get("page_num", b.get("page", "?"))
        bbox = b.get("bbox", [])
        snippet = text.replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        lines.append(f"- Page {page}, bbox={bbox}: {snippet}")

    if not lines:
        return ""

    header = (
        "Layout snapshot (text blocks with approximate positions).\n"
        "Each bullet is one block on the page:\n"
    )
    return header + "\n".join(lines)


# ----------------------------------------------------------------------
# NEW helpers: structured mortgage key_fields JSON for prompts
# ----------------------------------------------------------------------


def _load_or_build_structured_json(index_dir: str) -> Dict[str, Any]:
    """
    Load data/key_fields/{doc_id}.json (created by extract_mortgage_key_fields).
    If it doesn't exist or is invalid, call extract_mortgage_key_fields to build it.
    Returns the inner key_fields dict (with document_type, loan_terms, etc.).
    """
    doc_id = Path(index_dir).name
    key_path = Path("data") / "key_fields" / f"{doc_id}.json"

    data: Dict[str, Any] = {}

    if key_path.exists():
        try:
            with key_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}

    
    if not data:
        try:
            result = extract_mortgage_key_fields(index_dir=index_dir)
            if isinstance(result, dict):
                key_fields = result.get("key_fields")
                if isinstance(key_fields, dict):
                    data = key_fields
        except Exception:
            # Silent failure – we'll just return {}
            data = {}

    return data


def _structured_json_to_markdown(data: Dict[str, Any]) -> str:
    """
    Convert the structured mortgage key_fields JSON into a compact markdown summary.

    Supports the extended schema with other_numeric_fields and other_text_fields.
    """
    if not isinstance(data, dict) or not data:
        return ""

    lines: List[str] = []
    lines.append("Structured mortgage key fields (pre-extracted JSON summary):")
    lines.append("")

    # document_type
    dt = data.get("document_type")
    if dt:
        lines.append(f"- **document_type**: `{dt}`")

    # loan_terms
    lt = data.get("loan_terms") or {}
    if lt:
        lines.append("- **loan_terms:**")
        for k in ("loan_amount", "interest_rate_percent", "term_months", "rate_type"):
            v = lt.get(k)
            if v is not None:
                lines.append(f"  - {k}: {v}")

    # payment_info
    pay = data.get("payment_info") or {}
    if pay:
        lines.append("- **payment_info:**")
        for k in (
            "estimated_monthly_payment",
            "principal_interest",
            "escrow",
            "first_payment_date",
        ):
            v = pay.get(k)
            if v is not None:
                lines.append(f"  - {k}: {v}")

    # parties
    parties = data.get("parties") or {}
    if parties:
        lines.append("- **parties:**")
        borrowers = parties.get("borrower_names") or []
        lender = parties.get("lender_name")
        other = parties.get("other_parties") or []
        if borrowers:
            lines.append(f"  - borrower_names: {', '.join(borrowers)}")
        if lender:
            lines.append(f"  - lender_name: {lender}")
        if other:
            lines.append(f"  - other_parties: {', '.join(other)}")

    # fee_items (just a few)
    fees = data.get("fee_items") or []
    if fees:
        lines.append(f"- **fee_items:** {len(fees)} items (showing up to 8)")
        for fee in fees[:8]:
            label = fee.get("label")
            amount = fee.get("amount")
            cat = fee.get("category")
            parts = []
            if label:
                parts.append(label)
            if amount is not None:
                parts.append(f"amount={amount}")
            if cat:
                parts.append(f"category={cat}")
            if parts:
                lines.append("  - " + ", ".join(parts))

    # risk_notes (sample)
    risk = data.get("risk_notes") or []
    if risk:
        lines.append("- **risk_notes (sample):**")
        for note in risk[:5]:
            lines.append(f"  - {note}")

    # other_numeric_fields
    onf = data.get("other_numeric_fields") or []
    if onf:
        lines.append("- **other_numeric_fields (sample):**")
        for item in onf[:8]:
            section = item.get("section")
            label = item.get("label")
            value = item.get("value")
            unit = item.get("unit")
            notes = item.get("notes")
            parts = []
            if section:
                parts.append(f"section={section}")
            if label:
                parts.append(f"label={label}")
            if value is not None:
                parts.append(f"value={value}")
            if unit:
                parts.append(f"unit={unit}")
            if notes:
                parts.append(f"notes={notes}")
            if parts:
                lines.append("  - " + ", ".join(parts))

    # other_text_fields
    otf = data.get("other_text_fields") or []
    if otf:
        lines.append("- **other_text_fields (sample):**")
        for item in otf[:8]:
            section = item.get("section")
            label = item.get("label")
            value = item.get("value")
            parts = []
            if section:
                parts.append(f"section={section}")
            if label:
                parts.append(f"label={label}")
            if value:
                parts.append(f"value={value}")
            if parts:
                lines.append("  - " + ", ".join(parts))

    return "\n".join(lines)


def _load_key_fields_snippet(index_dir: str) -> str:
    """
    Backwards-compatible helper used by some agents.

    Loads or builds the structured key_fields JSON and returns a markdown summary.
    """
    data = _load_or_build_structured_json(index_dir=index_dir)
    return _structured_json_to_markdown(data)


# ----------------------------------------------------------------------
# Helper: patch numeric markdown using JSON so obvious fields aren't "missing"
# ----------------------------------------------------------------------


def _patch_numeric_markdown_with_key_fields(
    markdown: str, structured_json: Dict[str, Any]
) -> str:
    """
    If the numeric summary says something like 'Not clearly specified in context'
    or 'missing' for a fee that actually exists in the structured JSON, replace
    that line with the JSON amount.

    This is a light post-processing 'safety net' on top of the prompt rules.
    """
    if not structured_json:
        return markdown

    fee_items = structured_json.get("fee_items") or []
    label_to_amount: Dict[str, float] = {}

    for fee in fee_items:
        label = (fee.get("label") or "").strip()
        amount = fee.get("amount")
        if label and isinstance(amount, (int, float)):
            label_to_amount[label.lower()] = amount

    if not label_to_amount:
        return markdown

    lines = markdown.splitlines()
    new_lines: List[str] = []

    for line in lines:
        lowered = line.lower()
        # Only try to fix lines that claim something is missing / not specified
        if (
            ("not clearly specified" in lowered)
            or ("not specified" in lowered)
            or ("missing" in lowered)
        ) and ":" in line:
            for label_lower, amount in label_to_amount.items():
                if label_lower in lowered:
                    # Replace everything after the first ':' with the correct amount
                    prefix, _sep, _rest = line.partition(":")
                    line = f"{prefix}: {amount}"
                    break

        new_lines.append(line)

    return "\n".join(new_lines)


# ----------------------------------------------------------------------
# 1. Q&A crew (wired to /agents/financial/crew-chat)
# ----------------------------------------------------------------------


def run_financial_crew(
    index_dir: str,
    question: str,
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    Main entry point for the CrewAI financial Q&A workflow.

    - Retrieves relevant chunks from the FAISS index.
    - Runs a 2-agent CrewAI workflow (analyst + reviewer).
    - Returns a dict with `answer` (final text) and `contexts` (raw chunk dicts).
    """
    contexts = retrieve_contexts(index_dir=index_dir, question=question, top_k=top_k)

    if not contexts:
        return {
            "answer": (
                "I couldn't find any relevant information in the indexed document "
                "for this question. Make sure the document is indexed and the "
                "question is actually related to that document."
            ),
            "contexts": [],
        }

    context_text = format_contexts_for_prompt(contexts)
    layout_snippet = _load_layout_snippet(index_dir)
    llm = make_llm()

    if layout_snippet:
        layout_text_for_prompt = (
            "\n\n---\nLayout view (blocks + bounding boxes):\n"
            f"{layout_snippet}\n"
            "---\n"
        )
    else:
        layout_text_for_prompt = ""

    analyst = Agent(
        role="Loan & Compliance Analyst",
        goal=(
            "Read loan worksheets and financial disclosure documents and identify all "
            "risk, compliance, and data-quality issues that are relevant to the user's question."
        ),
        backstory=(
            "You are an experienced mortgage risk analyst. You only answer using facts "
            "from the provided document context and you always reference the chunk ids you used."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    reviewer = Agent(
        role="Risk Review Lead",
        goal=(
            "Double-check the analyst's work and produce a clean, concise answer that a non-technical user "
            "can understand, with a short 'Evidence' section referencing relevant chunk ids."
        ),
        backstory=(
            "You are a senior reviewer who is careful about hallucinations. You never invent numbers or rules "
            "that are not explicitly supported by the context."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    analysis_task = Task(
        description=(
            "You are analyzing loan / fee / compliance documents.\n\n"
            f"User question:\n{question}\n\n"
            "You are given document context, split into numbered chunks. "
            "Use ONLY this context. If something isn't supported by the context, say so.\n\n"
            f"{context_text}\n"
            f"{layout_text_for_prompt}\n"
            "Step-by-step:\n"
            "1. Identify which chunks are relevant.\n"
            "2. Extract the important facts from those chunks.\n"
            "3. Draft a detailed analysis that answers the user's question.\n"
            "4. Explicitly list which chunk ids you used in a section called 'Evidence chunks'."
        ),
        expected_output=(
            "A detailed analysis with sections: 'Analysis' (paragraphs) and 'Evidence chunks' "
            "(bullet list of chunk ids and short notes)."
        ),
        agent=analyst,
    )

    review_task = Task(
        description=(
            "You are reviewing another agent's analysis of the same documents.\n\n"
            f"User question:\n{question}\n\n"
            "Document context (same chunks used by the analyst):\n"
            f"{context_text}\n"
            f"{layout_text_for_prompt}\n"
            "You will see the previous agent's analysis in the conversation history.\n\n"
            "Your job:\n"
            "1. Check the analysis against the context.\n"
            "2. Fix any mistakes or overconfident claims.\n"
            "3. Remove duplication and make the answer shorter but clear.\n"
            "4. Produce final output with two sections:\n"
            "   - 'Answer' (2–5 short paragraphs or bullet points)\n"
            "   - 'Evidence' (bullet list with chunk ids and short quotes/summaries)\n"
        ),
        expected_output="Final answer with 'Answer' and 'Evidence' sections.",
        agent=reviewer,
    )

    crew = Crew(
        agents=[analyst, reviewer],
        tasks=[analysis_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_result = crew.kickoff()

    if hasattr(crew_result, "output"):
        final_answer = crew_result.output
    else:
        final_answer = str(crew_result)

    return {
        "answer": final_answer,
        "contexts": contexts,
    }


# ----------------------------------------------------------------------
# 2. Full-document multi-agent analysis crew
# ----------------------------------------------------------------------


def run_document_analysis_crew(
    index_dir: str,
    top_k: int = 12,
    layout_blocks: List[dict] | None = None,  # for future flexibility
) -> Dict[str, Any]:
    """
    Multi-agent document analysis using CrewAI.

    - Uses FAISS + embeddings to get top-k chunks as shared context.
    - Also loads layout_blocks JSON (if present) and pre-extracted key_fields JSON.
    - Runs several specialized agents:
        * OverviewAgent
        * NumericExtractionAgent
        * ChecklistAgent
        * RiskAgent
        * CrossValidationAgent
    """
    question = (
        "Give a high-level overview of this loan / fee / disclosure document and "
        "all key fields relevant for risk/compliance."
    )

    contexts = retrieve_contexts(index_dir=index_dir, question=question, top_k=top_k)

    if not contexts:
        return {
            "doc_id": Path(index_dir).name,
            "results": [],
        }

    # Raw text context from FAISS
    context_text = format_contexts_for_prompt(contexts)

    # Layout snapshot 
    layout_md = _load_layout_snippet(index_dir)
    if layout_md:
        base_context = (
            context_text
            + "\n\n---\n\n"
            + "The following additional information comes from a layout-based parser "
            + "(blocks with bounding boxes):\n\n"
            + layout_md
        )
    else:
        base_context = context_text

    # build or load structured JSON for this doc_id
    doc_id = Path(index_dir).name
    structured_json = _load_or_build_structured_json(index_dir=index_dir)
    structured_md = _structured_json_to_markdown(structured_json)

    if structured_md:
        combined_context = (
            base_context
            + "\n\n---\n\n"
            + "Below is a structured JSON view of key mortgage fields extracted from "
            + "the same document. Treat this JSON as your PRIMARY source of numeric "
            + "truth. When values conflict, prefer the JSON but feel free to flag the "
            + "conflict as a risk. If a field is missing from the JSON, you may fall "
            + "back to the raw text above.\n\n"
            + structured_md
        )
    else:
        combined_context = base_context

    llm = make_llm()

    # --- Agents -------------------------------------------------------

    overview_agent = Agent(
        role="OverviewAgent",
        goal="Provide a clear, non-technical overview of the loan / disclosure document.",
        backstory=(
            "You summarize complex mortgage and disclosure forms for non-expert users. "
            "You highlight purpose, parties involved, and overall structure."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    numeric_agent = Agent(
        role="NumericExtractionAgent",
        goal="Extract key numeric fields and present them in a structured way.",
        backstory=(
            "You are very good at reading tables and numbers. You pull out loan amount, "
            "interest rate, term, major fee amounts, and monthly payments from the context."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    checklist_agent = Agent(
        role="ChecklistAgent",
        goal="Produce a checklist of items a reviewer should verify in this document.",
        backstory=(
            "You think like a mortgage underwriter who prepares checklists for junior staff. "
            "You turn the document into actionable review items."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    risk_agent = Agent(
        role="RiskAgent",
        goal="Identify risk, anomalies, or anything that ‘looks off’ in the document.",
        backstory=(
            "You are a cautious risk analyst who flags unusual fees, missing information, "
            "or anything that might require a second look."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    cross_agent = Agent(
        role="CrossValidationAgent",
        goal="Cross-check numbers and statements for internal consistency.",
        backstory=(
            "You carefully cross-validate totals, component sums, and narrative descriptions "
            "against the numeric tables to see if the story is consistent."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # --- Tasks --------------------------------------------------------

    overview_task = Task(
        description=(
            "You are given the following document context (loan / fees / disclosure).\n\n"
            f"{combined_context}\n\n"
            "Write ONLY the overview. Do NOT repeat these instructions and do NOT start "
            "with phrases like 'Okay, let's...' or 'Begin'. Start directly with the "
            "markdown content.\n\n"
            "Produce a concise but complete overview including:\n"
            "- What kind of document this is.\n"
            "- Who the borrowers and lender appear to be.\n"
            "- High-level description of the loan (type, term, interest rate if visible).\n"
            "- A short paragraph explaining what someone should understand from this document."
        ),
        expected_output="A markdown section titled 'Overview' with 2–4 short paragraphs.",
        agent=overview_agent,
    )

    numeric_task = Task(
        description=(
            "Using ONLY the context below, extract key numeric fields.\n\n"
            f"{combined_context}\n\n"
            "You have TWO sources of information in this context:\n"
            "1) The raw text chunks from the PDF\n"
            "2) The structured JSON summary of key mortgage fields\n\n"
            "RULES:\n"
            "- If the structured JSON already gives a non-null value for a field "
            "(for example, mortgage recording charge, underwriting fee, or document "
            "preparation fee), you MUST use that value and MUST NOT say it is missing.\n"
            "- If the JSON does NOT contain a field but you can clearly see it in the "
            "raw text, you may use the raw text value.\n"
            "- Only say 'Not clearly specified in context' if BOTH (a) the raw text "
            "AND (b) the structured JSON do not contain a usable value.\n\n"
            "Write ONLY the numeric summary. Do NOT repeat these instructions and do NOT "
            "start with phrases like 'Okay, let's...' or 'Begin'. Start directly with "
            "the markdown bullet lists.\n\n"
            "Return your answer as markdown under headings:\n"
            "- Loan terms (loan amount, interest rate, term)\n"
            "- Monthly payment details (if available)\n"
            "- Major fees (origination, appraisal, title, escrow, etc.) with amounts\n"
        ),
        expected_output="Markdown with bullet lists of key numeric fields and amounts.",
        agent=numeric_agent,
    )

    checklist_task = Task(
        description=(
            "Based on the context below, generate ONLY a checklist for a human reviewer.\n\n"
            f"{combined_context}\n\n"
            "Do NOT re-write an overview of the document and do NOT re-list all fees. "
            "Focus on actionable 'things to verify'. Do NOT repeat these instructions and "
            "do NOT start with phrases like 'Okay, let's...' or 'Begin'. Start directly "
            "with the checklist markdown.\n\n"
            "Return your answer as markdown with:\n"
            "- A very short intro sentence (one line)\n"
            "- A bullet list of concrete items to verify (documents, signatures, numbers, "
            "missing fields, unclear fees, etc.)."
        ),
        expected_output="Markdown checklist section with focused bullet points.",
        agent=checklist_agent,
    )

    risk_task = Task(
        description=(
            "Analyze the context below for risk flags or unusual issues.\n\n"
            f"{combined_context}\n\n"
            "You have both the raw text and a structured JSON summary. "
            "If the JSON has a non-null amount for a fee, you MUST NOT claim "
            "that fee is 'missing' or 'not specified'. You may still flag it as "
            "suspicious if the amount seems unreasonable.\n"
            "Only say a field is missing when it is absent from both the JSON "
            "AND the raw text.\n\n"
            "Do NOT re-write the numeric summary or checklist. Your job is to highlight "
            "the *riskiest* points only. Do NOT repeat these instructions and do NOT start "
            "with phrases like 'Okay, let's...' or 'Begin'. Start directly with the "
            "markdown content.\n\n"
            "Return your answer as markdown with:\n"
            "- 'Key risks' (bullet list)\n"
            "- 'Why it matters' (short explanation per risk).\n"
            "If nothing looks risky, explain briefly why the document appears normal."
        ),
        expected_output="Markdown section describing potential risks and why they matter.",
        agent=risk_agent,
    )

    cross_task = Task(
        description=(
            "Cross-check numbers and descriptions in the following context.\n\n"
            f"{combined_context}\n\n"
            "Do NOT re-write the overview or numeric summary. Focus ONLY on consistency "
            "checks. Do NOT repeat these instructions and do NOT start with phrases like "
            "'Okay, let's...' or 'Begin'. Start directly with the findings.\n\n"
            "Look for:\n"
            "- Totals that don't match their components\n"
            "- Inconsistent descriptions (e.g., term says 15 years vs 360 months)\n"
            "- Any mismatch between narrative text and numeric values.\n\n"
            "Return your answer as markdown with a 'Cross-validation findings' bullet list."
        ),
        expected_output="Markdown bullet list of consistency checks and any mismatches found.",
        agent=cross_agent,
    )

    crew = Crew(
        agents=[
            overview_agent,
            numeric_agent,
            checklist_agent,
            risk_agent,
            cross_agent,
        ],
        tasks=[overview_task, numeric_task, checklist_task, risk_task, cross_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        crew_result = crew.kickoff()
    except Exception as e:
        # Log to backend console
        print(f"[run_document_analysis_crew] CrewAI error: {e!r}")

        # Fallback: return a single “overview” result so the API still works
        fallback_markdown = (
            "## Analysis failed\n\n"
            "The multi-agent CrewAI analysis encountered an error:\n\n"
            f"`{e!r}`\n\n"
            "However, the document context was loaded correctly from the vector store. "
            "Check the backend logs and your CrewAI / LLM configuration."
        )

        return {
            "doc_id": doc_id,
            "results": [
                {
                    "agent": "OverviewAgent",
                    "title": "Analysis error (fallback)",
                    "markdown": fallback_markdown,
                    "contexts": contexts,
                }
            ],
        }

    # Normal path if crew_result is OK
    task_outputs = getattr(crew_result, "tasks_output", None)
    results: List[Dict[str, Any]] = []
    task_names = [
        ("OverviewAgent", "Overview"),
        ("NumericExtractionAgent", "Key numbers"),
        ("ChecklistAgent", "Checklist"),
        ("RiskAgent", "Risk analysis"),
        ("CrossValidationAgent", "Cross-validation"),
    ]

    if isinstance(task_outputs, list) and task_outputs:
        for (agent_name, title), to in zip(task_names, task_outputs):
            content = getattr(to, "raw_output", None) or getattr(
                to, "output", None
            ) or str(to)
            md = str(content)

            # NEW: enforce JSON consistency for the numeric summary
            if agent_name == "NumericExtractionAgent":
                md = _patch_numeric_markdown_with_key_fields(md, structured_json)

            results.append(
                {
                    "agent": agent_name,
                    "title": title,
                    "markdown": md,
                    "contexts": contexts,
                }
            )
    else:
        results.append(
            {
                "agent": "OverviewAgent",
                "title": "Full analysis",
                "markdown": str(getattr(crew_result, "output", crew_result)),
                "contexts": contexts,
            }
        )

    return {
        "doc_id": doc_id,
        "results": results,
    }

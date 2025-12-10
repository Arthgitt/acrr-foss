from app.rag.embeddings import embed_texts
from app.rag.vector_store import FaissIndex
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import requests  # for calling backend endpoints

from src.extract.pdf_text import extract_text_from_pdf_bytes, join_pages_to_single_text
from src.extract.chunking import make_char_chunks
from src.extract.layout_blocks import (
    extract_layout_blocks_from_pdf_bytes,
    find_key_values_by_keyword,  # NEW
)

# -------------------------------------------------------------------
# Backend URL (for RAG & CrewAI calls)
# -------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def call_native_rag(doc_id: str, question: str, top_k: int = 5) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/chat/rag"
    resp = requests.post(
        url,
        json={"doc_id": doc_id, "question": question, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_crew_chat(doc_id: str, question: str, top_k: int = 6) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/agents/financial/crew-chat"
    resp = requests.post(
        url,
        json={"doc_id": doc_id, "question": question, "top_k": top_k},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def call_analysis_crew(doc_id: str, top_k: int = 12) -> Dict[str, Any]:
    """
    Call FastAPI /agents/financial/analyze to run the full multi-agent crew.
    """
    url = f"{BACKEND_URL}/agents/financial/analyze"
    resp = requests.post(
        url,
        json={"doc_id": doc_id, "top_k": top_k},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def call_key_fields(doc_id: str, top_k: int = 30) -> Dict[str, Any]:
    """
    Call FastAPI /agents/financial/key-fields to run the mortgage key-fields extractor.
    """
    url = f"{BACKEND_URL}/agents/financial/key-fields"
    resp = requests.post(
        url,
        json={"doc_id": doc_id, "top_k": top_k},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def build_analysis_report_markdown(analysis: Dict[str, Any]) -> str:
    """
    Build a combined markdown report from the multi-agent analysis response.
    """
    if not analysis:
        return "# ACRR FOSS – Multi-agent Report\n\n_No analysis available._"

    doc_id = analysis.get("doc_id", "unknown_document")
    results = analysis.get("results", [])

    lines: List[str] = []
    lines.append("# ACRR FOSS – Multi-agent Document Report")
    lines.append("")
    lines.append(f"**Document ID:** `{doc_id}`")
    lines.append("")

    for block in results:
        title = block.get("title", block.get("agent", "Section"))
        lines.append(f"## {title}")
        lines.append("")
        lines.append(block.get("markdown", "_No content returned._"))
        lines.append("")

    return "\n".join(lines)


# Page config
st.set_page_config(
    page_title="ACRR FOSS – Financial PDF Inspector & Analyzer",
    layout="wide",
)

# ---- Session state setup ----
if "pages" not in st.session_state:
    st.session_state.pages: List[Dict[str, Any]] | None = None
if "combined_text" not in st.session_state:
    st.session_state.combined_text: str | None = None
if "filename" not in st.session_state:
    st.session_state.filename: str | None = None
if "chunks" not in st.session_state:
    st.session_state.chunks: List[Dict[str, Any]] | None = None
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id: str | None = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result: Dict[str, Any] | None = None
if "layout_blocks" not in st.session_state:
    st.session_state.layout_blocks: List[Dict[str, Any]] | None = None
# NEW: store UI results so toggles work
if "layout_matches" not in st.session_state:
    st.session_state.layout_matches: List[Dict[str, Any]] | None = None
if "semantic_results" not in st.session_state:
    st.session_state.semantic_results: List[Dict[str, Any]] | None = None


def show_sidebar() -> None:
    with st.sidebar:
        st.title("ACRR FOSS")
        st.caption("Automated Compliance & Risk Reporter – FOSS Edition")

        st.markdown("### 🧭 App Overview")
        st.write(
            "- **Step 1** – Upload a PDF & extract text/layout\n"
            "- **Step 2** – Build embeddings & FAISS index\n"
            "- **Step 3** – Ask questions / extract mortgage key fields\n"
            "- **Step 4** – Run multi-agent financial analysis\n"
            "- **Step 5** – Experimental chat interface"
        )

        st.markdown("### 💡 Tips")
        st.write(
            "- Works best on text-based PDFs (not only scanned images)\n"
            "- Chunks are simple character slices with overlap\n"
            "- Layout blocks show where each text snippet lives on the page\n"
            "- Use the **tabs** along the top to move between steps"
        )

        st.markdown("---")
        st.caption("Backend: FastAPI · FAISS · Local LLM via Ollama\nFrontend: Streamlit")


def show_page_summary(pages: List[Dict[str, Any]], filename: str) -> None:
    num_pages = len(pages)
    total_chars = sum(len(p["text"] or "") for p in pages)

    st.subheader("📄 PDF Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Pages", num_pages)
    col2.metric("Approx. characters", f"{total_chars:,}")
    col3.write(f"**File:** `{filename}`")

    if num_pages == 1:
        st.info(
            "ℹ️ This PDF has **only 1 page**, so "
            "**'Page by page'** and **'Single combined text'** will look the same."
        )


def show_page_text(pages: List[Dict[str, Any]]) -> None:
    st.subheader("🧾 Extracted Text (Page by Page)")
    for page in pages:
        with st.expander(f"Page {page['page_num']}", expanded=False):
            if page["text"].strip():
                st.text(page["text"])
            else:
                st.caption("No text found on this page (might be an image-only page).")


def show_chunks_debug(chunks: List[Dict[str, Any]]) -> None:
    st.subheader("🧩 Chunks (Debug View)")

    if not chunks:
        st.warning("No chunks generated (empty text).")
        return

    st.write(f"Total chunks: **{len(chunks)}**")

    table_rows = [
        {
            "Chunk ID": c["id"],
            "Start": c["start"],
            "End": c["end"],
            "Length": c["end"] - c["start"],
            "Preview": (c["text"][:80] + "…") if len(c["text"]) > 80 else c["text"],
        }
        for c in chunks
    ]
    st.dataframe(table_rows, use_container_width=True)

    st.markdown("### Inspect individual chunks")
    selected_id = st.number_input(
        "Chunk ID to view:",
        min_value=0,
        max_value=len(chunks) - 1,
        value=0,
        step=1,
    )

    selected_chunk = chunks[int(selected_id)]
    st.markdown(
        f"**Chunk {selected_chunk['id']}** "
        f"(chars {selected_chunk['start']}–{selected_chunk['end']}, "
        f"length {selected_chunk['end'] - selected_chunk['start']})"
    )
    st.text_area(
        "Chunk text",
        selected_chunk["text"],
        height=200,
    )


def show_layout_blocks_debug(blocks: List[Dict[str, Any]]) -> None:
    st.subheader("📐 Layout blocks (bounding boxes – debug view)")

    if not blocks:
        st.warning("No layout blocks extracted.")
        return

    st.write(f"Total blocks: **{len(blocks)}**")

    preview_rows = [
        {
            "Page": b["page_num"],
            "Block #": b.get("block_no", -1),
            "Type": b.get("block_type", "text"),
            "BBox": b.get("bbox"),
            "Preview": (b["text"][:80] + "…") if len(b["text"]) > 80 else b["text"],
        }
        for b in blocks
    ]
    st.dataframe(preview_rows, use_container_width=True)

    st.markdown("### Inspect a specific block")
    max_idx = len(blocks) - 1
    idx = st.number_input(
        "Block index",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1,
    )

    block = blocks[int(idx)]
    st.markdown(
        f"**Block {idx}** | Page {block['page_num']} | "
        f"Type: `{block.get('block_type', 'text')}` | "
        f"BBox: {block.get('bbox')}"
    )
    st.text_area("Block text", block["text"], height=200)


def main() -> None:
    show_sidebar()

    st.title("ACRR FOSS – Financial PDF Inspector & Analyzer")
    st.write(
        "This tool turns a **financial PDF** into a searchable, analyzable document.\n"
        "Use the steps below to go from raw PDF ➝ embeddings ➝ Q&A ➝ multi-agent analysis."
    )

    # ------------------------------
    # File upload + basic settings
    # ------------------------------
    uploaded_file = st.file_uploader(
        "📂 Upload a PDF file",
        type=["pdf"],
        help="Drag & drop a PDF here or click to browse.",
    )

    if uploaded_file is None:
        st.info("⬆️ Upload a PDF to unlock all steps.")
        return

    st.write("### File Info")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.write(f"**Name:** `{uploaded_file.name}`")
    info_col2.write(f"**Type:** `{uploaded_file.type}`")
    info_col3.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    st.session_state.filename = uploaded_file.name

    st.write("### Chunking Settings (for embeddings & debug views)")
    c1, c2 = st.columns(2)
    max_chars = c1.number_input(
        "Max characters per chunk",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
    )
    overlap = c2.number_input(
        "Overlap between chunks",
        min_value=0,
        max_value=1000,
        value=100,
        step=50,
    )

    if st.button("🚀 Extract Text & Layout"):
        pdf_bytes: bytes = uploaded_file.read()

        with st.spinner("Extracting text from PDF..."):
            pages = extract_text_from_pdf_bytes(pdf_bytes)
            combined_text = join_pages_to_single_text(pages)

        with st.spinner("Extracting layout blocks (bounding boxes)..."):
            layout_blocks = extract_layout_blocks_from_pdf_bytes(pdf_bytes)

        st.session_state.pages = pages
        st.session_state.combined_text = combined_text
        st.session_state.filename = uploaded_file.name
        st.session_state.layout_blocks = layout_blocks
        # clear downstream state because document changed
        st.session_state.semantic_results = None
        st.session_state.layout_matches = None

        st.success("✅ Extraction complete! Use the tabs below to explore and analyze.")

    # If we have extracted text, unlock the rest of the UI
    if st.session_state.pages is not None and st.session_state.combined_text is not None:
        pages = st.session_state.pages
        combined_text = st.session_state.combined_text
        filename = st.session_state.filename or uploaded_file.name

        # Pre-compute chunks for later tabs
        chunks = make_char_chunks(
            combined_text,
            max_chars=int(max_chars),
            overlap=int(overlap),
        )
        st.session_state.chunks = chunks

        show_page_summary(pages, filename)

        # ------------------------------
        # High-level navigation tabs
        # ------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "1️⃣ Extract & Inspect",
                "2️⃣ Index & Search",
                "3️⃣ Q&A & Key Fields",
                "4️⃣ Multi-Agent Analysis",
                "5️⃣ Chat (Experimental)",
            ]
        )

        # ==========================================================
        # TAB 1 – Extract & Inspect
        # ==========================================================
        with tab1:
            st.markdown("#### 1️⃣ Inspect extracted text, chunks, and layout")

            st.write("Use these views to **sanity check** the extraction quality before building embeddings.")

            st.write("##### View Options")
            view_mode = st.radio(
                "How do you want to view the extracted text?",
                options=[
                    "Page by page",
                    "Single combined text",
                    "Chunks (debug view)",
                    "Layout blocks (debug view)",
                ],
                index=0,
                horizontal=True,
                key="view_mode_radio",
            )

            if view_mode == "Page by page":
                show_page_text(pages)

            elif view_mode == "Single combined text":
                st.subheader("📚 All Pages Combined")
                show_combined = st.toggle(
                    "Show combined text (may be long)",
                    value=False,
                    key="show_combined_toggle",
                )
                if show_combined:
                    st.text_area(
                        "Combined text",
                        combined_text,
                        height=300,
                    )

            elif view_mode == "Chunks (debug view)":
                show_chunks_debug(chunks)

            else:
                layout_blocks = st.session_state.get("layout_blocks") or []
                show_layout_blocks_debug(layout_blocks)

            st.markdown("##### Download extracted text")
            st.download_button(
                label="💾 Download extracted text (.txt)",
                data=combined_text.encode("utf-8"),
                file_name=f"{filename.rsplit('.', 1)[0]}_extracted.txt",
                mime="text/plain",
            )

            # ---------- Layout-based key/value finder ----------
            st.markdown("---")
            with st.expander("🔬 Experimental: layout-based key/value finder", expanded=False):
                layout_blocks = st.session_state.get("layout_blocks") or []
                if not layout_blocks:
                    st.info("Run '🚀 Extract Text & Layout' first to populate layout blocks.")
                else:
                    keyword = st.text_input(
                        "Keyword / label to search for (case-insensitive)",
                        placeholder="Examples: 'Total Loan Amount', 'Hazard Insurance Premium'",
                        key="kv_keyword",
                    )

                    max_vertical = st.slider(
                        "Max vertical distance (same row tolerance, in PDF units)",
                        min_value=5,
                        max_value=60,
                        value=20,
                        step=5,
                        key="kv_max_vertical",
                    )

                    # Button updates session_state.layout_matches
                    if st.button("Find fields", key="kv_find"):
                        if not keyword.strip():
                            st.warning("Please enter a keyword or label.")
                            st.session_state.layout_matches = None
                        else:
                            matches = find_key_values_by_keyword(
                                layout_blocks,
                                keyword.strip(),
                                max_vertical_distance=float(max_vertical),
                            )
                            if not matches:
                                st.warning("No blocks found containing that keyword.")
                                st.session_state.layout_matches = None
                            else:
                                st.session_state.layout_matches = matches

                    matches = st.session_state.get("layout_matches")
                    if matches:
                        rows = []
                        for idx, m in enumerate(matches, start=1):
                            kb = m["keyword_block"]
                            vals = m["value_candidates"] or []
                            rows.append(
                                {
                                    "Match #": idx,
                                    "Page": kb["page_num"],
                                    "Block #": kb.get("block_no", -1),
                                    "Keyword snippet": (
                                        kb["text"][:80] + "…"
                                        if len(kb["text"]) > 80
                                        else kb["text"]
                                    ),
                                    "BBox": kb.get("bbox"),
                                    "Value candidates": ", ".join(vals)
                                    if vals
                                    else "(none)",
                                }
                            )

                        st.subheader("Matches (summary)")
                        st.dataframe(rows, use_container_width=True)

                        show_raw_matches = st.toggle(
                            "Show raw match details",
                            value=False,
                            key="show_raw_match_toggle",
                        )
                        if show_raw_matches:
                            st.markdown("#### Raw matches")
                            for idx, m in enumerate(matches, start=1):
                                kb = m["keyword_block"]
                                vals = m["value_candidates"] or []
                                st.markdown(
                                    f"**Match {idx}** — Page {kb['page_num']} | "
                                    f"Block #{kb.get('block_no', -1)} | BBox: {kb.get('bbox')}"
                                )
                                st.write(kb["text"])
                                st.caption(
                                    f"Value candidates: {', '.join(vals) if vals else '(none)'}"
                                )

        # ==========================================================
        # TAB 2 – Index & Search
        # ==========================================================
        with tab2:
            st.markdown("#### 2️⃣ Build embeddings & FAISS index, then test semantic search")

            st.write(
                "This step turns the PDF into **vector embeddings** and builds a **FAISS index** "
                "so later steps (Q&A, agents) can retrieve relevant chunks."
            )

            def build_index_for_current_pdf() -> None:
                chunks_list = st.session_state.get("chunks") or []
                if not chunks_list:
                    st.warning("No chunks available. Extract text and generate chunks first.")
                    return

                chunk_texts = [c["text"] for c in chunks_list]

                with st.spinner("Computing embeddings for all chunks..."):
                    embeddings = embed_texts(chunk_texts)

                pdf_name = st.session_state.get("filename") or "current_pdf"
                doc_id = Path(pdf_name).stem.replace(" ", "_")

                metadatas: List[Dict[str, Any]] = []
                for c in chunks_list:
                    metadatas.append(
                        {
                            "text": c["text"],
                            "source_id": doc_id,
                            "page_num": None,
                            "chunk_id": c["id"],
                            "start": c["start"],
                            "end": c["end"],
                        }
                    )

                index = FaissIndex.from_embeddings(embeddings, metadatas)

                st.session_state["faiss_index"] = index
                st.session_state["chunk_embeddings"] = embeddings
                st.session_state["current_doc_id"] = doc_id

                st.success(f"✅ Built FAISS index with {len(chunks_list)} chunks.")
                st.info(f"Current doc_id for this PDF: `{doc_id}`")

                out_dir = Path("data") / "vector_stores" / doc_id
                index.save(str(out_dir))
                st.caption(f"Index saved to `{out_dir}`")

                # Save layout blocks JSON for backend / CrewAI
                layout_blocks_saved = st.session_state.get("layout_blocks")
                if layout_blocks_saved:
                    layout_dir = Path("data") / "layout_blocks"
                    layout_dir.mkdir(parents=True, exist_ok=True)
                    layout_path = layout_dir / f"{doc_id}.json"

                    with layout_path.open("w", encoding="utf-8") as f:
                        json.dump(layout_blocks_saved, f, ensure_ascii=False, indent=2)

                    st.caption(f"Layout blocks JSON saved to `{layout_path}`")
                else:
                    st.caption("No layout blocks found to save.")

            if st.button("🔧 Build vector index for this PDF"):
                build_index_for_current_pdf()

            st.markdown("---")
            st.subheader("Test the index with a semantic search")

            query = st.text_input(
                "Type a question or phrase related to this PDF (e.g. 'interest rate fees')",
                key="semantic_query",
            )
            top_k = st.slider(
                "Number of chunks to show",
                min_value=1,
                max_value=10,
                value=3,
                key="semantic_top_k",
            )

            # Button updates session_state.semantic_results
            if st.button("Search in chunks", key="semantic_search_button") and query:
                if "faiss_index" not in st.session_state:
                    st.warning("Please build the index first.")
                    st.session_state.semantic_results = None
                else:
                    index: FaissIndex = st.session_state["faiss_index"]

                    with st.spinner("Embedding your query and searching..."):
                        query_emb = embed_texts([query])[0]
                        indices, distances = index.search(query_emb, top_k)

                    meta_list = index.metadata
                    results: List[Dict[str, Any]] = []
                    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
                        idx = int(idx)
                        if idx < 0 or idx >= len(meta_list):
                            continue
                        meta = meta_list[idx]
                        text = meta.get("text", "")
                        score = float(dist)
                        results.append(
                            {
                                "rank": rank,
                                "score": score,
                                "text": text,
                                "meta": meta,
                            }
                        )

                    st.session_state.semantic_results = results

            results = st.session_state.get("semantic_results")
            if results:
                st.write("### Top matches")
                for r in results:
                    preview_text = (
                        r["text"] if len(r["text"]) <= 400 else r["text"][:400] + "…"
                    )
                    with st.expander(
                        f"Result {r['rank']} — score: {r['score']:.3f}",
                        expanded=False,
                    ):
                        st.write(preview_text)
                        show_meta = st.toggle(
                            "Show metadata",
                            value=False,
                            key=f"search_meta_toggle_{r['rank']}",
                        )
                        if show_meta:
                            st.json(r["meta"])

        # ==========================================================
        # TAB 3 – Q&A & Key Fields
        # ==========================================================
        with tab3:
            st.markdown("#### 3️⃣ Ask questions & extract mortgage key fields")

            current_doc_id = st.session_state.get("current_doc_id")
            if not current_doc_id:
                st.info(
                    "Build a vector index in **Step 2** first. "
                    "That will set a doc_id and save the FAISS index for the backend."
                )
            else:
                st.write(f"Using doc_id: `{current_doc_id}`")

                qna_tab, keyfields_tab = st.tabs(["❓ Q&A (RAG / CrewAI)", "🏠 Mortgage key fields"])

                # ---------- Q&A ----------
                with qna_tab:
                    mode = st.radio(
                        "Answering mode",
                        ["Native RAG", "CrewAI multi-agent"],
                        horizontal=True,
                        key="rag_mode",
                    )

                    rag_question = st.text_input(
                        "Ask a question about this PDF",
                        placeholder="Example: What is the total loan amount?",
                        key="rag_question",
                    )

                    rag_top_k = st.slider(
                        "Number of chunks to send to the backend",
                        min_value=1,
                        max_value=10,
                        value=3,
                        key="rag_top_k",
                    )

                    if st.button("Get answer", key="rag_get_answer") and rag_question.strip():
                        try:
                            with st.spinner("Retrieving context and querying the backend..."):
                                if mode == "Native RAG":
                                    result = call_native_rag(
                                        current_doc_id, rag_question, rag_top_k
                                    )
                                else:
                                    result = call_crew_chat(
                                        current_doc_id, rag_question, rag_top_k
                                    )

                            st.subheader("Answer")
                            st.write(result.get("answer", ""))

                            with st.expander("Show retrieved chunks (contexts)", expanded=False):
                                for i, ctx in enumerate(result.get("contexts", []), start=1):
                                    score = ctx.get("score")
                                    score_str = (
                                        f"{score:.3f}"
                                        if isinstance(score, (int, float))
                                        else "N/A"
                                    )
                                    st.markdown(f"**Chunk {i} — score: {score_str}**")
                                    st.write(ctx.get("text", ""))
                                    st.caption(
                                        f"source_id={ctx.get('source_id')} | "
                                        f"page={ctx.get('page_num')} | "
                                        f"chunk_id={ctx.get('chunk_id')}"
                                    )
                        except Exception as e:
                            st.error(f"Backend error: {e}")

                # ---------- Mortgage key fields ----------
                with keyfields_tab:
                    st.write(f"Extracting key fields for doc_id: `{current_doc_id}`")

                    kf_top_k = st.slider(
                        "Number of chunks to use for key-field extraction",
                        min_value=10,
                        max_value=60,
                        value=30,
                        step=5,
                        key="kf_top_k",
                    )

                    if st.button("Run mortgage key-fields extractor", key="kf_run"):
                        try:
                            with st.spinner(
                                "Calling backend to extract structured mortgage fields…"
                            ):
                                kf_result = call_key_fields(current_doc_id, top_k=kf_top_k)
                            st.session_state["key_fields_result"] = kf_result
                            st.success("Key-fields JSON extracted.")
                        except Exception as e:
                            st.error(f"Key-fields backend error: {e}")

                    kf_result = st.session_state.get("key_fields_result")
                    if kf_result:
                        key_fields = kf_result.get("key_fields", {})
                        st.subheader("Key fields (summary)")
                        if isinstance(key_fields, dict) and key_fields:
                            keys_list = list(key_fields.keys())
                            st.write("Fields detected:")
                            st.write(", ".join(keys_list))
                        else:
                            st.write("No key fields returned.")

                        show_raw_kf = st.toggle(
                            "Show raw key-fields JSON",
                            value=False,
                            key="show_raw_keyfields_toggle",
                        )
                        if show_raw_kf:
                            st.json(key_fields)

        # ==========================================================
        # TAB 4 – Multi-agent document analysis
        # ==========================================================
        with tab4:
            st.markdown("#### 4️⃣ Multi-agent document analysis (CrewAI)")

            current_doc_id = st.session_state.get("current_doc_id")
            if not current_doc_id:
                st.info("Build a vector index first to enable multi-agent analysis.")
            else:
                st.write(f"Analyzing doc_id: `{current_doc_id}`")

                analysis_top_k = st.slider(
                    "Number of chunks to give to each agent",
                    min_value=4,
                    max_value=20,
                    value=12,
                    key="analysis_top_k",
                )

                if st.button("Run full analysis", key="run_full_analysis"):
                    try:
                        with st.spinner(
                            "Running Overview / Numeric / Checklist / Risk / Cross-check agents…"
                        ):
                            analysis_result = call_analysis_crew(
                                current_doc_id, analysis_top_k
                            )
                        st.session_state.analysis_result = analysis_result
                        st.success("Multi-agent analysis complete.")
                    except Exception as e:
                        st.error(f"Analysis backend error: {e}")

                analysis_result = st.session_state.get("analysis_result")
                if analysis_result:
                    blocks = analysis_result.get("results", [])
                    if not blocks:
                        st.warning("No analysis results returned from backend.")
                    else:
                        tab_labels = [
                            b.get("title", b.get("agent", "Section")) for b in blocks
                        ]
                        tabs = st.tabs(tab_labels)

                        for tab, block in zip(tabs, blocks):
                            with tab:
                                st.subheader(
                                    block.get("title", block.get("agent", "Section"))
                                )
                                st.markdown(
                                    block.get("markdown", "_No content returned._")
                                )

                                with st.expander("Show contexts used by this agent", expanded=False):
                                    for i, ctx in enumerate(
                                        block.get("contexts", []), start=1
                                    ):
                                        score = ctx.get("score")
                                        score_str = (
                                            f"{score:.3f}"
                                            if isinstance(score, (int, float))
                                            else "N/A"
                                        )
                                        st.markdown(
                                            f"**Chunk {i} — score: {score_str}**"
                                        )
                                        st.write(ctx.get("text", ""))

                        st.markdown("### Export combined report")
                        report_md = build_analysis_report_markdown(analysis_result)
                        default_name = f"{current_doc_id}_multi_agent_report.md"
                        st.download_button(
                            label="💾 Download multi-agent report (.md)",
                            data=report_md.encode("utf-8"),
                            file_name=default_name,
                            mime="text/markdown",
                        )

                        show_raw_analysis = st.toggle(
                            "Show raw analysis JSON (advanced)",
                            value=False,
                            key="show_raw_analysis_toggle",
                        )
                        if show_raw_analysis:
                            st.json(analysis_result)

        # ==========================================================
        # TAB 5 – Experimental chat
        # ==========================================================
        with tab5:
            st.markdown("#### 5️⃣ Experimental: Chat with this document")

            chat_doc_id = st.session_state.get("current_doc_id")
            if not chat_doc_id:
                st.info("Build a vector index first to enable chat.")
            else:
                st.write(f"Chat is using doc_id: `{chat_doc_id}`")

                chat_mode = st.radio(
                    "Chat mode",
                    ["Native RAG", "CrewAI multi-agent"],
                    horizontal=True,
                    key="chat_mode",
                )

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                user_msg = st.text_input(
                    "Type your message",
                    key="chat_user_input",
                    placeholder="Ask anything about this document…",
                )

                cols = st.columns([1, 1])
                send_clicked = cols[0].button("Send", key="chat_send")
                clear_clicked = cols[1].button("Clear chat", key="chat_clear")

                if clear_clicked:
                    st.session_state.chat_history = []

                if send_clicked:
                    if user_msg.strip():
                        st.session_state.chat_history.append(
                            {"role": "user", "content": user_msg}
                        )
                        try:
                            if chat_mode == "Native RAG":
                                result = call_native_rag(chat_doc_id, user_msg, top_k=5)
                            else:
                                result = call_crew_chat(chat_doc_id, user_msg, top_k=5)

                            reply = result.get("answer", "")
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": reply}
                            )
                        except Exception as e:
                            st.error(f"Chat backend error: {e}")
                    else:
                        st.warning("Please type a message first.")

                st.markdown("##### Conversation")
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['content']}")


if __name__ == "__main__":
    main()

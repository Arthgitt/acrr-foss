from app.rag.embeddings import embed_texts
from app.rag.vector_store import FaissIndex
from app.rag.rag_pipeline import RAGPipeline
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import requests  # NEW: for calling backend agents endpoint

from src.extract.pdf_text import extract_text_from_pdf_bytes, join_pages_to_single_text
from src.extract.chunking import make_char_chunks

# -------------------------------------------------------------------
# Backend URL (for agents + any other API calls)
# -------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# Page config
st.set_page_config(
    page_title="ACRR FOSS – PDF Upload, Text Extraction & Chunking",
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
if "current_doc_id" not in st.session_state:  # NEW: used by agents
    st.session_state.current_doc_id: str | None = None
if "agents_result" not in st.session_state:  # NEW
    st.session_state.agents_result: Dict[str, Any] | None = None


def show_sidebar() -> None:
    """Sidebar with info + instructions."""
    with st.sidebar:
        st.title("ACRR FOSS")
        st.caption("Automated Compliance & Risk Reporter – FOSS Edition")

        st.markdown("### 🔎 What this page does")
        st.write(
            "- Upload a PDF\n"
            "- Extract plain text using **PyMuPDF** (open-source)\n"
            "- View text page-by-page or combined\n"
            "- View **chunks** (for embeddings later)\n"
            "- Download the extracted text as `.txt`"
        )

        st.markdown("### 💡 Tips")
        st.write(
            "- Works best on text-based PDFs (not only scanned images)\n"
            "- For a **1-page PDF**, page and combined views look the same\n"
            "- Chunks are simple character slices with overlap"
        )

        st.markdown("---")
        st.markdown("Step 2 + 2.5 complete ✅\n\nNext: embeddings + FAISS.")


def show_page_summary(pages: List[Dict[str, Any]], filename: str) -> None:
    """Display summary about the PDF."""
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
    """Show text page-by-page in expandable sections."""
    st.subheader("🧾 Extracted Text (Page by Page)")

    for page in pages:
        with st.expander(f"Page {page['page_num']}"):
            if page["text"].strip():
                st.text(page["text"])
            else:
                st.caption("No text found on this page (might be an image-only page).")


def show_chunks_debug(chunks: List[Dict[str, Any]]) -> None:
    """Show a simple debug view of chunks."""
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
    st.text(selected_chunk["text"])


# -------------------------------------------------------------------
# Helper: call backend agents endpoint
# -------------------------------------------------------------------
def run_agents_on_doc(doc_id: str) -> Dict[str, Any]:
    """
    Call the FastAPI /api/agents/analyze/{doc_id} endpoint for the given doc_id.
    """
    url = f"{BACKEND_URL}/api/agents/analyze/{doc_id}"
    resp = requests.post(url)
    resp.raise_for_status()
    return resp.json()

def build_agent_report_markdown(agents_result: Dict[str, Any]) -> str:
    """
    Build a simple Markdown report from the agents_result payload.
    """
    if not agents_result:
        return "# ACRR FOSS – Agent Report\n\n_No agent results available._"

    doc_id = agents_result.get("doc_id", "unknown_document")
    results = agents_result.get("results", [])

    lines: List[str] = []
    lines.append(f"# ACRR FOSS – Compliance Report")
    lines.append("")
    lines.append(f"**Document ID:** `{doc_id}`")
    lines.append("")

    # Add a section per agent
    for agent_block in results:
        title = agent_block.get("title", agent_block.get("agent", "Section"))
        agent_name = agent_block.get("agent", "").lower()

        lines.append(f"## {title}")
        lines.append("")

        if agent_name == "overview":
            lines.append(agent_block.get("summary", "_No summary returned._"))
        elif agent_name == "risk":
            lines.append(agent_block.get("risks", "_No risk analysis returned._"))
        elif agent_name == "checklist":
            lines.append(agent_block.get("checklist", "_No checklist returned._"))
        else:
            # fallback: dump JSON-ish
            lines.append("```json")
            lines.append(str(agent_block))
            lines.append("```")

        lines.append("")  # blank line after each section

    return "\n".join(lines)


def main() -> None:
    show_sidebar()

    st.title("ACRR FOSS – PDF Upload, Text Extraction & Chunking")
    st.write(
        "Upload a **financial PDF** (or any PDF), extract plain text with **PyMuPDF**, "
        "and view a simple **chunking debug view** to prepare for embeddings."
    )

    # ---- Upload widget ----
    uploaded_file = st.file_uploader(
        "📂 Upload a PDF file",
        type=["pdf"],
        help="Drag & drop a PDF here or click to browse.",
    )

    if uploaded_file is None:
        st.info("⬆️ Please upload a PDF to begin.")
        return

    # File info
    st.write("### File Info")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.write(f"**Name:** `{uploaded_file.name}`")
    info_col2.write(f"**Type:** `{uploaded_file.type}`")
    info_col3.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    # keep filename in session for saving index
    st.session_state.filename = uploaded_file.name

    # ---- Chunking settings (used later) ----
    st.write("### Chunking Settings (for debug view)")
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

    # ---- Extract button: fills session_state ----
    if st.button("🚀 Extract Text"):
        pdf_bytes: bytes = uploaded_file.read()

        with st.spinner("Extracting text from PDF..."):
            pages = extract_text_from_pdf_bytes(pdf_bytes)
            combined_text = join_pages_to_single_text(pages)

        # Store in session_state so it survives widget changes
        st.session_state.pages = pages
        st.session_state.combined_text = combined_text
        st.session_state.filename = uploaded_file.name

        st.success("✅ Extraction complete!")

    # ---- If we have extracted data, show views ----
    if st.session_state.pages is not None and st.session_state.combined_text is not None:
        pages = st.session_state.pages
        combined_text = st.session_state.combined_text
        filename = st.session_state.filename or uploaded_file.name

        show_page_summary(pages, filename)

        # Build chunks every run using current settings
        chunks = make_char_chunks(
            combined_text,
            max_chars=int(max_chars),
            overlap=int(overlap),
        )

        # ✅ store chunks in session_state for Step 3 (embeddings + FAISS)
        st.session_state.chunks = chunks

        st.write("### View Options")
        view_mode = st.radio(
            "How do you want to view the extracted text?",
            options=["Page by page", "Single combined text", "Chunks (debug view)"],
            index=0,
            horizontal=True,
        )

        if view_mode == "Page by page":
            show_page_text(pages)
        elif view_mode == "Single combined text":
            st.subheader("📚 All Pages Combined")
            st.text(combined_text)
        else:
            show_chunks_debug(chunks)

        # Download
        st.download_button(
            label="💾 Download extracted text (.txt)",
            data=combined_text.encode("utf-8"),
            file_name=f"{filename.rsplit('.', 1)[0]}_extracted.txt",
            mime="text/plain",
        )

        # ------------------------------------------------------------------
        # STEP 3: EMBEDDINGS + FAISS INDEX
        # ------------------------------------------------------------------
        st.markdown("---")
        st.header("Step 3: Embeddings + FAISS index")

        def build_index_for_current_pdf() -> None:
            """Compute embeddings for chunks and build a FAISS index."""
            chunks_list = st.session_state.get("chunks") or []
            if not chunks_list:
                st.warning("No chunks available. Extract text and generate chunks first.")
                return

            # Only the text from each chunk
            chunk_texts = [c["text"] for c in chunks_list]

            with st.spinner("Computing embeddings for all chunks..."):
                embeddings = embed_texts(chunk_texts)  # (num_chunks, dim)

            # Build rich metadata for each chunk
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

            # Create FAISS index using the new wrapper
            index = FaissIndex.from_embeddings(embeddings, metadatas)

            # keep in memory
            st.session_state["faiss_index"] = index
            st.session_state["chunk_embeddings"] = embeddings

            # store doc_id for later (agents)
            st.session_state["current_doc_id"] = doc_id

            st.success(f"Built FAISS index with {len(chunks_list)} chunks.")
            st.info(f"Current doc_id for this PDF: `{doc_id}`")

            # Save to disk for persistence:
            # data/vector_stores/<doc_id>/{index.faiss, meta.json}
            out_dir = Path("data") / "vector_stores" / doc_id
            index.save(str(out_dir))
            st.info(f"Index saved to {out_dir}")

        # Button: build the index
        if st.button("Build vector index for this PDF"):
            build_index_for_current_pdf()

        # --------------------------------------------------------------
        # STEP 3.5: Simple semantic search (no LLM)
        # --------------------------------------------------------------
        st.subheader("Test the index with a semantic search")

        query = st.text_input(
            "Type a question or phrase related to this PDF (e.g. 'interest rate fees')"
        )
        top_k = st.slider(
            "Number of chunks to show",
            min_value=1,
            max_value=10,
            value=3,
        )

        if st.button("Search in chunks") and query:
            if "faiss_index" not in st.session_state:
                st.warning("Please build the index first.")
            else:
                index: FaissIndex = st.session_state["faiss_index"]

                with st.spinner("Embedding your query and searching..."):
                    query_emb = embed_texts([query])[0]  # (dim,)
                    indices, distances = index.search(query_emb, top_k)

                meta_list = index.metadata
                st.write("### Top matches")
                for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
                    idx = int(idx)
                    if idx < 0 or idx >= len(meta_list):
                        continue
                    meta = meta_list[idx]
                    text = meta.get("text", "")
                    score = float(dist)

                    st.markdown(f"**Result {rank} — score: {score:.3f}**")
                    st.write(text)
                    st.caption(f"Metadata: {meta}")

        # --------------------------------------------------------------
        # STEP 4: RAG + LOCAL LLM (Ollama) — with prompt debug
        # --------------------------------------------------------------
        st.markdown("---")
        st.header("Step 4: Ask questions with RAG + local LLM")

        rag_question = st.text_input(
            "Ask a question about this PDF (RAG)",
            placeholder="Example: What is the total loan amount?",
            key="rag_question",
        )

        ollama_model = st.text_input(
            "Ollama model name",
            value="gemma3:4b",
            help="Make sure `ollama serve` is running and you've done `ollama pull gemma3:4b` once.",
            key="ollama_model",
        )

        rag_top_k = st.slider(
            "Number of chunks to send to the LLM",
            min_value=1,
            max_value=10,
            value=3,
            key="rag_top_k",
        )

        if st.button("Get RAG answer") and rag_question.strip():
            if "faiss_index" not in st.session_state:
                st.warning("Please build the index first.")
            else:
                index: FaissIndex = st.session_state["faiss_index"]

                rag = RAGPipeline(
                    index=index,
                    model_name=ollama_model,
                    #ollama_url=ollama_url,
                    #max_context_chunks=rag_top_k,
                )

                try:
                    with st.spinner("Retrieving context and querying the LLM..."):
                        rag_result = rag.query(question=rag_question, top_k=rag_top_k)

                    st.subheader("Answer")
                    st.write(rag_result.get("answer", ""))

                    # 🔍 Debug expander: full LLM prompt
                    with st.expander("🔍 Show full LLM prompt (debug)"):
                        prompt_text = rag_result.get("prompt", "")
                        if prompt_text:
                            st.text_area(
                                "Prompt sent to LLM",
                                prompt_text,
                                height=300,
                            )
                        else:
                            st.caption("No prompt returned from RAGPipeline.")

                    # Show retrieved chunks (context)
                    with st.expander("Show retrieved chunks (context)"):
                        for i, ctx in enumerate(rag_result.get("contexts", []), start=1):
                            if isinstance(ctx, dict):
                                text = ctx.get("text", "")
                                meta = {
                                    "source_id": ctx.get("source_id"),
                                    "page_num": ctx.get("page_num"),
                                }
                                score = ctx.get("score")
                            elif isinstance(ctx, tuple) and len(ctx) == 3:
                                text, meta, score = ctx
                            else:
                                text, meta, score = str(ctx), {}, None

                            score_str = (
                                f"{score:.3f}"
                                if isinstance(score, (int, float))
                                else "N/A"
                            )
                            st.markdown(f"**Chunk {i} — score: {score_str}**")
                            st.write(text)
                            if meta:
                                st.caption(f"Metadata: {meta}")
                except Exception as e:
                    st.error(f"RAG error: {e}")

        # --------------------------------------------------------------
        # STEP 6: Agents – Overview, Risk, Checklist
        # --------------------------------------------------------------
        # st.markdown("---")
        # st.header("Step 6: Run analysis agents on this document")

        # current_doc_id = st.session_state.get("current_doc_id")

        # if not current_doc_id:
        #     st.info(
        #         "Build a vector index for this PDF first. "
        #         "That will set a doc_id and save the FAISS index."
        #     )
        # else:
        #     st.write(f"**Current doc_id:** `{current_doc_id}`")

        #     if st.button("Run agents (overview, risk, checklist)"):
        #         try:
        #             with st.spinner("Running agents via backend…"):
        #                 agents_result = run_agents_on_doc(current_doc_id)

        #             st.success("Agents finished.")
        #             results = agents_result.get("results", [])

        #             if not results:
        #                 st.warning("No agent results returned.")
        #             else:
        #                 tab_labels = [
        #                     r.get("title", r.get("agent", "Agent")) for r in results
        #                 ]
        #                 tabs = st.tabs(tab_labels)

        #                 for tab, agent_result in zip(tabs, results):
        #                     with tab:
        #                         st.subheader(
        #                             agent_result.get(
        #                                 "title", agent_result.get("agent", "Agent")
        #                             )
        #                         )

        #                         agent_name = agent_result.get("agent")

        #                         if agent_name == "overview":
        #                             st.markdown(
        #                                 agent_result.get(
        #                                     "summary", "_No summary returned._"
        #                                 )
        #                             )
        #                         elif agent_name == "risk":
        #                             st.markdown(
        #                                 agent_result.get(
        #                                     "risks", "_No risk analysis returned._"
        #                                 )
        #                             )
        #                         elif agent_name == "checklist":
        #                             st.markdown(
        #                                 agent_result.get(
        #                                     "checklist", "_No checklist returned._"
        #                                 )
        #                             )
        #                         else:
        #                             # fallback for any future agents
        #                             st.json(agent_result)

        #                         # Show RAG source chunks used by this agent
        #                         with st.expander("Show RAG source chunks (contexts)"):
        #                             for i, ctx in enumerate(
        #                                 agent_result.get("contexts", []), start=1
        #                             ):
        #                                 score = ctx.get("score")
        #                                 if score is not None:
        #                                     st.markdown(
        #                                         f"**Chunk {i} — score {score:.3f}**"
        #                                     )
        #                                 else:
        #                                     st.markdown(f"**Chunk {i}**")
        #                                 st.code(ctx.get("text", ""))
        #                                 st.markdown("---")

        #         except Exception as e:
        #             st.error(f"An error occurred while running agents: {e}")
        # --------------------------------------------------------------
        # STEP 6: Agents – Overview, Risk, Checklist
        # --------------------------------------------------------------
        st.markdown("---")
        st.header("Step 6: Run analysis agents on this document")

        current_doc_id = st.session_state.get("current_doc_id")

        if not current_doc_id:
            st.info(
                "Build a vector index for this PDF first. "
                "That will set a doc_id and save the FAISS index."
            )
        else:
            st.write(f"**Current doc_id:** `{current_doc_id}`")

            if st.button("Run agents (overview, risk, checklist)"):
                try:
                    with st.spinner("Running agents via backend…"):
                        agents_result = run_agents_on_doc(current_doc_id)

                    # ✅ remember result for download/export
                    st.session_state["agents_result"] = agents_result

                    st.success("Agents finished.")
                except Exception as e:
                    st.error(f"An error occurred while running agents: {e}")

            # Always read the latest result from session_state
            agents_result = st.session_state.get("agents_result")

            if agents_result:
                results = agents_result.get("results", [])

                if not results:
                    st.warning("No agent results returned.")
                else:
                    tab_labels = [
                        r.get("title", r.get("agent", "Agent")) for r in results
                    ]
                    tabs = st.tabs(tab_labels)

                    for tab, agent_result in zip(tabs, results):
                        with tab:
                            st.subheader(
                                agent_result.get("title", agent_result.get("agent", "Agent"))
                            )

                            agent_name = agent_result.get("agent")

                            if agent_name == "overview":
                                st.markdown(
                                    agent_result.get("summary", "_No summary returned._")
                                )
                            elif agent_name == "risk":
                                st.markdown(
                                    agent_result.get("risks", "_No risk analysis returned._")
                                )
                            elif agent_name == "checklist":
                                st.markdown(
                                    agent_result.get(
                                        "checklist", "_No checklist returned._"
                                    )
                                )
                            else:
                                st.json(agent_result)

                            # Show RAG source chunks used by this agent
                            with st.expander("Show RAG source chunks (contexts)"):
                                for i, ctx in enumerate(
                                    agent_result.get("contexts", []), start=1
                                ):
                                    score = ctx.get("score")
                                    if score is not None:
                                        st.markdown(f"**Chunk {i} — score {score:.3f}**")
                                    else:
                                        st.markdown(f"**Chunk {i}**")
                                    st.code(ctx.get("text", ""))
                                    st.markdown("---")

                # ✅ Download button for the full report
                st.markdown("### Export report")
                report_md = build_agent_report_markdown(agents_result)
                default_name = f"{current_doc_id}_compliance_report.md"

                st.download_button(
                    label="💾 Download agents report (.md)",
                    data=report_md.encode("utf-8"),
                    file_name=default_name,
                    mime="text/markdown",
                )
            else:
                st.caption("Run the agents to see results and enable report download.")



if __name__ == "__main__":
    main()

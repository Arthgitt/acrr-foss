from __future__ import annotations

from typing import List, Dict, Any, Optional
import textwrap
import os

import numpy as np
import requests

from app.rag.embeddings import embed_texts
from app.rag.vector_store import FaissIndex  # adjust if your class name is different


class RAGPipeline:
    """
    Simple RAG pipeline:
    - Takes a FAISS index already built over PDF chunks
    - Embeds the question
    - Retrieves top-k similar chunks
    - Builds a prompt
    - Calls a local LLM via Ollama (http://localhost:11434)
    """

    def __init__(
        self,
        index: FaissIndex,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "mistral",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.index = index
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name

        self.system_prompt = system_prompt or textwrap.dedent(
            """
            You are a compliance and risk analysis assistant.
            You ONLY answer using the provided context from financial documents.
            If the context does not contain the answer, say you do not know.
            Be concise and use bullet points where helpful.
            """
        ).strip()

    # ---------- 1. Retrieve chunks from FAISS ----------
    # def _retrieve_contexts(
    #     self,
    #     question: str,
    #     top_k: int = 5,
    # ) -> List[Dict[str, Any]]:
    #     # 1) Embed the question
    #     q_emb: np.ndarray = embed_texts([question])[0]

    #     # 2) Search index
    #     # Expecting search() -> (indices, distances)
    #     indices, distances = self.index.search(q_emb, top_k)

    #     contexts: List[Dict[str, Any]] = []
    #     for idx, dist in zip(indices, distances):
    #         meta = self.index.metadata[idx]  # metadata list from when you built the index
    #         contexts.append(
    #             {
    #                 "text": meta.get("text", ""),
    #                 "source_id": meta.get("source_id", ""),
    #                 "page_num": meta.get("page_num", None),
    #                 "score": float(dist),
    #             }
    #         )
    #     return contexts

    def _retrieve_contexts(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Embed the question, search FAISS, and return a list of context dicts.
        """
        # 1) Embed the question
        q_emb: np.ndarray = embed_texts([question])[0]

        # 2) Search the index
        indices, distances = self.index.search(q_emb, top_k)

        indices = np.array(indices).reshape(-1)
        distances = np.array(distances).reshape(-1)

        contexts: List[Dict[str, Any]] = []

        meta_list = self.index.metadata
        n_meta = len(meta_list)

        for idx, dist in zip(indices, distances):
            idx = int(idx)

            # Skip FAISS "empty" indices or out-of-range
            if idx < 0 or idx >= n_meta:
                continue

            meta = meta_list[idx]

            text: str = ""
            source_id: Optional[str] = None
            page_num: Optional[int] = None

            if isinstance(meta, dict):
                text = meta.get("text", "") or ""
                source_id = meta.get("source_id")
                page_num = meta.get("page_num")
            elif isinstance(meta, (list, tuple)):
                if len(meta) >= 1:
                    text = str(meta[0])
                if len(meta) >= 2:
                    source_id = str(meta[1]) if meta[1] is not None else None
                if len(meta) >= 3:
                    try:
                        page_num = int(meta[2]) if meta[2] is not None else None
                    except Exception:
                        page_num = None
            else:
                text = str(meta)

            contexts.append(
                {
                    "text": text,
                    "source_id": source_id or "",
                    "page_num": page_num,
                    "score": float(dist),
                }
            )

        return contexts



    # ---------- 2. Build LLM prompt ----------
    def _build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        context_texts = "\n\n---\n\n".join(
            f"[Source: {c.get('source_id','')} | Page: {c.get('page_num','?')}]"
            f"\n{c['text']}"
            for c in contexts
        )

        prompt = textwrap.dedent(
            f"""
            SYSTEM:
            {self.system_prompt}

            CONTEXT:
            {context_texts}

            QUESTION:
            {question}

            INSTRUCTIONS:
            - Use ONLY the information in CONTEXT.
            - If the answer is not in CONTEXT, say you do not know.
            - Highlight any potential compliance or risk issues if relevant.
            """
        ).strip()

        return prompt

    # ---------- 3. Call Ollama ----------
    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Non-streaming: Ollama returns a dict with 'response'
        return (data.get("response") or "").strip()

    # ---------- 4. Public method used by FastAPI & agents ----------
    def query(
        self,
        question: str,
        top_k: int = 5,
        include_contexts: bool = True,
    ) -> Dict[str, Any]:
        contexts = self._retrieve_contexts(question=question, top_k=top_k)
        prompt = self._build_prompt(question=question, contexts=contexts)
        answer = self._call_ollama(prompt)

        result: Dict[str, Any] = {"answer": answer,"prompt": prompt,}
        if include_contexts:
            result["contexts"] = contexts
        return result


# ===== Global loader for FastAPI (lazy singleton) =====

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Returns a single RAGPipeline instance (created on first call).
    FastAPI will use this as a dependency.
    """
    global _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline

    # Directory where your FAISS index/metadata were saved in Step 3
    index_dir = os.getenv("RAG_INDEX_DIR", "data/faiss_index")
    print("--------------------------------")
    print("Using RAG_INDEX_DIR:", index_dir)
    print("--------------------------------")

    index = FaissIndex.load(index_dir)

    ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "mistral")

    _rag_pipeline = RAGPipeline(
        index=index,
        ollama_url=ollama_url,
        model_name=model_name,
    )
    return _rag_pipeline

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Sequence

import faiss
import numpy as np
import json


class FaissIndex:
    """
    Thin wrapper around a FAISS index + metadata.

    - self.index: the faiss.Index instance
    - self.metadata: a list of dicts (or other JSON-serializable objects),
      one per vector
    """

    def __init__(self, index: faiss.Index, metadata: Sequence[Dict[str, Any]]) -> None:
        self.index = index
        self.metadata: List[Dict[str, Any]] = list(metadata)

    # -------------- construction helpers --------------

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        metadatas: Sequence[Dict[str, Any]],
        use_cosine: bool = False,
    ) -> "FaissIndex":
        """
        Build a FAISS index from a batch of embeddings and metadata.

        embeddings: shape (N, D), float32
        metadatas:  length N, each a dict with at least {"text": "..."} ideally
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype="float32")
        embeddings = embeddings.astype("float32")

        n, dim = embeddings.shape

        if use_cosine:
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings)
        return cls(index=index, metadata=metadatas)

    # -------------- save / load --------------

    def save(self, dir_path: str) -> None:
        """
        Save FAISS index to <dir_path>/index.faiss
        and metadata to <dir_path>/meta.json  (list of dicts)
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(p / "index.faiss"))

        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: str) -> "FaissIndex":
        """
        Load FAISS index + metadata from a directory.
        Expects:
          - <dir_path>/index.faiss
          - <dir_path>/meta.json  (list; if missing, creates empty dicts)
        """
        p = Path(dir_path)

        index = faiss.read_index(str(p / "index.faiss"))

        meta_path = p / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = []

        # Make metadata length match index.ntotal
        n = index.ntotal
        metadata = list(metadata)
        if len(metadata) < n:
            metadata.extend({} for _ in range(n - len(metadata)))
        elif len(metadata) > n:
            metadata = metadata[:n]

        return cls(index=index, metadata=metadata)

    # -------------- search --------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search the FAISS index.

        Returns:
            indices: 1D numpy array of length top_k
            distances: 1D numpy array of length top_k
        """
        v = np.array(query_embedding, dtype="float32")
        if v.ndim == 1:
            v = v.reshape(1, -1)

        distances, indices = self.index.search(v, top_k)
        distances = distances[0]
        indices = indices[0]
        return indices, distances

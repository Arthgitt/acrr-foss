from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Small, free, fast model
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model only once.
    lru_cache makes sure it is reused instead of re-loaded.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Take a list of strings and return a 2D numpy array of embeddings
    with shape (num_texts, embedding_dim).
    """
    if not texts:
        # This model uses 384-dim vectors; keep type consistent
        return np.zeros((0, 384), dtype="float32")

    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product ~= cosine similarity
    )
    return embeddings.astype("float32")

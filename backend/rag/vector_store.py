from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorStore:
    """
    Vector store with cosine similarity.
    Uses FAISS if available, otherwise falls back to a NumPy implementation.
    """

    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss and HAS_FAISS
        self.text_chunks: List[str] = []

        if self.use_faiss:
            # Cosine similarity via inner product on L2-normalized vectors
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.embeddings: List[np.ndarray] = []

    @staticmethod
    def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

    def add(self, embeddings: List[List[float]], chunks: List[str]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        vecs = np.asarray(embeddings, dtype="float32")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {vecs.shape[1]}")

        if self.use_faiss:
            vecs = self._l2_normalize(vecs)
            self.index.add(vecs)
        else:
            self.embeddings.extend(vecs)

        self.text_chunks.extend(chunks)

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.text_chunks:
            return []

        q = np.asarray([query_embedding], dtype="float32")
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {q.shape[1]}")

        if self.use_faiss:
            q = self._l2_normalize(q)
            scores, indices = self.index.search(q, top_k)
            return [
                (self.text_chunks[idx], float(scores[0][i]))
                for i, idx in enumerate(indices[0])
                if idx != -1 and idx < len(self.text_chunks)
            ]
        else:
            q_vec = q[0]
            q_norm = np.linalg.norm(q_vec) + 1e-10
            sims = []
            for i, emb in enumerate(self.embeddings):
                denom = (np.linalg.norm(emb) + 1e-10) * q_norm
                sims.append(float(np.dot(emb, q_vec) / denom))
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(self.text_chunks[i], sims[i]) for i in top_idx]

    def __len__(self) -> int:
        return len(self.text_chunks)
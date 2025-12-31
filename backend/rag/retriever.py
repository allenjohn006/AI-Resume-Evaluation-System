from typing import List, Tuple
from embeddings import EmbeddingGenerator  # your real embedding function
from .vector_store import VectorStore


class Retriever:
    """
    Retrieves top-k most similar chunks from the vector store given a text query.
    """

    def __init__(self, store: VectorStore):
        self.store = store
        self.embedding_gen = EmbeddingGenerator()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Compute the embedding for the query and return top-k (chunk, score) matches.
        """
        query_embedding = self.embedding_gen.generate_embedding(query)
        return self.store.similarity_search(query_embedding, top_k=top_k)
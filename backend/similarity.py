"""
Similarity module for calculating similarity between text embeddings
"""

import numpy as np
from typing import List, Tuple


class SimilarityCalculator:
    """
    Calculates similarity scores between text embeddings
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        """
        Initialize the similarity calculator
        
        Args:
            similarity_metric: Type of similarity metric to use (cosine, euclidean, dot)
        """
        self.similarity_metric = similarity_metric
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Distance value
        """
        return np.linalg.norm(vec1 - vec2)
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate similarity using the specified metric
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score
        """
        if self.similarity_metric == "cosine":
            return self.cosine_similarity(vec1, vec2)
        elif self.similarity_metric == "euclidean":
            return -self.euclidean_distance(vec1, vec2)  # Negative for similarity
        elif self.similarity_metric == "dot":
            return np.dot(vec1, vec2)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Alias for calculate_similarity for compatibility
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score
        """
        return self.calculate_similarity(vec1, vec2)
    
    def find_most_similar(self, query_vec: np.ndarray, 
                         candidate_vecs: List[np.ndarray], 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar vectors to the query vector
        
        Args:
            query_vec: Query vector
            candidate_vecs: List of candidate vectors
            top_k: Number of top results to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity
        """
        similarities = []
        for idx, candidate_vec in enumerate(candidate_vecs):
            similarity = self.calculate_similarity(query_vec, candidate_vec)
            similarities.append((idx, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

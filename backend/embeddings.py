"""
Embedding module for generating vector embeddings from text
"""

import os
import numpy as np
import requests
from typing import List


class EmbeddingGenerator:
    """
    Generates embeddings using OpenRouter API
    """
    
    def __init__(self, model: str = "openai/text-embedding-3-small", api_key: str = None):
        """
        Initialize the embedding generator
        
        Args:
            model: OpenRouter model name for embeddings
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            return np.array([])
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": text
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            
            data = response.json()
            embedding = data["data"][0]["embedding"]
            
            return np.array(embedding, dtype=np.float32)
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenRouter API: {e}")
            raise
        except (KeyError, IndexError) as e:
            print(f"Error parsing embedding response: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "input": batch
                }
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                response.raise_for_status()
                
                data = response.json()
                
                # Sort by index to maintain order
                batch_embeddings = sorted(data["data"], key=lambda x: x["index"])
                
                for item in batch_embeddings:
                    embeddings.append(np.array(item["embedding"], dtype=np.float32))
            
            except requests.exceptions.RequestException as e:
                print(f"Error in batch embedding: {e}")
                raise
        
        return embeddings

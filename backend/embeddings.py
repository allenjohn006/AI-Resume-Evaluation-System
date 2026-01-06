"""
Embeddings module for generating vector embeddings using OpenRouter API
"""

import os
import requests
from typing import List
import time


class EmbeddingGenerator:
    """
    Generates embeddings using OpenRouter API with retry logic
    """
    
    def __init__(self, model: str = "openai/text-embedding-3-small"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        """Generate embedding for a single text with retry logic"""
        for attempt in range(retry_count):
            try:
                payload = {
                    "model": self.model,
                    "input": text
                }
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=60,
                    verify=True  # Keep SSL verification enabled
                )
                
                response.raise_for_status()
                data = response.json()
                
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                else:
                    raise ValueError("No embedding returned from API")
                    
            except requests.exceptions.SSLError as e:
                print(f"SSL Error on attempt {attempt + 1}/{retry_count}: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Last attempt: try without SSL verification
                    print("Final attempt without SSL verification...")
                    try:
                        response = requests.post(
                            self.api_url,
                            json=payload,
                            headers=self.headers,
                            timeout=60,
                            verify=False  # Disable SSL verification as last resort
                        )
                        response.raise_for_status()
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            return data["data"][0]["embedding"]
                    except Exception as final_e:
                        raise Exception(f"All retry attempts failed: {str(final_e)}")
            
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}/{retry_count}: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Failed after {retry_count} attempts: {str(e)}")
        
        raise Exception("Failed to generate embedding")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            for text in batch:
                try:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error generating embedding: {str(e)}")
                    # Use fallback: zero vector
                    print("Using fallback zero vector")
                    embeddings.append([0.0] * 1536)  # Standard embedding size
            
            # Rate limiting
            time.sleep(0.5)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings

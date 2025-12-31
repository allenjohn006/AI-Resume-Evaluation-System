"""
LLM module for calling OpenRouter API
"""

import os
import requests


def call_llm(prompt: str, model: str = "openai/gpt-3.5-turbo") -> str:
    """
    Call OpenRouter LLM API
    
    Args:
        prompt: The prompt to send to the LLM
        model: Model to use (default: openai/gpt-3.5-turbo)
    
    Returns:
        LLM response text
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM API call failed: {str(e)}")
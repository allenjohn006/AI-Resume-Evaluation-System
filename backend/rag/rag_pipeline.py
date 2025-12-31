from typing import Any, Dict, List

from llm import call_llm  # OpenRouter (or other) LLM call
from .prompt import build_prompt
from .retriever import Retriever
from .vector_store import VectorStore


class RAGPipeline:
    """
    Simple RAG pipeline:
    1) Retrieve relevant resume chunks
    2) Retrieve relevant JD chunks
    3) Build a combined evaluation prompt
    4) Call the LLM to generate the assessment
    """

    def __init__(self, resume_store: VectorStore, jd_store: VectorStore, top_k: int = 5):
        self.resume_store = resume_store
        self.jd_store = jd_store
        self.top_k = top_k
        self.resume_retriever = Retriever(resume_store)
        self.jd_retriever = Retriever(jd_store)

    def run(self) -> Dict[str, Any]:
        """Run the RAG pipeline"""
        # Retrieve top-k relevant chunks for resume and job description
        resume_hits = self.resume_retriever.retrieve(
            query="candidate experience and skills", top_k=self.top_k
        )
        jd_hits = self.jd_retriever.retrieve(
            query="job requirements and skills", top_k=self.top_k
        )

        resume_chunks: List[str] = [chunk for chunk, _ in resume_hits]
        jd_chunks: List[str] = [chunk for chunk, _ in jd_hits]

        # Build evaluation prompt
        prompt = build_prompt(resume_chunks, jd_chunks)

        # Call LLM to generate the response
        llm_response = call_llm(prompt)

        return {
            "prompt": prompt,
            "response": llm_response,
            "resume_chunks": resume_chunks,
            "jd_chunks": jd_chunks,
            "resume_scores": [score for _, score in resume_hits],
            "jd_scores": [score for _, score in jd_hits],
        }



"""
Main entry point for the AI Resume Evaluation System
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np

from pdf_loader import PDFLoader
from chunker import TextChunker
from embeddings import EmbeddingGenerator
from similarity import SimilarityCalculator


def _embed_chunks(embedding_gen: EmbeddingGenerator, chunks: List[str]) -> List[np.ndarray]:
    """Embed a list of text chunks."""
    if not chunks:
        return []
    return [embedding_gen.generate_embedding(chunk) for chunk in chunks]


def analyze(
    resume_pdf: str,
    jd_pdf: str,
    *,
    chunk_size: int = 500,
) -> float:
    """
    Compute an overall match score between a resume PDF and a job description PDF.

    The score is the mean similarity across all pairwise combinations of resume
    and job-description chunks.
    """
    # Initialize components
    pdf_loader = PDFLoader()
    chunker = TextChunker(chunk_size=chunk_size)
    embedding_gen = EmbeddingGenerator()
    similarity_calc = SimilarityCalculator()
    
    # 1) Load text
    resume_text = pdf_loader.load_pdf(resume_pdf)
    jd_text = pdf_loader.load_pdf(jd_pdf)

    # 2) Chunk text
    resume_chunks = chunker.chunk_text(resume_text)
    jd_chunks = chunker.chunk_text(jd_text)

    if not resume_chunks or not jd_chunks:
        raise ValueError("No text chunks were produced from one or both PDFs.")

    # 3) Embed chunks
    resume_vecs = _embed_chunks(embedding_gen, resume_chunks)
    jd_vecs = _embed_chunks(embedding_gen, jd_chunks)

    # 4) Compute pairwise similarities
    scores: List[float] = []
    for r_vec in resume_vecs:
        for j_vec in jd_vecs:
            scores.append(similarity_calc.compute_similarity(r_vec, j_vec))

    if not scores:
        raise ValueError("No similarity scores were computed.")

    return float(np.mean(scores))


def main():
    """
    Main function to run the resume evaluation pipeline
    """
    parser = argparse.ArgumentParser(description="AI Resume Evaluation System")
    parser.add_argument("resume_pdf", type=Path, help="Path to the resume PDF")
    parser.add_argument("jd_pdf", type=Path, help="Path to the job description PDF")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of words per chunk (default: 500)",
    )
    args = parser.parse_args()

    print("AI Resume Evaluation System")
    print("-" * 50)
    
    score = analyze(
        str(args.resume_pdf),
        str(args.jd_pdf),
        chunk_size=args.chunk_size,
    )
    print(f"Overall Match Score: {score:.4f}")


if __name__ == "__main__":
    main()
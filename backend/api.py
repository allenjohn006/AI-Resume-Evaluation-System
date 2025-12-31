"""
FastAPI backend for the AI Resume Evaluation System with RAG integration
"""

import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from main import analyze
from embeddings import EmbeddingGenerator
from chunker import TextChunker
from pdf_loader import PDFLoader
from rag.vector_store import VectorStore
from rag.rag_pipeline import RAGPipeline
from llm import call_llm


app = FastAPI(
    title="AI Resume Evaluation System API",
    description="API for evaluating resume match with job descriptions using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for uploads
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize components
pdf_loader = PDFLoader()
chunker = TextChunker()
embedding_gen = EmbeddingGenerator()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Resume Evaluation System API with RAG",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "evaluate": "/evaluate",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Resume Evaluation System"
    }


@app.post("/evaluate")
async def evaluate_resume(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: UploadFile = File(..., description="Job Description PDF file"),
    chunk_size: int = Form(default=500, description="Number of words per chunk"),
    use_rag: bool = Form(default=True, description="Enable RAG evaluation")
):
    """
    Evaluate resume match with job description using semantic similarity and optional RAG
    
    Args:
        resume: Resume PDF file
        job_description: Job Description PDF file
        chunk_size: Number of words per chunk (default: 500)
        use_rag: Enable RAG-based evaluation (default: True)
    
    Returns:
        Evaluation results with match score and RAG insights
    """
    
    # Validate file types
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")
    
    if job_description.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Job description must be a PDF file")
    
    # Validate chunk size
    if chunk_size < 100 or chunk_size > 1000:
        raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 1000")
    
    resume_path = None
    jd_path = None
    
    try:
        # Step 1: Save uploaded files temporarily
        resume_path = TEMP_DIR / f"resume_{resume.filename}"
        jd_path = TEMP_DIR / f"jd_{job_description.filename}"
        
        with open(resume_path, "wb") as f:
            content = await resume.read()
            f.write(content)
        
        with open(jd_path, "wb") as f:
            content = await job_description.read()
            f.write(content)
        
        # Step 2: Extract text from PDFs
        resume_text = pdf_loader.load_pdf(str(resume_path))
        jd_text = pdf_loader.load_pdf(str(jd_path))
        
        # Step 3: Chunk the text
        resume_chunks = chunker.chunk_text(resume_text, chunk_size=chunk_size)
        jd_chunks = chunker.chunk_text(jd_text, chunk_size=chunk_size)
        
        if not resume_chunks or not jd_chunks:
            raise ValueError("No text chunks were produced from one or both PDFs.")
        
        # Step 4: Generate embeddings
        resume_embeddings = embedding_gen.generate_embeddings_batch(resume_chunks)
        jd_embeddings = embedding_gen.generate_embeddings_batch(jd_chunks)
        
        # Step 5: Run semantic similarity scoring
        similarity_score = analyze(
            str(resume_path),
            str(jd_path),
            chunk_size=chunk_size
        )
        
        # Determine match level
        if similarity_score >= 0.75:
            match_level = "Excellent"
        elif similarity_score >= 0.50:
            match_level = "Good"
        else:
            match_level = "Needs Improvement"
        
        result = {
            "success": True,
            "score": float(similarity_score),
            "score_percentage": f"{similarity_score * 100:.2f}%",
            "match_level": match_level,
            "analysis": {
                "raw_score": float(similarity_score),
                "percentage": similarity_score * 100,
                "chunk_size": chunk_size,
                "resume_chunks_count": len(resume_chunks),
                "jd_chunks_count": len(jd_chunks)
            },
            "interpretation": get_interpretation(similarity_score)
        }
        
        # Step 6: Run RAG pipeline if enabled
        if use_rag:
            try:
                # Create vector stores
                embedding_dim = len(resume_embeddings[0]) if resume_embeddings else 384
                resume_store = VectorStore(dim=embedding_dim, use_faiss=True)
                jd_store = VectorStore(dim=embedding_dim, use_faiss=True)
                
                # Add embeddings to stores
                resume_store.add(resume_embeddings, resume_chunks)
                jd_store.add(jd_embeddings, jd_chunks)
                
                # Run RAG pipeline
                rag_pipeline = RAGPipeline(resume_store, jd_store, top_k=5)
                rag_result = rag_pipeline.run()
                
                # Call LLM for detailed evaluation
                llm_response = call_llm(rag_result["prompt"])
                
                result["rag_evaluation"] = {
                    "enabled": True,
                    "llm_response": llm_response,
                    "retrieved_resume_chunks": rag_result["resume_chunks"],
                    "retrieved_jd_chunks": rag_result["jd_chunks"],
                    "resume_relevance_scores": rag_result["resume_scores"],
                    "jd_relevance_scores": rag_result["jd_scores"]
                }
            except Exception as e:
                result["rag_evaluation"] = {
                    "enabled": False,
                    "error": f"RAG pipeline error: {str(e)}",
                    "note": "Falling back to similarity-based evaluation"
                }
        else:
            result["rag_evaluation"] = {
                "enabled": False,
                "note": "RAG evaluation disabled"
            }
        
        return JSONResponse(status_code=200, content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during evaluation: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        if resume_path and resume_path.exists():
            resume_path.unlink()
        if jd_path and jd_path.exists():
            jd_path.unlink()


def get_interpretation(score: float) -> dict:
    """
    Get interpretation and recommendations based on score
    
    Args:
        score: Match score between 0 and 1
    
    Returns:
        Dictionary with interpretation and recommendations
    """
    if score >= 0.75:
        return {
            "level": "Excellent Match",
            "message": "Your resume is an excellent match for this job!",
            "recommendations": [
                "Your skills and experience align well with the job requirements",
                "Proceed with confidence in your application",
                "Customize your cover letter to highlight your strongest matches"
            ]
        }
    elif score >= 0.50:
        return {
            "level": "Good Match",
            "message": "Your resume is a good match for this job.",
            "recommendations": [
                "Highlight the relevant skills and experience you have",
                "Consider adding more keywords from the job description",
                "Emphasize accomplishments that match their requirements",
                "Use your cover letter to explain how you meet their needs"
            ]
        }
    else:
        return {
            "level": "Needs Improvement",
            "message": "Your resume may need some adjustments.",
            "recommendations": [
                "Identify key skills and keywords from the job description",
                "Add or emphasize relevant experience and projects",
                "Consider taking courses or certifications they mention",
                "Reorder your resume to highlight most relevant skills first",
                "Ensure you have work experience in their required domain"
            ]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
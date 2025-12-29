"""
FastAPI backend for the AI Resume Evaluation System
"""

import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from main import analyze


app = FastAPI(
    title="AI Resume Evaluation System API",
    description="API for evaluating resume match with job descriptions",
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Resume Evaluation System API",
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
    chunk_size: int = Form(default=500, description="Number of words per chunk")
):
    """
    Evaluate how well a resume matches a job description
    
    Args:
        resume: Resume PDF file
        job_description: Job Description PDF file
        chunk_size: Number of words per chunk (default: 500)
    
    Returns:
        Evaluation results with match score
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
        # Save uploaded files temporarily
        resume_path = TEMP_DIR / f"resume_{resume.filename}"
        jd_path = TEMP_DIR / f"jd_{job_description.filename}"
        
        # Write files
        with open(resume_path, "wb") as f:
            content = await resume.read()
            f.write(content)
        
        with open(jd_path, "wb") as f:
            content = await job_description.read()
            f.write(content)
        
        # Run analysis
        score = analyze(
            str(resume_path),
            str(jd_path),
            chunk_size=chunk_size
        )
        
        # Determine match level
        if score >= 0.75:
            match_level = "Excellent"
        elif score >= 0.50:
            match_level = "Good"
        else:
            match_level = "Needs Improvement"
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "score": float(score),
                "score_percentage": f"{score * 100:.2f}%",
                "match_level": match_level,
                "analysis": {
                    "raw_score": float(score),
                    "percentage": score * 100,
                    "chunk_size": chunk_size
                },
                "interpretation": get_interpretation(score)
            }
        )
    
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


@app.post("/evaluate-batch")
async def evaluate_batch(
    resumes: list[UploadFile] = File(..., description="List of resume PDF files"),
    job_description: UploadFile = File(..., description="Job Description PDF file"),
    chunk_size: int = Form(default=500, description="Number of words per chunk")
):
    """
    Evaluate multiple resumes against a single job description
    
    Args:
        resumes: List of resume PDF files
        job_description: Job Description PDF file
        chunk_size: Number of words per chunk
    
    Returns:
        List of evaluation results
    """
    
    if job_description.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Job description must be a PDF file")
    
    jd_path = None
    results = []
    
    try:
        # Save job description
        jd_path = TEMP_DIR / f"jd_{job_description.filename}"
        with open(jd_path, "wb") as f:
            content = await job_description.read()
            f.write(content)
        
        # Process each resume
        for resume in resumes:
            if resume.content_type != "application/pdf":
                results.append({
                    "filename": resume.filename,
                    "success": False,
                    "error": "File must be a PDF"
                })
                continue
            
            resume_path = None
            
            try:
                resume_path = TEMP_DIR / f"resume_{resume.filename}"
                with open(resume_path, "wb") as f:
                    content = await resume.read()
                    f.write(content)
                
                score = analyze(str(resume_path), str(jd_path), chunk_size=chunk_size)
                
                results.append({
                    "filename": resume.filename,
                    "success": True,
                    "score": float(score),
                    "score_percentage": f"{score * 100:.2f}%",
                    "match_level": "Excellent" if score >= 0.75 else "Good" if score >= 0.50 else "Needs Improvement"
                })
            
            except Exception as e:
                results.append({
                    "filename": resume.filename,
                    "success": False,
                    "error": str(e)
                })
            
            finally:
                if resume_path and resume_path.exists():
                    resume_path.unlink()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "total": len(resumes),
                "results": results
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
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
    uvicorn.run(
        "api:app",  # Changed from app to "api:app" import string
        host="0.0.0.0",
        port=8000,
        reload=True
    )
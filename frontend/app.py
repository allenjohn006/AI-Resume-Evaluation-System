"""
Streamlit UI for the AI Resume Evaluation System
"""

import streamlit as st
import requests
from pathlib import Path
import os

# API endpoint
API_URL = "http://localhost:8000"


def call_api(resume_file, jd_file, chunk_size, use_rag=True):
    """
    Call the FastAPI backend for evaluation
    """
    # Reset file pointers to beginning
    resume_file.seek(0)
    jd_file.seek(0)
    
    files = {
        "resume": ("resume.pdf", resume_file, "application/pdf"),
        "job_description": ("jd.pdf", jd_file, "application/pdf")
    }
    data = {
        "chunk_size": chunk_size,
        "use_rag": use_rag
    }
    
    response = requests.post(
        f"{API_URL}/evaluate",
        files=files,
        data=data,
        timeout=120
    )
    
    response.raise_for_status()
    return response.json()


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Resume Evaluation System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📄 AI Resume Evaluation System")
    st.markdown("---")
    st.markdown(
        "Evaluate how well your resume matches a job description using AI-powered similarity analysis and RAG."
    )
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Settings")
    
    chunk_size = st.sidebar.slider(
        "Chunk Size (words)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Number of words per text chunk for analysis"
    )
    
    use_rag = st.sidebar.checkbox(
        "Enable RAG Evaluation",
        value=True,
        help="Use Retrieval-Augmented Generation for detailed AI insights"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info(
        "This tool uses AI-powered embeddings to compare your resume with job descriptions. "
        "RAG provides detailed explanations and recommendations."
    )
    
    # Check API status
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Not Responding")
    except:
        st.sidebar.error("❌ API Offline")
        st.sidebar.info("Start backend: `python backend/api.py`")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Resume PDF")
        resume_file = st.file_uploader(
            "Upload your resume PDF",
            type="pdf",
            key="resume_upload"
        )
    
    with col2:
        st.subheader("💼 Job Description PDF")
        jd_file = st.file_uploader(
            "Upload the job description PDF",
            type="pdf",
            key="jd_upload"
        )
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🔍 Analyze Match Score", type="primary", use_container_width=True):
        if resume_file is None or jd_file is None:
            st.error("❌ Please upload both resume and job description PDFs")
        else:
            try:
                with st.spinner("Analyzing documents... This may take a minute."):
                    # Call API
                    result = call_api(resume_file, jd_file, chunk_size, use_rag)
                    
                    if not result.get("success"):
                        st.error(f"❌ Error: {result.get('detail', 'Unknown error')}")
                        return
                    
                    score = result.get("score", 0)
                    match_level = result.get("match_level", "Unknown")
                    interpretation = result.get("interpretation", {})
                    analysis = result.get("analysis", {})
                    rag_eval = result.get("rag_evaluation", {})
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Score display with color coding
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.metric(
                            label="Overall Match Score",
                            value=f"{score:.2%}",
                            delta=match_level
                        )
                    
                    # Detailed metrics
                    st.markdown("### 📈 Score Breakdown")
                    col_raw, col_percent, col_chunks = st.columns(3)
                    with col_raw:
                        st.metric("Raw Score", f"{score:.4f}")
                    with col_percent:
                        st.metric("Percentage Match", f"{score * 100:.2f}%")
                    with col_chunks:
                        st.metric("Chunks Analyzed", 
                                f"R:{analysis.get('resume_chunks_count', 0)} | "
                                f"JD:{analysis.get('jd_chunks_count', 0)}")
                    
                    # Score interpretation
                    st.markdown("### 💡 Interpretation")
                    if score >= 0.75:
                        st.success(interpretation.get("message", ""))
                    elif score >= 0.50:
                        st.info(interpretation.get("message", ""))
                    else:
                        st.warning(interpretation.get("message", ""))
                    
                    # Recommendations
                    st.markdown("### 🎯 Recommendations")
                    recommendations = interpretation.get("recommendations", [])
                    for rec in recommendations:
                        st.write(f"• {rec}")
                    
                    # RAG Evaluation Results
                    if rag_eval.get("enabled"):
                        st.markdown("---")
                        st.markdown("### 🤖 AI-Powered Detailed Evaluation (RAG)")
                        
                        llm_response = rag_eval.get("llm_response", "")
                        if llm_response:
                            st.markdown(llm_response)
                        
                        # Show retrieved chunks
                        with st.expander("📚 View Retrieved Context"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Resume Chunks:**")
                                resume_chunks = rag_eval.get("retrieved_resume_chunks", [])
                                for i, chunk in enumerate(resume_chunks, 1):
                                    st.text_area(
                                        f"Resume Chunk {i}",
                                        chunk,
                                        height=100,
                                        key=f"resume_chunk_{i}"
                                    )
                            
                            with col2:
                                st.markdown("**Job Description Chunks:**")
                                jd_chunks = rag_eval.get("retrieved_jd_chunks", [])
                                for i, chunk in enumerate(jd_chunks, 1):
                                    st.text_area(
                                        f"JD Chunk {i}",
                                        chunk,
                                        height=100,
                                        key=f"jd_chunk_{i}"
                                    )
                    elif rag_eval.get("error"):
                        st.warning(f"⚠️ {rag_eval.get('note', 'RAG evaluation unavailable')}")
                        st.info(f"Details: {rag_eval.get('error', 'Unknown error')}")
                    
                    # Download results
                    st.markdown("---")
                    st.markdown("### 📥 Export Results")
                    
                    result_text = f"""AI Resume Evaluation Results
=============================

Overall Match Score: {score:.2%}
Raw Score: {score:.4f}
Match Level: {match_level}

Analysis Settings:
- Chunk Size: {chunk_size} words
- Resume Chunks: {analysis.get('resume_chunks_count', 0)}
- JD Chunks: {analysis.get('jd_chunks_count', 0)}
- RAG Enabled: {use_rag}

Interpretation:
{interpretation.get('message', '')}

Recommendations:
"""
                    for rec in recommendations:
                        result_text += f"- {rec}\n"
                    
                    if rag_eval.get("enabled") and rag_eval.get("llm_response"):
                        result_text += f"\n\nAI Detailed Evaluation:\n{rag_eval.get('llm_response', '')}\n"
                    
                    result_text += f"\nGenerated by AI Resume Evaluation System v1.0\n"
                    
                    st.download_button(
                        label="📄 Download Results as Text",
                        data=result_text,
                        file_name="resume_evaluation_results.txt",
                        mime="text/plain"
                    )
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the FastAPI backend is running.")
                st.info("Run in terminal: `python backend/api.py`")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The analysis is taking too long.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API Error: {e}")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please ensure both PDFs are valid and readable.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>AI Resume Evaluation System</strong> v1.0</p>
            <p>Powered by advanced embedding, similarity analysis, and RAG</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
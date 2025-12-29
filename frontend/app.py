"""
Streamlit UI for the AI Resume Evaluation System
"""

import streamlit as st
import requests
from pathlib import Path
import os

# API endpoint
API_URL = "http://localhost:8000"


def call_api(resume_file, jd_file, chunk_size):
    """
    Call the FastAPI backend for evaluation
    """
    files = {
        "resume": ("resume.pdf", resume_file),
        "job_description": ("jd.pdf", jd_file)
    }
    data = {"chunk_size": chunk_size}
    
    response = requests.post(
        f"{API_URL}/evaluate",
        files=files,
        data=data
    )
    
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
        "Evaluate how well your resume matches a job description using AI-powered similarity analysis."
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
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info(
        "This tool uses AI-powered embeddings to compare your resume with job descriptions. "
        "Higher scores indicate better matches."
    )
    
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
                with st.spinner("Analyzing documents..."):
                    # Call API
                    result = call_api(resume_file, jd_file, chunk_size)
                    
                    if not result.get("success"):
                        st.error(f"❌ Error: {result.get('detail', 'Unknown error')}")
                        return
                    
                    score = result.get("score", 0)
                    match_level = result.get("match_level", "Unknown")
                    interpretation = result.get("interpretation", {})
                    
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
                    col_raw, col_percent = st.columns(2)
                    with col_raw:
                        st.metric("Raw Score", f"{score:.4f}")
                    with col_percent:
                        st.metric("Percentage Match", f"{score * 100:.2f}%")
                    
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

Interpretation:
{interpretation.get('message', '')}

Recommendations:
"""
                    for rec in recommendations:
                        result_text += f"- {rec}\n"
                    
                    result_text += f"\nGenerated by AI Resume Evaluation System v1.0\n"
                    
                    st.download_button(
                        label="📄 Download Results as Text",
                        data=result_text,
                        file_name="resume_evaluation_results.txt",
                        mime="text/plain"
                    )
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the FastAPI backend is running.")
                st.info("Run: `python backend/api.py` in another terminal")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please ensure both PDFs are valid and readable.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>AI Resume Evaluation System</strong> v1.0</p>
            <p>Powered by advanced embedding and similarity analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
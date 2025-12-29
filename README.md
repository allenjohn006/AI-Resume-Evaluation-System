# AI Resume Evaluation System

An AI-powered system for evaluating and analyzing resumes.

## Project Structure

```
AI_RESUME_EVALUATION_SYSTEM/
│
├── backend/          # Backend application code
├── frontend/         # Frontend application code
├── data/            # Data files and datasets
├── utils/           # Utility functions and helpers
├── .env             # Environment variables
├── requirements.txt # Python dependencies
├── .gitignore       # Git ignore rules
└── README.md        # Project documentation
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`
4. Run the application

## Features

- Resume parsing and analysis
- AI-powered evaluation
- Scoring and recommendations

## License

[Add your license here]

## How to Run the Project (Backend + Frontend)

The AI Resume Evaluation System follows a client–server architecture:

- Backend (FastAPI) → Handles resume evaluation, embeddings, similarity scoring, and RAG logic
- Frontend (Streamlit) → Provides an interactive UI for uploading resumes and viewing results

Both run as separate services.

### Prerequisites
- Python 3.9+
- Git
- VS Code (recommended)

### Project Setup

Clone the repository:
```bash
git clone https://github.com/your-username/AI_RESUME_EVALUATION_SYSTEM.git
cd AI_RESUME_EVALUATION_SYSTEM
```

Create and activate a virtual environment:
- Windows
```bash
python -m venv venv
venv\Scripts\activate
```
- macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_api_key_here
```
Required for embedding generation and RAG-based reasoning.

### Running the Application
Use two terminals: one for the backend and one for the frontend.

Terminal 1: Start FastAPI Backend
```bash
cd backend
uvicorn main:app --reload
```
If successful:
- Uvicorn running on http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

Terminal 2: Start Streamlit Frontend
```bash
cd frontend
streamlit run app.py
```
Streamlit will open at:
- http://localhost:8501

### How the System Works (Runtime Flow)
1. User uploads Resume & JD (Streamlit UI)
2. Frontend sends request to FastAPI backend
3. Backend processes PDFs & embeddings
4. Similarity / RAG reasoning is performed
5. Structured results returned to frontend
6. Results displayed in UI

### Features Available
- PDF resume & job description upload
- Embedding-based semantic similarity scoring
- Section-wise match analysis
- Skill gap identification
- Recruiter-style explanations (RAG)
- Explainable and grounded AI outputs

### Development Notes
Backend and frontend run as independent services, enabling:
- Scalability
- Easier debugging
- Production deployment readiness

### Project Status
- Core semantic matching engine ✅
- FastAPI backend integration ✅
- Streamlit frontend interface ✅

# HealthAI Clinical Intelligence Platform

HealthAI is a healthcare AI project that combines a FastAPI backend, a Streamlit frontend, RAG-based treatment recommendations, triage support, medication safety checks, analytics exports, and a disease risk prediction model.

## Features

- Patient, staff, appointment, and health record management
- AI triage classification for patient symptoms and vitals
- Disease risk prediction with scikit-learn
- Retrieval-augmented clinical recommendations from guideline documents
- Medication and allergy safety analysis
- Admin analytics dashboard with PDF and Excel exports
- Streamlit dashboard with FastAPI service integration

## Project Structure

```text
backend/     FastAPI app, database models, RAG, ML, triage, medication logic
frontend/    Streamlit dashboard
scripts/     Database setup, guideline generation, model training, maintenance scripts
data/        Reference data, generated guidelines, and trained model artifacts
run.py       Unified startup script for database setup, model training, API, and UI
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Configure environment variables.

```powershell
Copy-Item .env.example .env
```

Add API keys to `.env` if you want LLM-powered recommendations.

4. Start the full app.

```powershell
python run.py
```

The backend runs at `http://127.0.0.1:8000` and the frontend runs at `http://127.0.0.1:8501`.

## Notes

- `.env`, virtual environments, logs, SQLite databases, and uploaded clinical files are intentionally ignored by git.
- This project is for educational and prototype use. It is not a substitute for professional medical advice or clinical judgment.


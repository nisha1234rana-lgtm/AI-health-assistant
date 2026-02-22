# AI-Driven Personalized Symptom-Based Health Recommendation System

## Overview
An AI-powered healthcare assistant that predicts possible diseases based on symptoms, asks adaptive follow-up questions using information gain, and provides safe, non-diagnostic health guidance.

## Tech Stack
- Python
- FastAPI
- Streamlit
- scikit-learn (Random Forest)
- Entropy-based question engine

## How to Run

1. Install dependencies
pip install -r requirements.txt

2. Train the model
python train_model.py

3. Start Backend
cd backend
python -m uvicorn app:app --port 8001

4. Start Frontend
cd ../frontend
python -m streamlit run streamlit_app.py
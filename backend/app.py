from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import uuid

# ---------------- App ----------------
app = FastAPI(title="AI Symptom Assistant (Educational)")

DISCLAIMER = (
    "Educational tool only — not a doctor and not a diagnosis. "
    "If symptoms are severe, worsening, or you feel unsafe, seek professional medical care."
)

# ---------------- Load model artifacts ----------------
rf = joblib.load("../models/rf_disease_model.joblib")
symptom_columns = joblib.load("../models/symptom_columns.joblib")
classes = rf.classes_
REC_DF = pd.read_csv("disease_recommendations.csv")

def get_recommendation_for(disease: str):
    row = REC_DF[REC_DF["disease"] == disease]
    if row.empty:
        return None
    return {
        "self_care": [x.strip() for x in row.iloc[0]["self_care"].split(",")],
        "warning_signs_seek_care": [x.strip() for x in row.iloc[0]["warning_signs"].split(",")]
    }

# ---------------- In-memory sessions (demo) ----------------
SESSIONS = {}  # session_id -> {"answers": dict, "asked": list}

# ---------------- Helpers ----------------
def safe_normalize(arr: np.ndarray) -> np.ndarray:
    s = float(arr.sum())
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s

def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

def build_input(user_answers: dict) -> pd.DataFrame:
    x = pd.DataFrame([np.zeros(len(symptom_columns))], columns=symptom_columns)
    for s, v in user_answers.items():
        if s in x.columns:
            x.loc[0, s] = 1 if int(v) == 1 else 0
    return x

def predict_topk(x: pd.DataFrame, k: int = 3):
    probs = rf.predict_proba(x)[0]
    idx = np.argsort(probs)[::-1][:k]
    return classes[idx].tolist(), [float(probs[i]) for i in idx], probs

def choose_next_symptom(user_answers: dict, asked: list, top_diseases: list) -> Optional[str]:
    x = build_input(user_answers)
    probs_full = rf.predict_proba(x)[0]
    focus_mask = np.isin(classes, top_diseases)

    p_focus = safe_normalize(probs_full[focus_mask])
    H_before = entropy(p_focus)

    candidates = [s for s in symptom_columns if s not in user_answers and s not in asked]
    if not candidates:
        return None

    best_symptom = None
    best_gain = -1e9

    for s in candidates:
        x_yes = x.copy()
        x_yes.loc[0, s] = 1
        p_yes = safe_normalize(rf.predict_proba(x_yes)[0][focus_mask])
        H_yes = entropy(p_yes)

        x_no = x.copy()
        x_no.loc[0, s] = 0
        p_no = safe_normalize(rf.predict_proba(x_no)[0][focus_mask])
        H_no = entropy(p_no)

        H_after = 0.5 * H_yes + 0.5 * H_no
        gain = H_before - H_after

        if gain > best_gain:
            best_gain = gain
            best_symptom = s

    return best_symptom

def symptom_to_question(symptom: str) -> str:
    if symptom == "loss_of_taste_smell":
        return "Have you noticed a loss of taste or smell recently?"
    return f"Do you have {symptom.replace('_', ' ')}?"

# ---------------- API models ----------------
class StartResponse(BaseModel):
    session_id: str
    disclaimer: str

class AnswerRequest(BaseModel):
    session_id: str
    symptom: str
    value: int = Field(..., ge=0, le=1)

class NextResponse(BaseModel):
    session_id: str
    top_diseases: List[str]
    top_probs: List[float]
    next_symptom: Optional[str]
    next_question: Optional[str]
    stop: bool
    disclaimer: str

# ---------------- Endpoints ----------------
@app.post("/start_session", response_model=StartResponse)
def start_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"answers": {}, "asked": []}
    return StartResponse(session_id=sid, disclaimer=DISCLAIMER)

@app.post("/answer")
def answer(req: AnswerRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.symptom not in symptom_columns:
        raise HTTPException(status_code=400, detail="Unknown symptom")

    SESSIONS[req.session_id]["answers"][req.symptom] = int(req.value)
    if req.symptom not in SESSIONS[req.session_id]["asked"]:
        SESSIONS[req.session_id]["asked"].append(req.symptom)

    return {"status": "ok"}

@app.get("/next", response_model=NextResponse)
def next_step(session_id: str, stop_confidence: float = 0.80):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    state = SESSIONS[session_id]
    x = build_input(state["answers"])

    top_diseases, top_probs, _ = predict_topk(x, k=3)

    # stop rule
    if top_probs[0] >= stop_confidence or len(state["asked"]) >= 8:
        return NextResponse(
            session_id=session_id,
            top_diseases=top_diseases,
            top_probs=top_probs,
            next_symptom=None,
            next_question=None,
            stop=True,
            disclaimer=DISCLAIMER
        )

    next_symptom = choose_next_symptom(state["answers"], state["asked"], top_diseases)
    next_question = symptom_to_question(next_symptom) if next_symptom else None

    if next_symptom and next_symptom not in state["asked"]:
        state["asked"].append(next_symptom)

    return NextResponse(
        session_id=session_id,
        top_diseases=top_diseases,
        top_probs=top_probs,
        next_symptom=next_symptom,
        next_question=next_question,
        stop=False,
        disclaimer=DISCLAIMER
    )

@app.get("/final")
def final(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    state = SESSIONS[session_id]
    x = build_input(state["answers"])
    top_diseases, top_probs, _ = predict_topk(x, k=3)

    primary = top_diseases[0]
    rec = get_recommendation_for(primary)

    if rec is None:
        rec = {
            "self_care": [
                "Rest and drink fluids regularly.",
                "Monitor symptoms and avoid heavy exertion if unwell."
            ],
            "red_flags_seek_care": [
                "Trouble breathing, chest pain, confusion, fainting",
                "Rapidly worsening symptoms"
            ]
        }

    return {
        "session_id": session_id,
        "top_diseases": top_diseases,
        "top_probs": top_probs,
        "recommendations": {
            "note": "Not a diagnosis. Educational guidance only.",
            **rec
        },
        "disclaimer": DISCLAIMER
    }
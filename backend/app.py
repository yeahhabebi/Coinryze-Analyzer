# backend/app.py
import os, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Coinryze Backend")
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_model.joblib")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    last_numbers: list

class PredictResponse(BaseModel):
    predicted_number: int
    predicted_color: str
    predicted_size: str
    odd_even: str
    confidence: float = 0.0

model = None
try:
    import joblib
    model = joblib.load(MODEL_PATH)
    print("Loaded RF model:", MODEL_PATH)
except Exception as e:
    model = None
    print("Warning: model not loaded:", e)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    seq = np.array(req.last_numbers[-30:], dtype=float) if req.last_numbers else np.array([0.0])
    feat = np.array([[float(seq[-1]), float(seq.mean()), float(seq.std())]])
    pred = int(model.predict(feat)[0])
    odd_even = "Even" if pred % 2 == 0 else "Odd"
    color = "Green" if pred % 2 == 0 else "Red"
    size = "Big" if pred >= 25 else "Small"
    return PredictResponse(predicted_number=int(pred), predicted_color=color, predicted_size=size, odd_even=odd_even, confidence=0.6)

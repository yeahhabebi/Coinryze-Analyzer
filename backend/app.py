from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Coinryze Analyzer Backend")

class PredictionRequest(BaseModel):
    issue_id: str
    number: int
    color: str
    size: str
    odd_even: str

@app.get("/")
def root():
    return {"status": "Backend OK", "message": "CoinryzeAnalyzer API Running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    # Example prediction logic
    try:
        # dummy rule for demo (replace later)
        color_signal = {"Green": 1, "Red": -1, "Purple": 0}.get(req.color, 0)
        prediction = (req.number + color_signal) % 2
        result = "Odd" if prediction else "Even"
        return {"predicted_next": result}
    except Exception as e:
        return {"error": str(e)}

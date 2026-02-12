from fastapi import FastAPI
import joblib
import pandas as pd
from app.schemas import ChurnInput, ChurnPrediction

app = FastAPI(title="Customer Churn Prediction API")

# Load model
model = joblib.load("models/churn_model.joblib")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "ready"}

@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(input_data: ChurnInput):
    # Convert to DataFrame for the pipeline
    input_df = pd.DataFrame([input_data.dict()])
    
    probability = model.predict_proba(input_df)[0][1]
    prediction = probability >= 0.5
    
    return {
        "churn_probability": float(probability),
        "will_churn": bool(prediction)
    }
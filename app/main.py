from fastapi import FastAPI
import joblib
import pandas as pd
from app.schemas import ChurnInput, ChurnPrediction

app = FastAPI(title="Customer Churn Prediction API")

# Load model (this will show error in logs if file missing)
try:
    model = joblib.load("models/churn_model.joblib")
    print("Model loaded successfully from models/churn_model.joblib")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    model = None  # Don't crash the app

@app.get("/")
def root():
    return {
        "message": "API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "ready" if model is not None else "failed to load"}

@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(input_data: ChurnInput):
    if model is None:
        return {"detail": "Model not loaded - check logs"}
    
    # Convert to DataFrame for the pipeline
    input_df = pd.DataFrame([input_data.dict()])
    
    probability = model.predict_proba(input_df)[0][1]
    prediction = probability >= 0.5
    
    return {
        "churn_probability": float(probability),
        "will_churn": bool(prediction)
    }
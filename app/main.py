print("THIS IS THE MAIN MODULE BEING EXECUTED")

from fastapi import FastAPI
import joblib
import pandas as pd
# from app.schemas import ChurnInput, ChurnPrediction   # commented out for now

app = FastAPI(title="Customer Churn Prediction API")

print("FastAPI app created - routes should register next")

# Load model with debug
try:
    model = joblib.load("models/churn_model.joblib")
    print("Model loaded successfully from models/churn_model.joblib")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    model = None

@app.get("/")
def root():
    return {
        "message": "API is running",
        "docs_url": "/docs",
        "health_url": "/health",
        "debug_test_url": "/debug-test",
        "predict_url": "POST /predict (debug mode)"
    }

@app.get("/health")
def health_check():
    model_status = "ready" if model is not None else "failed to load"
    return {"status": "healthy", "model": model_status}

@app.get("/debug-test")
def debug_test():
    return {"debug": "this plain GET route should appear in /docs"}

@app.post("/predict")
def predict_churn():
    if model is None:
        return {"detail": "Model not loaded - check server logs"}
    
    # Debug/simplified response (no input required yet)
    return {
        "test_message": "Predict endpoint is now reachable!",
        "churn_probability": 0.42,
        "will_churn": False
    }

# Original version - commented out until debug POST works
"""
@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(input_data: ChurnInput):
    if model is None:
        return {"detail": "Model not loaded"}
    
    input_df = pd.DataFrame([input_data.dict()])
    probability = model.predict_proba(input_df)[0][1]
    prediction = probability >= 0.5
    
    return {
        "churn_probability": float(probability),
        "will_churn": bool(prediction)
    }
"""

print("predict endpoint registered (debug version)")
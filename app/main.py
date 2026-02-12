print("THIS IS THE MAIN MODULE BEING EXECUTED")
from fastapi import FastAPI
import joblib
import pandas as pd
# Temporarily comment out pydantic imports to isolate if schemas.py is the issue
# from app.schemas import ChurnInput, ChurnPrediction

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
        "predict_url": "POST /predict (currently in debug mode)"
    }

@app.get("/health")
def health_check():
    model_status = "ready" if model is not None else "failed to load"
    return {"status": "healthy", "model": model_status}

# Temporary simplified version - no Pydantic dependency
# This helps check if the route registers at all
@app.post("/predict")
def predict_churn():
    if model is None:
        return {"detail": "Model not loaded - check server logs"}
    
    # For testing: always return a fixed response
    # Later restore full logic once route is confirmed working
    return {
        "test_message": "Predict endpoint is now reachable!",
        "churn_probability": 0.42,  # dummy value
        "will_churn": False
    }

# Original full version - commented out for debugging
"""
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
"""

print("predict endpoint registered (debug version)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
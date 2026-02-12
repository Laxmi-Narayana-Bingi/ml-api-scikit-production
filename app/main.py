from fastapi import FastAPI

app = FastAPI()

print("Minimal app started")

@app.get("/")
def root():
    return {"hello": "world"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    return {"predict": "works now"}
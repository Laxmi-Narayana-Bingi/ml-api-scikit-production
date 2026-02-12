# ml-api-scikit-production
A production-ready REST API that predicts customer churn (whether a telecom customer will leave) using scikit-learn. It has a clean FastAPI backend, Docker container, CI basics, great README, live deployment, and a 2-minute demo video by covering full lifecycle (data → model → API → container → cloud) in one clickable place.

Production-ready Customer Churn Prediction API  
**Live Demo** → https://ml-api-scikit-production.onrender.com/docs  
**Accuracy** → 76.33% on test set

## Problem
Predict if a telecom customer will churn using usage & contract data.

## Architecture
![Architecture Diagram](images/architecture.png)  
*(diagram will be added after deployment)*

## Tech Stack
- FastAPI  
- scikit-learn (RandomForest)  
- Docker  
- Python 3.12

## How to run locally

```bash
git clone https://github.com/Laxmi-Narayana-Bingi/ml-api-scikit-production.git
cd ml-api-scikit-production
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
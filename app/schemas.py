from pydantic import BaseModel

class ChurnInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

class ChurnPrediction(BaseModel):
    churn_probability: float
    will_churn: bool
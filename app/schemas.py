from pydantic import BaseModel, Field

class CardioRiskRequest(BaseModel):
    age: float = Field(..., example=63.0, ge=1.0, le=120.0, description="Age in years")
    sex: int = Field(..., example=1, ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., example=1, ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., example=145.0, ge=50.0, le=250.0, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., example=233.0, ge=80.0, le=600.0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., example=1, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., example=2, ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., example=150.0, ge=50.0, le=230.0, description="Max heart rate achieved")
    exang: int = Field(..., example=0, ge=0, le=1, description="Exercise-induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., example=2.3, ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., example=3, ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float = Field(..., example=0.0, ge=0.0, le=3.0, description="Number of major vessels (0-3)")
    thal: float = Field(..., example=6.0, description="Thal (3=normal, 6=fixed, 7=reversable)")

class CardioRiskResponse(BaseModel):
    prediction: int
    heart_disease_risk: str
    confidence: float
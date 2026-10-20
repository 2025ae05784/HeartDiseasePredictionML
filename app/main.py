import os
import joblib
import pandas as pd
import logging
from app.config import settings
from fastapi import FastAPI, HTTPException
from app.schemas import CardioRiskRequest, CardioRiskResponse
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(
    title="Cardiovascular Risk Prediction Service",
    description="End-to-End MLOps Service for UCI Heart Disease Risk Assessment",
    version="1.0.0"
)

# Instrument Prometheus Metrics
Instrumentator().instrument(app).expose(app)

MODEL_PATH = settings.MODEL_PATH
model_pipeline = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Critical error: Model binary missing at {MODEL_PATH}")
        return None

    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Successfully loaded ML pipeline artifact.")
        return pipeline
    except Exception:
        logger.exception("Critical error: Failed to load ML pipeline artifact.")
        return None

@app.on_event("startup")
def initialize_model():
    global model_pipeline
    model_pipeline = load_model()

@app.get("/healthstatus", tags=["System Health"])
def health_check():
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = load_model()
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline uninitialized.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/evaluate-risk", response_model=CardioRiskResponse, tags=["Inference"])
def predict_cardiac_risk(payload: CardioRiskRequest):
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = load_model()
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is unavailable.")

    try:
        data_dict = payload.dict()
        df = pd.DataFrame([data_dict])
        
        pred = int(model_pipeline.predict(df)[0])
        confidence = float(model_pipeline.predict_proba(df)[0][pred])

        risk_label = "High Risk (Presence)" if pred == 1 else "Low Risk (Absence)"
        logger.info(f"Inference Requested | Risk: {risk_label} | Confidence: {confidence:.4f}")

        return CardioRiskResponse(
            prediction=pred,
            heart_disease_risk=risk_label,
            confidence=round(confidence, 4)
        )
    except Exception as e:
        logger.error(f"Inference failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")
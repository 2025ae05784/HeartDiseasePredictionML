import os
from pathlib import Path


class Settings:
    PROJECT_NAME: str = "Cardiovascular Risk Prediction Service"
    VERSION: str = "1.0.0"
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "vascular_risk_estimator.joblib"
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
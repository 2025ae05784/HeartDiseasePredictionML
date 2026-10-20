import os
import joblib
import numpy as np
import pandas as pd

def test_trained_pipeline_artifact_validity():
    model_path = os.path.join("models", "vascular_risk_estimator.joblib")
    assert os.path.exists(model_path), "Model binary missing! Run `python src/train.py` first."

    pipeline = joblib.load(model_path)
    
    sample_patient = pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0.0, "thal": 6.0
    }])

    pred = pipeline.predict(sample_patient)
    probabilities = pipeline.predict_proba(sample_patient)

    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)
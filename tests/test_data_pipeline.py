import pandas as pd
import numpy as np
from src.feature_engineering import process_raw_data, ClinicalFeatureBuilder

def test_raw_data_cleaning_and_target_binarization():
    sample_raw = pd.DataFrame({
        "age": [63, 67, 57],
        "sex": [1, 1, 0],
        "cp": [1, 4, 3],
        "trestbps": [145, 160, 120],
        "chol": [233, 286, 254],
        "fbs": [1, 0, 0],
        "restecg": [2, 2, 0],
        "thalach": [150, 108, 163],
        "exang": [0, 1, 0],
        "oldpeak": [2.3, 1.5, 0.6],
        "slope": [3, 2, 2],
        "ca": [0.0, np.nan, 1.0],  # Includes NaN
        "thal": [6.0, 3.0, np.nan], # Includes NaN
        "target": [0, 2, 3]        # Multi-class target
    })

    cleaned = process_raw_data(sample_raw)
    
    assert cleaned['ca'].isnull().sum() == 0, "Missing values in 'ca' were not imputed."
    assert cleaned['thal'].isnull().sum() == 0, "Missing values in 'thal' were not imputed."
    assert set(cleaned['target'].unique()).issubset({0, 1}), "Target was not properly binarized."
    assert cleaned.loc[1, 'target'] == 1, "Multi-class target > 0 must map to 1."

def test_clinical_feature_builder_transformer():
    sample_df = pd.DataFrame({
        "age": [50.0],
        "chol": [200.0],
        "thalach": [170.0],
        "exang": [1],
        "oldpeak": [2.0],
        "ca": [1.0]
    })
    
    builder = ClinicalFeatureBuilder()
    transformed = builder.transform(sample_df)
    
    assert "chol_age_ratio" in transformed.columns
    assert "hrr_ratio" in transformed.columns
    assert "vascular_risk_score" in transformed.columns
    assert np.isclose(transformed.loc[0, "chol_age_ratio"], 4.0)
    assert np.isclose(transformed.loc[0, "hrr_ratio"], 170.0 / 170.0)
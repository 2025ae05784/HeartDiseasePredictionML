import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

PROCESSED_PATH = os.path.join("data", "processed", "cardio_processed_features.csv")

class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer adding domain-specific clinical indicators:
    1. Chol-to-Age Ratio: Serum cholesterol normalized by age
    2. Heart Rate Reserve Ratio: Maximum achieved heart rate relative to predicted max rate (220 - age)
    3. Cardiac Risk Index: Compound score combining exercise angina, oldpeak, and vessel blockage
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        # 1. Chol-to-Age Ratio
        X_df['chol_age_ratio'] = X_df['chol'] / (X_df['age'] + 1e-5)
        
        # 2. Heart Rate Reserve Ratio (HRR Ratio)
        predicted_max_hr = 220.0 - X_df['age']
        X_df['hrr_ratio'] = X_df['thalach'] / (predicted_max_hr + 1e-5)
        
        # 3. Compound Vascular Risk Score
        X_df['vascular_risk_score'] = (X_df['exang'] * 1.5) + X_df['oldpeak'] + (X_df['ca'].fillna(0) * 1.2)
        
        return X_df

def process_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans missing records and converts target into binary classification."""
    df = raw_df.copy()
    
    # Impute missing values ('?') using median for numeric and mode for categorical
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    # Convert multi-class target (0..4) to binary (0 = healthy, 1 = disease present)
    df['target'] = (df['target'] > 0).astype(int)
    
    return df

def generate_eda_report(df: pd.DataFrame, save_dir: str = "screenshots") -> None:
    """Generates professional analytics visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Chart 1: Class Balance & Gender Demographics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x='target', hue='sex', ax=axes[0], palette=['#4C72B0', '#DD8452'])
    axes[0].set_title("Heart Disease Presence by Biological Sex")
    axes[0].set_xticklabels(["Absence (0)", "Presence (1)"])
    axes[0].set_xlabel("Diagnostic Output")
    axes[0].set_ylabel("Patient Count")
    
    # Chart 2: Clinical Risk Score vs Age Distribution
    builder = ClinicalFeatureBuilder()
    df_eng = builder.transform(df)
    sns.boxplot(data=df_eng, x='target', y='vascular_risk_score', ax=axes[1], palette="Set2")
    axes[1].set_title("Compound Vascular Risk Score vs Heart Disease")
    axes[1].set_xticklabels(["Absence (0)", "Presence (1)"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_demographics_risk.png"), dpi=300)
    plt.close()

    # Chart 3: Correlation Heatmap
    plt.figure(figsize=(12, 9))
    corr = df_eng.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Feature Correlation Matrix (Including Engineered Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    from data_loader import fetch_uci_dataset
    df_raw = fetch_uci_dataset()
    df_clean = process_raw_data(df_raw)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    generate_eda_report(df_clean)
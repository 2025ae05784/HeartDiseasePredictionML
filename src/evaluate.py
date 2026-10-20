import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import process_raw_data
from src.train import build_full_pipeline
from sklearn.ensemble import RandomForestClassifier


def evaluate(data_path: str = "data/raw/processed.cleveland.data") -> dict[str, float]:
    raw_data = pd.read_csv(data_path, header=None, names=[
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target"
    ], na_values="?")
    data = process_raw_data(raw_data)
    features = data.drop(columns=["target"])
    target = data["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    model = build_full_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)).fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/processed.cleveland.data")
    print(evaluate(os.fspath(parser.parse_args().data)))

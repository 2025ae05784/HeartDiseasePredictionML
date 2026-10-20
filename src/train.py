import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.data_loader import fetch_uci_dataset
from src.feature_engineering import ClinicalFeatureBuilder, PROCESSED_PATH, process_raw_data

def build_full_pipeline(classifier) -> Pipeline:
    """Constructs an end-to-end scikit-learn pipeline including feature engineering and scaling."""
    numeric_base = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    numeric_engineered = ['chol_age_ratio', 'hrr_ratio', 'vascular_risk_score']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_base + numeric_engineered),
        ('cat', cat_pipeline, categorical_cols)
    ])

    return Pipeline([
        ('feature_builder', ClinicalFeatureBuilder()),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def _load_training_data() -> pd.DataFrame:
    if os.path.exists(PROCESSED_PATH):
        return pd.read_csv(PROCESSED_PATH)

    df = process_raw_data(fetch_uci_dataset())
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return df

def execute_experimentation_suite():
    df = _load_training_data()
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    mlflow.set_experiment("Heart_Disease_Risk_Engine")

    candidate_models = {
        "ElasticNet_LogisticRegression": {
            "model": LogisticRegression(solver="saga", max_iter=2000, random_state=42),
            "params": {
                "classifier__penalty": ["elasticnet"],
                "classifier__l1_ratio": [0.2, 0.5, 0.8],
                "classifier__C": [0.05, 0.1, 0.5, 1.0]
            }
        },
        "Tuned_RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 8],
                "classifier__min_samples_split": [2, 4],
                "classifier__max_features": ["sqrt", "log2"]
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 4]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_auc = 0.0
    winning_pipeline = None
    winning_model_name = ""

    for name, config in candidate_models.items():
        with mlflow.start_run(run_name=name):
            pipeline = build_full_pipeline(config['model'])
            
            grid = GridSearchCV(
                pipeline,
                param_grid=config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            # Log to MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_best_auc", grid.best_score_)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_roc_auc", auc)

            # Log Confusion Matrix Artifact
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            cm_filename = f"{name}_cm.png"
            plt.savefig(cm_filename)
            plt.close()
            mlflow.log_artifact(cm_filename)
            os.remove(cm_filename)

            mlflow.sklearn.log_model(best_model, artifact_path="model")

            print(f"Model: {name:<30} | Test Accuracy: {acc:.4f} | Test ROC-AUC: {auc:.4f}")

            if auc > best_overall_auc:
                best_overall_auc = auc
                winning_pipeline = best_model
                winning_model_name = name

    # Persist Winning Artifact
    os.makedirs("models", exist_ok=True)
    export_path = os.path.join("models", "vascular_risk_estimator.joblib")
    joblib.dump(winning_pipeline, export_path)
    print(f"\n Champion Model Selected: {winning_model_name} (ROC-AUC: {best_overall_auc:.4f})")
    print(f" Exported pipeline artifact to: {export_path}")

if __name__ == "__main__":
    execute_experimentation_suite()
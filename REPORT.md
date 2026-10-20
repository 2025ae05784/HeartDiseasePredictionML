# Heart Disease Risk API: MLOps Experimental Learning Report

## 1. Executive summary

This project implements an end-to-end binary classifier for the UCI Heart Disease
dataset. The workflow acquires and validates the Cleveland data, performs
missing-value treatment and target binarization, creates clinically motivated
features, trains several candidate classifiers with cross-validation, tracks
experiments in MLflow, packages the winning scikit-learn pipeline, and serves it
through a FastAPI application.

The service exposes `/evaluate-risk`, `/healthstatus`, and `/metrics`. Docker and Kubernetes
manifests are included for local or cluster deployment. GitHub Actions runs
dependency installation, linting, data preparation, model training, evaluation,
tests, image build, and container verification.

This is an educational risk-classification demonstration, not a medical
diagnosis system. Predictions require qualified clinical review.

## 2. Repository map

| Concern | Implementation |
| --- | --- |
| Data acquisition | `src/data_loader.py` |
| Cleaning and EDA | `src/feature_engineering.py` |
| Feature engineering | `ClinicalFeatureBuilder` in `src/feature_engineering.py` |
| Model training and tuning | `src/train.py` |
| Holdout evaluation | `src/evaluate.py` |
| API schema and service | `app/schemas.py`, `app/main.py` |
| Automated tests | `tests/` |
| CI/CD | `.github/workflows/production_pipeline.yml` |
| Container | `Dockerfile`, `docker-compose.yml` |
| Kubernetes | `k8s/` |
| Monitoring | `monitoring/prometheus.yml`, `/metrics` |
| EDA notebook | `notebooks/eda_and_experimentation.ipynb` |

## 3. Data acquisition and EDA

The bundled archive is under `data/raw/heart-disease/` and contains the UCI
metadata, the Cleveland, Hungarian, Switzerland, and VA data variants, and the
cost files. The pipeline's reproducible default is
`data/raw/processed.cleveland.data`, with the UCI URL retained in
`src/data_loader.py` for cache-aware downloading.

The Cleveland file contains 303 records and 14 columns. Missing `?` values are
parsed as nulls. `ca` and `thal` are imputed with their training-data mode, and
the original multi-class disease value is mapped to binary presence/absence:
`target = (target > 0).astype(int)`.

The EDA workflow produces:

- `screenshots/eda_demographics_risk.png`: class balance and risk-score comparison
- `screenshots/eda_correlation_heatmap.png`: correlation matrix including engineered features
- `notebooks/eda_and_experimentation.ipynb`: reproducible inspection and evaluation

## 4. Feature engineering and modeling

The pipeline preserves preprocessing and modeling as one serialized object,
preventing training-serving skew. `ClinicalFeatureBuilder` adds:

1. Cholesterol-to-age ratio.
2. Maximum heart-rate to predicted maximum heart-rate ratio.
3. A vascular-risk score combining exercise angina, ST depression, and vessel count.

Numeric features are median-imputed and standardized. Categorical features are
mode-imputed and one-hot encoded with unknown-category handling.

The training workflow evaluates three candidates:

- Elastic-net logistic regression.
- Tuned random forest.
- Gradient boosting.

Each candidate uses stratified five-fold cross-validation with ROC-AUC scoring,
grid search, a held-out test split, confusion-matrix artifact generation, and
MLflow parameter/metric/model logging. The highest held-out ROC-AUC model is
saved to `models/vascular_risk_estimator.joblib`.

The standalone evaluation command is:

```text
python -m src.evaluate
```

One verified local run reported:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9344 |
| Precision | 0.9286 |
| Recall | 0.9286 |
| F1 | 0.9286 |
| ROC-AUC | 0.9610 |

These numbers are a single fixed split and should not be interpreted as clinical
validation or a population-level performance guarantee.

## 5. Experiment tracking and reproducibility

MLflow experiments are written to `mlruns/` during training. Each candidate run
records tuned parameters, cross-validation ROC-AUC, holdout metrics, the
confusion matrix, and the MLflow model artifact. The generated joblib artifact
contains the feature builder, preprocessing transformer, and classifier.

A clean setup is:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m src.data_loader
python -m src.feature_engineering
python -m src.train
python -m src.evaluate
python -m pytest -q
```

The dependency file pins the application and ML stack and constrains setuptools
to retain the `pkg_resources` compatibility required by the pinned MLflow
release.

## 6. API and packaging

`POST /evaluate-risk` accepts the 13 Cleveland input features as validated JSON and
returns:

```json
{
  "prediction": 1,
  "heart_disease_risk": "High Risk (Presence)",
  "confidence": 0.91
}
```

`GET /healthstatus` reports model readiness, while `/metrics` is exposed by
`prometheus-fastapi-instrumentator`. Model-load errors are logged explicitly and
the API returns HTTP 503 until a valid artifact is available.

Run locally with:

```text
uvicorn app.main:app --reload
```

## 7. CI/CD

The GitHub Actions workflow:

1. Checks out the repository.
2. Installs Python 3.10 dependencies.
3. Runs blocking flake8 syntax and undefined-name checks.
4. Executes data acquisition, EDA generation, training, and evaluation.
5. Runs the complete pytest suite.
6. Builds the Docker image.
7. Runs the isolated container verification script.
8. Uploads the trained model artifact.

The workflow is configured to fail on lint, test, training, evaluation, build, or
health-check errors.

## 8. Container and Kubernetes deployment

Build and run locally:

```text
docker build -t cardio-risk-api:latest .
docker run --rm -p 8000:8000 cardio-risk-api:latest
curl http://localhost:8000/healthstatus
```

The compose file also starts Prometheus and Grafana. Kubernetes manifests
provide a two-replica deployment, resource limits, liveness/readiness probes,
an HPA from two to five replicas, a LoadBalancer service, and an Ingress.
`imagePullPolicy: Never` intentionally supports local Docker Desktop or Minikube;
use a registry image and an appropriate pull policy for a public cluster.

## 9. Monitoring and operations

Prometheus scrapes `cardio_risk_api:8000/metrics` every five seconds. Docker Compose
automatically provisions the Prometheus data source and the dashboard from
`monitoring/grafana_dashboard.json` using the files under
`monitoring/grafana/provisioning/`.
Application logs record model loading, prediction outcomes, confidence, and
inference failures. The load generator in `scripts/load_test.py` can produce
traffic for a local dashboard demonstration.

## 10. Verification checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Data and EDA | Bundled archive, loader, notebook, two plots | Complete |
| Two or more classifiers | Three candidates and grid searches | Complete |
| Metrics and cross-validation | MLflow runs and evaluation command | Complete |
| Reusable model | Joblib pipeline artifact | Complete |
| Requirements | Pinned `requirements.txt` | Complete |
| Tests and CI | Five passing tests and GitHub Actions | Complete |
| API | FastAPI `/evaluate-risk`, `/healthstatus`, `/metrics` | Complete |
| Container definition | Dockerfile, compose, verifier | Ready; run with Docker |
| Kubernetes | Deployment, HPA, Service, Ingress | Ready for local cluster |
| Monitoring | Prometheus and Grafana configuration | Ready for local stack |
| Deployment screenshots | Existing EDA screenshots only | Capture locally |
| Public cloud URL | Not provisioned in this repository | Student action required |
| Ten-page PDF/DOCX | This Markdown report is source material | Export and personalize |
| Video demonstration | Cannot be generated by repository code | Student action required |

## 11. Academic integrity and limitations

The implementation should be submitted only after the student understands and can
explain the data transformations, model-selection rationale, API behavior,
deployment commands, and monitoring configuration. Replace the measured results
with results from your own final run, add your own screenshots and architecture
diagram, cite the UCI source, and record your own demonstration. Do not present
the repository's generated artifacts, screenshots, or prose as independent work
without review and personalization.

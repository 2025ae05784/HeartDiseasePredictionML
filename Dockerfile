# Stage 1: Dependency Assembly
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment in /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production Execution Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files and grant permissions to non-root user
COPY --chown=10001:10001 src/ ./src/
COPY --chown=10001:10001 app/ ./app/
COPY --chown=10001:10001 models/vascular_risk_estimator.joblib ./models/vascular_risk_estimator.joblib

RUN test -s /app/models/vascular_risk_estimator.joblib

EXPOSE 8000

USER 10001

ENV MODEL_PATH=/app/models/vascular_risk_estimator.joblib

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
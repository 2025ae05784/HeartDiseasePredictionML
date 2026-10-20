#!/usr/bin/env bash
set -eo pipefail

CONTAINER_NAME="cardio_risk_api_sandbox"
PORT=8000
HOST="http://localhost:${PORT}"

echo "==> Cleaning up prior test containers..."
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

echo "==> Spinning up container ${CONTAINER_NAME}..."
docker run -d --name ${CONTAINER_NAME} -p ${PORT}:8000 cardio-risk-api:latest

cleanup() {
  echo "==> Terminating sandbox container..."
  docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Polling /healthstatus endpoint..."
MAX_RETRIES=15
COUNTER=0
until [ $(curl -s -o /dev/null -w "%{http_code}" ${HOST}/healthstatus) -eq 200 ]; do
  COUNTER=$((COUNTER+1))
  if [ ${COUNTER} -eq ${MAX_RETRIES} ]; then
    echo "x-> Health check failed after ${MAX_RETRIES} attempts!"
    exit 1
  fi
  sleep 1
done

echo "==> Container is online and healthy!"
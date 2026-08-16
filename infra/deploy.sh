#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-3e52c857-15d9-4fe6-b2f}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-legal-lenz}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}:$(git rev-parse --short HEAD)"

gcloud config set project "${PROJECT_ID}"
gcloud builds submit --tag "${IMAGE}"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --service-account "${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=true" \
  --set-env-vars "INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME},DB_NAME=${DB_NAME},DB_USER=${DB_USER},UPLOAD_BUCKET=${UPLOAD_BUCKET}" \
  --set-secrets "/app/.streamlit/secrets.toml=legal-lenz-streamlit-secrets:latest"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-2d06ae6e-1f76-495e-809}"
REGION="${REGION:-asia-south1}"
JOB="${JOB:-legal-lenz-cleanup}"
IMAGE="${IMAGE:?set IMAGE to the deployed Legal Lenz image}"
INSTANCE_CONNECTION_NAME="${INSTANCE_CONNECTION_NAME:?set INSTANCE_CONNECTION_NAME}"
DB_NAME="${DB_NAME:-legal_lenz}"
DB_USER="${DB_USER:?set DB_USER}"

gcloud run jobs deploy "${JOB}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --command python \
  --args manage.py,cleanup \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=true,INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME},DB_NAME=${DB_NAME},DB_USER=${DB_USER}" \
  --service-account "legal-lenz@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud scheduler jobs create http "${JOB}-daily" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --schedule "0 3 * * *" \
  --uri "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB}:run" \
  --http-method POST \
  --oauth-service-account-email "legal-lenz@${PROJECT_ID}.iam.gserviceaccount.com" || true

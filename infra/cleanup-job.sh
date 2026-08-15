#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-3e52c857-15d9-4fe6-b2f}"
REGION="${REGION:-asia-south1}"
JOB="${JOB:-legal-lenz-cleanup}"
IMAGE="${IMAGE:?set IMAGE to the deployed Legal Lenz image}"

gcloud run jobs deploy "${JOB}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --command python \
  --args manage.py,cleanup \
  --service-account "legal-lenz@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud scheduler jobs create http "${JOB}-daily" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --schedule "0 3 * * *" \
  --uri "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB}:run" \
  --http-method POST \
  --oauth-service-account-email "legal-lenz@${PROJECT_ID}.iam.gserviceaccount.com" || true

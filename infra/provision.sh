#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-3e52c857-15d9-4fe6-b2f}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-legal-lenz}"
BUCKET="${UPLOAD_BUCKET:?set UPLOAD_BUCKET}"

gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com sqladmin.googleapis.com storage.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com cloudscheduler.googleapis.com
gcloud iam service-accounts create "${SERVICE}" --display-name "Legal Lenz Cloud Run" || true
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" --uniform-bucket-level-access || true
gcloud storage buckets update "gs://${BUCKET}" --clear-soft-delete
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member "serviceAccount:${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" --role roles/cloudsql.client
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member "serviceAccount:${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" --role roles/aiplatform.user
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" --role roles/storage.objectAdmin
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member "serviceAccount:${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" --role roles/secretmanager.secretAccessor
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member "serviceAccount:${SERVICE}@${PROJECT_ID}.iam.gserviceaccount.com" --role roles/run.developer

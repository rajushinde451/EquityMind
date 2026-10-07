#!/usr/bin/env bash
# One-time GCP bootstrap for EquityMind CI/CD (idempotent).
#
# Creates: Artifact Registry repo, Secret Manager secret, runtime + deploy
# service accounts, and Workload Identity Federation for GitHub Actions.
#
# Usage:
#   PROJECT_ID=my-proj GITHUB_REPO=owner/EquityMind GOOGLE_API_KEY=... ./deploy/setup_gcp.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${GITHUB_REPO:?set GITHUB_REPO (owner/repo)}"
REGION="${REGION:-us-central1}"
REPO="equitymind"
RUNTIME_SA="equitymind-runtime"
DEPLOY_SA="equitymind-deployer"
POOL="github-pool"
PROVIDER="github-provider"

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo ">> Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  cloudtrace.googleapis.com logging.googleapis.com iamcredentials.googleapis.com aiplatform.googleapis.com

echo ">> Artifact Registry"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION"

echo ">> Secret Manager"
gcloud secrets describe equitymind-google-api-key >/dev/null 2>&1 || \
  gcloud secrets create equitymind-google-api-key --replication-policy automatic
if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
  printf '%s' "$GOOGLE_API_KEY" | gcloud secrets versions add equitymind-google-api-key --data-file=-
fi

echo ">> Service accounts"
for sa in "$RUNTIME_SA" "$DEPLOY_SA"; do
  gcloud iam service-accounts describe "$sa@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$sa"
done
RUNTIME="$RUNTIME_SA@$PROJECT_ID.iam.gserviceaccount.com"
DEPLOYER="$DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com"

# Runtime: least privilege — read the secret, write traces/logs.
for role in roles/secretmanager.secretAccessor roles/cloudtrace.agent roles/logging.logWriter roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$RUNTIME" --role "$role" --condition=None >/dev/null
done
# Deployer: push images, deploy Cloud Run, act as runtime SA.
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$DEPLOYER" --role "$role" --condition=None >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME" \
  --member "serviceAccount:$DEPLOYER" --role roles/iam.serviceAccountUser >/dev/null
# Lets CI mint an ID token as the deployer to smoke-test the private service.
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" \
  --member "serviceAccount:$DEPLOYER" --role roles/iam.serviceAccountTokenCreator >/dev/null

echo ">> Workload Identity Federation (keyless GitHub auth)"
gcloud iam workload-identity-pools describe "$POOL" --location global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$POOL" --location global --display-name "GitHub"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global --workload-identity-pool "$POOL" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location global --workload-identity-pool "$POOL" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition "assertion.repository=='$GITHUB_REPO'"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$GITHUB_REPO" >/dev/null

cat <<EOF

Done. Add these GitHub repository *variables* (Settings > Secrets and variables > Actions > Variables):
  GCP_PROJECT_ID   = $PROJECT_ID
  GCP_REGION       = $REGION
  GCP_WIF_PROVIDER = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER
  GCP_DEPLOY_SA    = $DEPLOYER
  GCP_RUNTIME_SA   = $RUNTIME
And the repository *secret* GOOGLE_API_KEY (used by the CI agent-eval job).
EOF

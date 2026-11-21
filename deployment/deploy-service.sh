#!/usr/bin/env bash
#
# deploy-service.sh — build and deploy the query agent service (query_service/,
# ub-query-agent#92) to Cloud Run from a laptop.
#
# SUPERSEDED by .github/workflows/deploy-query-service.yml (D37, #105). Kept
# only until that workflow has deployed once: it has never been run, and its
# deployer account still needs the IAM listed in its header. Delete this script
# after the workflow's first good prod run.
#
# Private: no unauthenticated access. UB's runtime account is its only invoker
# (UB terraform query_agent.tf, query_agent_provisioned = true after the first
# deploy). The infra it runs on (account, database, IAM database user, KMS key)
# is UB terraform's query_agent.tf; apply that first.
#
# Postgres needs no secret (IAM database auth). The Gemini Enterprise client's
# id and secret, when the Environment names their Secret Manager secrets, are
# mounted by name (--update-secrets); their values are never read here.
#
# Every per-environment value is read from the GitHub Environment named by the
# argument (the same variables the workflow reads), so nothing here is a
# literal copy of prod's values.
#
# Usage: deployment/deploy-service.sh <prod|stage|dev>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/deployment/github-env.sh"

GH_ENV="${1:?usage: deployment/deploy-service.sh <prod|stage|dev>}"

need() {
  local value
  value="$(gh_env_var "$GH_ENV" "$1")"
  [[ -n "$value" ]] || { echo "error: variable $1 is not set in GitHub Environment $GH_ENV" >&2; exit 1; }
  printf '%s' "$value"
}

PROJECT="$(need GCP_PROJECT_ID)"
PROJECT_NUMBER="$(need GCP_PROJECT_NUMBER)"
REGION="$(need GCP_REGION)"
SERVICE="$(need QUERY_SERVICE_NAME)"
SA="$(need QUERY_SERVICE_SA)"
ENGINE_ID="$(need QUERY_SERVICE_ENGINE_ID)"
UB_BASE_URL="$(need UB_BASE_URL)"
CLOUD_SQL_INSTANCE="$(need CLOUD_SQL_INSTANCE)"
DB_NAME="$(need QUERY_SERVICE_DB_NAME)"
GRANT_KMS_KEY="$(need QUERY_SERVICE_GRANT_KMS_KEY)"
PROD_ENGINE_ID="$(prod_engine_id)"
# Optional (connect page and Workspace, #93 / #90): see the workflow.
PUBLIC_URL="$(gh_env_var "$GH_ENV" QUERY_SERVICE_PUBLIC_URL)"
GOOGLE_CLIENT_ID_SECRET="$(gh_env_var "$GH_ENV" QUERY_SERVICE_GOOGLE_CLIENT_ID_SECRET)"
GOOGLE_CLIENT_SECRET_SECRET="$(gh_env_var "$GH_ENV" QUERY_SERVICE_GOOGLE_CLIENT_SECRET_SECRET)"
GOOGLE_HOSTED_DOMAIN="$(gh_env_var "$GH_ENV" QUERY_SERVICE_GOOGLE_HOSTED_DOMAIN)"
if [[ -n "$GOOGLE_CLIENT_ID_SECRET" || -n "$GOOGLE_CLIENT_SECRET_SECRET" ]] \
   && [[ -z "$GOOGLE_CLIENT_ID_SECRET" || -z "$GOOGLE_CLIENT_SECRET_SECRET" ]]; then
  echo "error: set both QUERY_SERVICE_GOOGLE_CLIENT_ID_SECRET and ..._CLIENT_SECRET_SECRET, or neither" >&2
  exit 1
fi

# Today the service talks to the SANDBOX engine, never the engine the
# production Chat bot uses. Moving /ask to the prod engine is
# ub-query-agent#106 (owner approval); it lifts this guard deliberately.
[[ "$ENGINE_ID" != "$PROD_ENGINE_ID" ]] || {
  echo "error: QUERY_SERVICE_ENGINE_ID is the production engine ($PROD_ENGINE_ID)" >&2
  exit 1
}
[[ "$SA" == *"@$PROJECT.iam.gserviceaccount.com" ]] || {
  echo "error: QUERY_SERVICE_SA ($SA) is not in project $PROJECT" >&2
  exit 1
}

ENV_VARS=(
  "ENGINE=projects/${PROJECT_NUMBER}/locations/${REGION}/reasoningEngines/${ENGINE_ID}"
  "UB_BASE_URL=${UB_BASE_URL}"
  "CLOUD_SQL_INSTANCE=${CLOUD_SQL_INSTANCE}"
  "DB_NAME=${DB_NAME}"
  "DB_IAM_USER=${SA%.gserviceaccount.com}"
  "GRANT_KMS_KEY=${GRANT_KMS_KEY}"
)
[[ -z "$PUBLIC_URL" ]] || ENV_VARS+=("PUBLIC_BASE_URL=${PUBLIC_URL}")
[[ -z "$GOOGLE_HOSTED_DOMAIN" ]] || ENV_VARS+=("GOOGLE_HOSTED_DOMAIN=${GOOGLE_HOSTED_DOMAIN}")
SECRETS=()
if [[ -n "$GOOGLE_CLIENT_ID_SECRET" ]]; then
  SECRETS=(--update-secrets "GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID_SECRET}:latest,GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET_SECRET}:latest")
fi

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source query_service \
  --service-account "$SA" \
  --no-allow-unauthenticated \
  --set-env-vars "^||^$(IFS='|'; echo "${ENV_VARS[*]}" | sed 's/|/||/g')" \
  "${SECRETS[@]}" \
  --labels "app=ub-query-agent,commit=$(git rev-parse --short HEAD)" \
  --cpu 1 --memory 512Mi \
  --concurrency 20 --min-instances 0 --max-instances 3 \
  --timeout 300 \
  --quiet

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'

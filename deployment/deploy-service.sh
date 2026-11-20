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
# No secret is passed: the service reaches Postgres with IAM database auth.
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

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source query_service \
  --service-account "$SA" \
  --no-allow-unauthenticated \
  --set-env-vars "^||^$(IFS='|'; echo "${ENV_VARS[*]}" | sed 's/|/||/g')" \
  --labels "app=ub-query-agent,commit=$(git rev-parse --short HEAD)" \
  --cpu 1 --memory 512Mi \
  --concurrency 20 --min-instances 0 --max-instances 3 \
  --timeout 300 \
  --quiet

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'

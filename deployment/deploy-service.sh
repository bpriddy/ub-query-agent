#!/usr/bin/env bash
#
# deploy-service.sh — build and deploy the query agent service (query_service/,
# ub-query-agent#92) to Cloud Run, on dev.
#
# Private: no unauthenticated access. UB's runtime account is its only invoker
# (UB terraform query_agent.tf, query_agent_provisioned = true after the first
# deploy). The infra it runs on (account, database, IAM database user, KMS key)
# is UB terraform's query_agent.tf; apply that first.
#
# No secret is passed: the service reaches Postgres with IAM database auth.
#
# Usage: deployment/deploy-service.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROJECT="os-test-491819"
PROJECT_NUMBER="843516467880"
REGION="us-central1"
ENVIRONMENT="dev"
SERVICE="ub-query-agent-${ENVIRONMENT}"
SA="sa-ub-query-agent-${ENVIRONMENT}@${PROJECT}.iam.gserviceaccount.com"

# The SANDBOX engine: dev never talks to the engine the production Chat bot
# uses (deploy-sandbox.env SANDBOX_AGENT_ENGINE_ID).
ENGINE_ID="$(sed -n 's/^SANDBOX_AGENT_ENGINE_ID=//p' deploy-sandbox.env)"
PROD_ENGINE_ID="9136379226620952576"
[[ -n "$ENGINE_ID" && "$ENGINE_ID" != "$PROD_ENGINE_ID" ]] || {
  echo "error: no sandbox engine id, or it is the production engine" >&2
  exit 1
}

UB_BASE_URL="https://gcp-universal-backend-${ENVIRONMENT}-${PROJECT_NUMBER}.${REGION}.run.app"

ENV_VARS=(
  "ENGINE=projects/${PROJECT_NUMBER}/locations/${REGION}/reasoningEngines/${ENGINE_ID}"
  "UB_BASE_URL=${UB_BASE_URL}"
  "CLOUD_SQL_INSTANCE=${PROJECT}:${REGION}:gub-platform"
  "DB_NAME=ub_query_agent_${ENVIRONMENT}"
  "DB_IAM_USER=${SA%.gserviceaccount.com}"
  "GRANT_KMS_KEY=projects/${PROJECT}/locations/${REGION}/keyRings/${SERVICE}/cryptoKeys/workspace-grant"
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

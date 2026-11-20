#!/usr/bin/env bash
#
# teardown-sandbox.sh — delete the SANDBOX Agent Engine.
#
# Agent Engine bills per deployed engine, so an idle sandbox costs money. This
# deletes it. There is no gcloud surface for reasoning engines (checked:
# `gcloud [alpha|beta] ai reasoning-engines` does not exist), so the call goes
# straight to the Vertex REST API with the caller's ADC/gcloud token.
#
# Usage:
#   deployment/teardown-sandbox.sh               # ID from deploy-sandbox.env
#   deployment/teardown-sandbox.sh <engine-id>   # explicit
#
# Refuses, always, to delete the production engine. After a successful delete,
# blank SANDBOX_AGENT_ENGINE_ID in deploy-sandbox.env so the next
# deploy-sandbox.sh run creates a fresh engine instead of failing on a 404.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="deploy-sandbox.env"
REGION="us-central1"
# Read from GitHub Environment `prod` (AGENT_ENGINE_ID); fails closed (D37).
source "$REPO_ROOT/deployment/github-env.sh"
PROD_ENGINE_ID="$(prod_engine_id)"

ENGINE_ID="${1:-${SANDBOX_AGENT_ENGINE_ID:-$(
  sed -nE 's/^[[:space:]]*SANDBOX_AGENT_ENGINE_ID[[:space:]]*=[[:space:]]*//p' "$ENV_FILE" 2>/dev/null | tail -1
)}}"
ENGINE_ID="${ENGINE_ID##*/}"   # accept a full resource name too

[[ -n "$ENGINE_ID" ]] || { echo "error: no engine id (argument, SANDBOX_AGENT_ENGINE_ID, or $ENV_FILE)." >&2; exit 1; }
[[ "$ENGINE_ID" =~ ^[0-9]+$ ]] || { echo "error: engine id must be numeric, got '$ENGINE_ID'." >&2; exit 1; }

if [[ "$ENGINE_ID" == "$PROD_ENGINE_ID" ]]; then
  echo "REFUSED: $ENGINE_ID is the PRODUCTION engine (Chat bot + Gemini Enterprise)." >&2
  exit 1
fi

PROJECT="${GCP_PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}"
if [[ -z "$PROJECT" && -f .env ]]; then
  PROJECT="$(sed -nE 's/^[[:space:]]*GCP_PROJECT_ID[[:space:]]*=[[:space:]]*//p' .env | tail -1)"
fi
[[ -n "$PROJECT" ]] || { echo "error: set GCP_PROJECT_ID (or GOOGLE_CLOUD_PROJECT)." >&2; exit 1; }

RESOURCE="projects/$PROJECT/locations/$REGION/reasoningEngines/$ENGINE_ID"
API="https://$REGION-aiplatform.googleapis.com/v1"
# ADC first (what adk deploy itself authenticates with), the gcloud user
# credential as fallback — the latter needs an interactive reauth when stale.
TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null || gcloud auth print-access-token)"

# Show what we are about to delete — display name is the human check that this
# is gub-agent-sandbox and not something else that happens to share a project.
echo "About to delete: $RESOURCE"
curl -sfS -H "Authorization: Bearer $TOKEN" "$API/$RESOURCE" \
  | python3 -c 'import json,sys; e=json.load(sys.stdin); print(f"  displayName: {e.get(\"displayName\")}\n  description: {e.get(\"description\")}\n  updated:     {e.get(\"updateTime\")}")'

if [[ "${YES:-}" != "1" ]]; then
  read -r -p "Type the engine id to confirm deletion: " confirm
  [[ "$confirm" == "$ENGINE_ID" ]] || { echo "aborted."; exit 1; }
fi

# force=true also removes child resources (sessions, memories) the engine owns;
# without it a used engine returns FAILED_PRECONDITION.
curl -sfS -X DELETE -H "Authorization: Bearer $TOKEN" "$API/$RESOURCE?force=true" \
  | python3 -c 'import json,sys; op=json.load(sys.stdin); print("  operation:", op.get("name"))'

echo "Deleted $RESOURCE (the operation above completes asynchronously)."
echo "Now blank SANDBOX_AGENT_ENGINE_ID in $ENV_FILE."

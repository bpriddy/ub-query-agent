#!/usr/bin/env bash
#
# deploy-sandbox.sh — deploy this working tree to the SANDBOX Agent Engine.
#
# Same package as production, different env: the sandbox engine is deployed
# with deploy-sandbox.env, which turns SANDBOX_ENABLED on (and EMIT_THINKING with
# it). Isolation is a property of the deploy, not of caller discipline — an
# experiment aimed at the sandbox cannot reach the engine serving the Chat bot
# because that engine has the flag off and ignores state["sandbox"] entirely.
#
# Deliberately does NOT call deployment/register_agent.py: the sandbox engine
# must never appear in the Gemini Enterprise agent list, or real users would be
# routed to whatever prompt an experiment left behind.
#
# Usage:
#   deployment/deploy-sandbox.sh                       # create or update
#   deployment/deploy-sandbox.sh "what changed"        # with a description
#
# First run creates a new engine and prints its ID; record it in deploy-sandbox.env
# as SANDBOX_AGENT_ENGINE_ID so later runs update in place instead of creating
# (and billing for) another one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ABSOLUTE on purpose: `adk deploy` chdir()s into its temp folder BEFORE it
# reads --env_file (ADK 2.6.1 cli_deploy.py:1000 vs :1094), so a relative path
# is looked up in the wrong directory and SILENTLY skipped — the engine comes up
# with no env at all. That is exactly how the production engine ended up with an
# empty deploymentSpec while deploy.yml passed a relative --env_file (fixed:
# deploy.yml now passes $GITHUB_WORKSPACE/deploy-prod.env and verifies it).
ENV_FILE="$REPO_ROOT/deploy-sandbox.env"
AGENT_PACKAGE="gub_agent"          # the package exporting root_agent
DISPLAY_NAME="gub-agent-sandbox"
REGION="us-central1"               # engine RESOURCE region; the model stays on
                                   # the global endpoint, pinned in config.py
# The production engine this script must never touch. Read from GitHub
# Environment `prod` (AGENT_ENGINE_ID), not a literal (D37, #105); fails closed
# if it cannot be resolved.
source "$REPO_ROOT/deployment/github-env.sh"
PROD_ENGINE_ID="$(prod_engine_id)"
DESCRIPTION="${1:-sandbox — $(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# ── Preconditions ─────────────────────────────────────────────────────────────

[[ -f "$ENV_FILE" ]] || { echo "error: $ENV_FILE not found in $REPO_ROOT" >&2; exit 1; }

# The one thing that makes this engine a sandbox. Deploying without it would
# produce a second engine that ignores every override — a silent no-op sandbox.
if ! grep -Eq '^[[:space:]]*SANDBOX_ENABLED[[:space:]]*=[[:space:]]*(1|true|yes)[[:space:]]*$' "$ENV_FILE"; then
  echo "error: $ENV_FILE does not set SANDBOX_ENABLED to 1/true/yes." >&2
  echo "       Without it the deployed engine ignores state[\"sandbox\"]." >&2
  exit 1
fi

PROJECT="${GCP_PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}"
if [[ -z "$PROJECT" && -f .env ]]; then
  # .env is gitignored and local-only; read the project out of it without
  # sourcing the file (it holds a service JWT we have no business exporting).
  PROJECT="$(sed -nE 's/^[[:space:]]*GCP_PROJECT_ID[[:space:]]*=[[:space:]]*//p' .env | tail -1)"
fi
[[ -n "$PROJECT" ]] || { echo "error: set GCP_PROJECT_ID (or GOOGLE_CLOUD_PROJECT)." >&2; exit 1; }

# Existing sandbox engine, if one has been recorded.
ENGINE_ID="${SANDBOX_AGENT_ENGINE_ID:-$(
  sed -nE 's/^[[:space:]]*SANDBOX_AGENT_ENGINE_ID[[:space:]]*=[[:space:]]*//p' "$ENV_FILE" | tail -1
)}"

# The guard that matters: never let this script touch the engine Gemini
# Enterprise and the Chat bot are pointed at, however the ID got here.
if [[ -n "$ENGINE_ID" && "$ENGINE_ID" == *"$PROD_ENGINE_ID"* ]]; then
  echo "error: SANDBOX_AGENT_ENGINE_ID resolves to the PRODUCTION engine" >&2
  echo "       ($PROD_ENGINE_ID). Refusing to deploy an experiment build to it." >&2
  exit 1
fi

# ── Deploy ────────────────────────────────────────────────────────────────────

args=(
  --project="$PROJECT"
  --region="$REGION"
  --display_name="$DISPLAY_NAME"
  --description="$DESCRIPTION"
  # NOTE: --env_file is deprecated in ADK 2.6.1 (warns, still works — it
  # populates the deploy's env_vars). Its successor is an `env_vars` block in
  # an .agent_engine_config.json passed with --agent_engine_config_file. Keep
  # the env in a per-deploy FILE either way: a config committed inside
  # gub_agent/ would be picked up by the production deploy too.
  --env_file="$ENV_FILE"
  # The repo-root services.py, staged at /app/services.py: `adk api_server`
  # imports it at start and it registers the agentengine:// session service
  # (SESSION_CLIENT_REUSE). Left out, the engine silently runs ADK's stock one.
  --extra_packages="$REPO_ROOT/services.py"
)
if [[ -n "$ENGINE_ID" ]]; then
  echo "Updating sandbox engine $ENGINE_ID in place."
  args+=(--agent_engine_id="$ENGINE_ID")
else
  echo "No SANDBOX_AGENT_ENGINE_ID recorded — creating a NEW engine."
  echo "Agent Engine bills per deployed engine: record the ID below in $ENV_FILE."
fi

log="$(mktemp -t deploy-sandbox.XXXXXX)"
trap 'rm -f "$log"' EXIT

set +e
adk deploy agent_engine "${args[@]}" "$AGENT_PACKAGE" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e

# ── Report the engine ID ──────────────────────────────────────────────────────

# ADK prints "Created a new instance: projects/.../reasoningEngines/N" and
# "Deployed to Agent Platform: projects/.../reasoningEngines/N".
DEPLOYED_ID="$(grep -oE 'reasoningEngines/[0-9]+' "$log" | tail -1 | cut -d/ -f2)"

echo
if [[ -z "$DEPLOYED_ID" ]]; then
  echo "Deploy did not report an engine ID — read the output above." >&2
  exit $(( status == 0 ? 1 : status ))
fi
if [[ "$DEPLOYED_ID" == "$PROD_ENGINE_ID" ]]; then
  echo "STOP: the deploy addressed the PRODUCTION engine $PROD_ENGINE_ID." >&2
  exit 1
fi

RESOURCE="projects/$PROJECT/locations/$REGION/reasoningEngines/$DEPLOYED_ID"
echo "SANDBOX_AGENT_ENGINE_ID=$DEPLOYED_ID"
echo "  resource: $RESOURCE"
echo "  prod is:  $PROD_ENGINE_ID  (they must differ — they do)"

# ── Verify the env actually landed ────────────────────────────────────────────
# The whole point of this engine is SANDBOX_ENABLED. Read it back from the
# deployed resource so a silently-skipped env file (see ENV_FILE above) fails
# here, loudly, instead of producing a sandbox that ignores every override.
if [[ "$status" -eq 0 ]]; then
  # ADC first (what adk deploy itself authenticated with), the gcloud user
  # credential as fallback — the latter needs an interactive reauth when stale.
  TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null || gcloud auth print-access-token)"
  ENGINE_ENV="$(curl -sfS -H "Authorization: Bearer $TOKEN" \
      "https://$REGION-aiplatform.googleapis.com/v1/$RESOURCE" \
    | python3 -c 'import json, sys
e = json.load(sys.stdin)
spec = e.get("spec", {}).get("deploymentSpec", {})
pairs = [v["name"] + "=" + str(v.get("value", "")) for v in (spec.get("env") or [])]
print(" ".join(pairs) if pairs else "<none: deploymentSpec=" + json.dumps(spec) + ">")')"
  echo "  engine env: $ENGINE_ENV"
  if [[ "$ENGINE_ENV" != *"SANDBOX_ENABLED="* ]]; then
    echo "error: SANDBOX_ENABLED did not reach the engine — the env file was not baked in." >&2
    echo "       The engine exists but is NOT a sandbox. Fix the deploy and re-run (it updates in place)." >&2
    exit 1
  fi
fi
if [[ -z "$ENGINE_ID" ]]; then
  echo
  echo "Record it so the next run updates instead of creating another engine:"
  echo "  echo 'SANDBOX_AGENT_ENGINE_ID=$DEPLOYED_ID' >> $ENV_FILE"
fi
exit "$status"

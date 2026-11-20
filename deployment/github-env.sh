#!/usr/bin/env bash
#
# github-env.sh — read this repo's GitHub Environment variables from a laptop
# script (D37, ub-query-agent#105). Sourced, not run.
#
# Per-environment values (engine ids, project, service names) live in the
# GitHub Environments `prod`, `stage` and `dev`, the same place the workflows
# read them. Laptop scripts read them from there too, so no script carries a
# literal copy that could drift.
#
#   source deployment/github-env.sh
#   gh_env_var prod AGENT_ENGINE_ID      # prints the value, or nothing
#   prod_engine_id                       # prints prod's engine id, or exits
#
# Needs `gh` logged in with read access to the repo. An explicit environment
# variable of the same name always wins (e.g. PROD_AGENT_ENGINE_ID=...).

GH_REPO_SLUG="${GH_REPO_SLUG:-bpriddy/ub-query-agent}"

gh_env_var() {
  local env="$1" name="$2"
  command -v gh >/dev/null 2>&1 || return 0
  gh api "repos/$GH_REPO_SLUG/environments/$env/variables/$name" --jq .value 2>/dev/null || true
}

# The production engine id (Chat bot + Gemini Enterprise). Every sandbox
# script refuses to touch it, so it must resolve: fail closed if it cannot.
prod_engine_id() {
  local id="${PROD_AGENT_ENGINE_ID:-$(gh_env_var prod AGENT_ENGINE_ID)}"
  if [[ ! "$id" =~ ^[0-9]+$ ]]; then
    echo "error: cannot resolve the PRODUCTION engine id, so the guard cannot run." >&2
    echo "       Log in with gh (it reads AGENT_ENGINE_ID from GitHub Environment prod)," >&2
    echo "       or set PROD_AGENT_ENGINE_ID." >&2
    exit 1
  fi
  printf '%s' "$id"
}

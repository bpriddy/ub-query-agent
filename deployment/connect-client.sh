#!/usr/bin/env bash
#
# connect-client.sh — which Google OAuth client the query service mounts for
# the connect page and Workspace (#93, #90). Sourced, not run, by
# deploy-service.sh and .github/workflows/deploy-query-service.yml, so both
# resolve the same Secret Manager NAMES. Values are never read here.
#
# Inputs (from the GitHub Environment):
#   CONNECT_CLIENT   QUERY_SERVICE_CONNECT_CLIENT:
#                      unset/none  no client: the page answers
#                                  CONNECT_NOT_CONFIGURED (today's default)
#                      ub          UB's own client, one per environment:
#                                  ub-connect-google-client-id and
#                                  ub-connect-google-client-secret in this
#                                  environment's project (UB terraform
#                                  query_agent.tf makes the shells and the
#                                  runtime account's access)
#                      bot         FALLBACK ONLY, prod only: the sunsetting chat
#                                  bot's client (gub-gchat-bot-oauth-client-
#                                  {id,secret}-dev in os-test-491819). Needs a
#                                  secretAccessor grant on both for the runtime
#                                  account and the UB callback registered on
#                                  the bot's client, neither of which exists.
#   GOOGLE_CLIENT_ID_SECRET, GOOGLE_CLIENT_SECRET_SECRET
#                    QUERY_SERVICE_GOOGLE_CLIENT_{ID,SECRET}_SECRET: explicit
#                    secret names. When set they win over CONNECT_CLIENT.
#   PROJECT          the environment's project id.
#
# Output: GOOGLE_CLIENT_ID_SECRET and GOOGLE_CLIENT_SECRET_SECRET set to the
# two names, or both empty. Returns non-zero with a message on bad input.

CONNECT_UB_CLIENT_ID_SECRET="ub-connect-google-client-id"
CONNECT_UB_CLIENT_SECRET_SECRET="ub-connect-google-client-secret"
CONNECT_BOT_PROJECT="os-test-491819"
CONNECT_BOT_CLIENT_ID_SECRET="gub-gchat-bot-oauth-client-id-dev"
CONNECT_BOT_CLIENT_SECRET_SECRET="gub-gchat-bot-oauth-client-secret-dev"

resolve_connect_client() {
  if [[ -n "${GOOGLE_CLIENT_ID_SECRET:-}" || -n "${GOOGLE_CLIENT_SECRET_SECRET:-}" ]]; then
    if [[ -z "${GOOGLE_CLIENT_ID_SECRET:-}" || -z "${GOOGLE_CLIENT_SECRET_SECRET:-}" ]]; then
      echo "error: set both QUERY_SERVICE_GOOGLE_CLIENT_ID_SECRET and QUERY_SERVICE_GOOGLE_CLIENT_SECRET_SECRET, or neither" >&2
      return 1
    fi
    return 0
  fi
  case "${CONNECT_CLIENT:-none}" in
    none)
      GOOGLE_CLIENT_ID_SECRET=""
      GOOGLE_CLIENT_SECRET_SECRET=""
      ;;
    ub)
      GOOGLE_CLIENT_ID_SECRET="$CONNECT_UB_CLIENT_ID_SECRET"
      GOOGLE_CLIENT_SECRET_SECRET="$CONNECT_UB_CLIENT_SECRET_SECRET"
      ;;
    bot)
      if [[ "${PROJECT:-}" != "$CONNECT_BOT_PROJECT" ]]; then
        echo "error: QUERY_SERVICE_CONNECT_CLIENT=bot exists only in $CONNECT_BOT_PROJECT (got ${PROJECT:-none})" >&2
        return 1
      fi
      GOOGLE_CLIENT_ID_SECRET="$CONNECT_BOT_CLIENT_ID_SECRET"
      GOOGLE_CLIENT_SECRET_SECRET="$CONNECT_BOT_CLIENT_SECRET_SECRET"
      ;;
    *)
      echo "error: QUERY_SERVICE_CONNECT_CLIENT must be none, ub or bot (got $CONNECT_CLIENT)" >&2
      return 1
      ;;
  esac
}

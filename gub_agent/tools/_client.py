"""
_client.py — Authenticated HTTP client for the GUB backend.

Handles two auth modes:
  1. Gemini Enterprise runtime: Google OAuth access token injected into
     ToolContext by the platform → exchanged for a GUB JWT on first call,
     then cached for the session.
  2. Local dev / CI: GUB_SERVICE_JWT env var used directly (no exchange needed).

All GUB API calls go through gub_get() / gub_post() which pick the right mode
automatically. Tool functions never need to think about auth.

Every request — token exchange included — goes through one shared
httpx.AsyncClient so connections are pooled and reused instead of being
torn down per call, and the event loop is never blocked while a request
is in flight. Timeout and pool tuning live in TIMEOUT / LIMITS below.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..config import GUB_AUTHORIZATION_ID, GUB_BASE_URL, GUB_SERVICE_JWT
from ..tenant import require_app_scope

logger = logging.getLogger(__name__)

# Returned to the model as the tool result, which it can quote into its answer:
# a fixed, system-neutral sentence. The exception text (hostnames, TLS detail)
# stays in the logger.error line next to each use and never reaches the model.
_UNREACHABLE_MESSAGE = "Could not reach company records right now."

#: The caller's GUB JWT for one turn. ADK applies `temp:` keys for the turn and
#: strips them before the session is saved (gcp-universal-backend#162).
TURN_JWT_STATE_KEY = "temp:gub_jwt"

# ── Shared client ─────────────────────────────────────────────────────────────

TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)
LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=100)

_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared client, creating it lazily on first use.

    Keyed on the running loop: a client's pool is bound to the loop that
    opened its connections, and `is_closed` stays False when a loop is
    discarded without `aclose_client()`.
    """
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(http2=True, timeout=TIMEOUT, limits=LIMITS)
        _client_loop = loop
    return _client


async def aclose_client() -> None:
    """Close the shared client. No runtime caller today — tests use it per-run;
    a graceful-shutdown hook can adopt it if the runtime ever exposes one."""
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _client_loop = None


# ── Token resolution ──────────────────────────────────────────────────────────


async def _resolve_gub_jwt(tool_context: Any | None) -> str:
    """
    Resolve a valid GUB JWT for this request.

    Priority:
      0. The caller's GUB JWT for this turn only (`temp:gub_jwt`, set by the
         query agent service, #92): never saved, never cached
      1. Cached GUB JWT in ToolContext session state (avoids re-exchange per call)
      2. Google OAuth access token injected by Gemini Enterprise → exchange for GUB JWT
      3. GUB_SERVICE_JWT env var (local dev / CI fallback)

    The token exchange is itself a GUB call, so a tenant turn without a scope
    is refused here too, before it (the tools also warm this cache directly).
    """
    if tool_context is not None:
        require_app_scope(_state_of_tool_context(tool_context))
        # 0. Passed for this turn. Returned as is: writing it to `gub_jwt`
        #    would save it with the session.
        turn_jwt = tool_context.state.get(TURN_JWT_STATE_KEY)
        if turn_jwt and isinstance(turn_jwt, str):
            return turn_jwt
        # 1. Cached from a previous tool call in this session
        cached = tool_context.state.get("gub_jwt")
        if cached:
            return cached

        # Log available state KEY NAMES for debugging — never values: the
        # state can hold the user's Google access token, and values must not
        # reach Cloud Logging.
        try:
            state_dict = tool_context.state.to_dict()
            logger.info("auth debug — state keys: %s", sorted(state_dict.keys()))
        except Exception as e:
            logger.info("auth debug — state.to_dict() failed: %s", e)
        try:
            app_prefix = getattr(tool_context.state, "APP_PREFIX", None)
            temp_prefix = getattr(tool_context.state, "TEMP_PREFIX", None)
            logger.info("auth debug — APP_PREFIX=%s TEMP_PREFIX=%s", app_prefix, temp_prefix)
        except Exception:
            pass

        # 2a. Try the direct state key pattern (ADK article pattern)
        #     tool_context.state[auth_id] → access token string
        #     Use to_dict() since State.get() may prepend a prefix
        try:
            state_dict = tool_context.state.to_dict()
            google_access_token = state_dict.get(GUB_AUTHORIZATION_ID)
            if google_access_token and isinstance(google_access_token, str):
                logger.warning("Found OAuth token via to_dict()['%s']", GUB_AUTHORIZATION_ID)
                gub_jwt = await _exchange_google_token(google_access_token)
                tool_context.state["gub_jwt"] = gub_jwt
                return gub_jwt
            elif google_access_token:
                # Type only — the value may be a token-bearing structure.
                logger.warning(
                    "State key '%s' exists but is type %s (expected str)",
                    GUB_AUTHORIZATION_ID,
                    type(google_access_token).__name__,
                )
        except Exception as e:
            logger.warning("Error reading state via to_dict: %s", e)

        # 2b. Try the nested auth_tokens pattern
        try:
            auth_tokens = tool_context.state.get("auth_tokens") or {}
            gub_auth = auth_tokens.get(GUB_AUTHORIZATION_ID) or {}
            google_access_token = (gub_auth.get("token") or {}).get("access_token")
            if google_access_token:
                logger.info("Found OAuth token via auth_tokens['%s']", GUB_AUTHORIZATION_ID)
                gub_jwt = await _exchange_google_token(google_access_token)
                tool_context.state["gub_jwt"] = gub_jwt
                return gub_jwt
        except Exception as e:
            logger.info("Error reading auth_tokens: %s", e)

        # 2c. Try temp: prefix pattern (older ADK versions)
        try:
            temp_token = tool_context.state.get(f"temp:{GUB_AUTHORIZATION_ID}")
            if temp_token and isinstance(temp_token, str):
                logger.info("Found OAuth token via temp:%s", GUB_AUTHORIZATION_ID)
                gub_jwt = await _exchange_google_token(temp_token)
                tool_context.state["gub_jwt"] = gub_jwt
                return gub_jwt
        except Exception as e:
            logger.info("Error reading temp state: %s", e)

        logger.warning("No OAuth token found in tool_context.state for '%s'", GUB_AUTHORIZATION_ID)

    # 3. Local dev / service account fallback
    if GUB_SERVICE_JWT:
        return GUB_SERVICE_JWT

    raise RuntimeError(
        "No GUB JWT available. Set GUB_SERVICE_JWT for local dev, or ensure "
        f"Gemini Enterprise is injecting an OAuth token under authorization ID "
        f"'{GUB_AUTHORIZATION_ID}'."
    )


async def _exchange_google_token(google_access_token: str) -> str:
    """
    Exchange a Google OAuth access token for a GUB JWT via the broker endpoint.
    GUB verifies the token with Google's userinfo endpoint, then issues its own JWT.
    """
    client = _get_client()
    resp = await client.post(
        f"{GUB_BASE_URL}/auth/google/access-token-exchange",
        json={"accessToken": google_access_token},
    )
    if not resp.is_success:
        logger.error("GUB token exchange failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"GUB token exchange failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    access_token = data.get("accessToken")
    if not access_token:
        raise RuntimeError("GUB token exchange response missing 'accessToken'")
    return access_token


# ── HTTP helpers ──────────────────────────────────────────────────────────────


# ── App scope (UBScope, gub #119) ─────────────────────────────────────────────
#
# A tenant bot declares which records its surface covers and puts that in
# session state as `app_scope`: a STRING, the bot's serialized JSON. This
# module forwards that string as the `X-App-Scope` request header and does
# nothing else with it. Never parsed, rebuilt, re-serialised or defaulted here —
# the backend owns interpretation, and a re-serialisation would change key
# order or escapes and make the agent's idea of the scope drift from the bot's.
#
# A tenant turn (`state["tenant"]` set) with no scope raises before any request
# (`tenant.require_app_scope`): the fast path runs no instruction provider, so
# the provider's own check cannot be the only door.
#
# Read-only by structure: no tool takes a scope parameter, nothing in this
# package writes `app_scope`, and `_exchange_google_token` (the broker
# exchange, not a records read) never carries the header.

APP_SCOPE_STATE_KEY = "app_scope"
APP_SCOPE_HEADER = "X-App-Scope"


def _state_of_tool_context(tool_context: Any | None) -> Any:
    return getattr(tool_context, "state", None)


def _scope_headers(state: Any) -> dict[str, str]:
    """`{"X-App-Scope": <state value>}`, or `{}` when the state carries none.

    The value goes out exactly as stored. A non-string value is not forwarded:
    it cannot be sent byte for byte, and guessing a serialisation is the one
    thing this function must never do.
    """
    if state is None:
        return {}
    try:
        raw = state.get(APP_SCOPE_STATE_KEY)
    except AttributeError:
        return {}
    if isinstance(raw, str):
        return {APP_SCOPE_HEADER: raw}
    return {}


# ── Account scope (tenant-10) ─────────────────────────────────────────────────
#
# A tenant bot's GUB session is scoped to one account's records. When the
# scope removes rows, the backend says so on the response — a header, because
# several org endpoints answer with a bare JSON array and an array has nowhere
# to put a field.
#
# That signal has to reach the model, and this is the ONE place all 12 tools
# funnel through. Without it a filtered read is indistinguishable from an
# empty one, and a model being truthful about what it got says "Budweiser has
# no campaigns" — a claim about the world built from a fact about permissions,
# and worse than a refusal.

SCOPE_HEADER = "x-account-scope-filtered"

ACCOUNT_SCOPE_NOTICE = (
    "Some records matching this request are outside this assistant's account "
    "scope and were not returned. Treat them as UNKNOWN to you: do not report "
    "them as non-existent, do not say the user lacks access, and do not "
    "invent them. Do NOT write a sentence about the scope either — the "
    "surface tells the user that itself, in one line, and a second telling "
    "reads as a stutter. If everything you would have answered from is "
    "affected, abstain instead of writing a placeholder answer."
)

# Where to hang a filtered bare-array payload so the shape stays meaningful.
# Only the first three can actually be filtered — staff and resourcing are
# unscoped by ruling — but the map is complete so a future scope does not
# silently fall back to a key nothing reads.
_ARRAY_KEYS = {
    "/org/search": "hits",
    # `tools/files.py` normalises the UNfiltered bare array to the same key, so
    # a scoped file turn and an unscoped one hand the model the same shape.
    "/org/files/search": "files",
    "/org/accounts": "accounts",
    "/org/staff": "staff",
    "/org/resourcing": "people",
}


def _array_key(path: str) -> str:
    return _ARRAY_KEYS.get(path.split("?")[0].rstrip("/"), "results")


def _with_scope_notice(payload: Any, resp: httpx.Response, path: str) -> Any:
    """Attach the out-of-scope notice when the backend flagged filtering.

    The key must not start with "_": underscore-prefixed keys are stripped
    from the model's view by context pruning and skipped by the evidence
    index, so a notice hidden behind one would be attached and never seen.

    A bare array is wrapped rather than left alone — the evidence index
    drops non-dict payloads entirely, so an array cannot carry a notice at
    all. The wrapper key is the one that endpoint's readers already use.
    """
    if resp.headers.get(SCOPE_HEADER) != "1":
        return payload

    if isinstance(payload, dict):
        payload["account_scope_notice"] = ACCOUNT_SCOPE_NOTICE
        return payload
    if isinstance(payload, list):
        return {_array_key(path): payload, "account_scope_notice": ACCOUNT_SCOPE_NOTICE}
    return payload


async def gub_get(
    path: str,
    tool_context: Any | None = None,
    **params: Any,
) -> dict:
    """
    Make an authenticated GET request to the GUB backend.

    Non-None query params are forwarded as URL query parameters.
    Returns the parsed JSON response, or an error dict on HTTP failure.
    """
    state = _state_of_tool_context(tool_context)
    require_app_scope(state)  # before ANY GUB call, the token exchange included
    scope_headers = _scope_headers(state)
    jwt = await _resolve_gub_jwt(tool_context)
    clean_params = {k: v for k, v in params.items() if v is not None}

    client = _get_client()
    try:
        resp = await client.get(
            f"{GUB_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {jwt}", **scope_headers},
            params=clean_params,
        )
        resp.raise_for_status()
        return _with_scope_notice(resp.json(), resp, path)
    except httpx.HTTPStatusError as exc:
        logger.warning("GUB GET %s returned %s", path, exc.response.status_code)
        return _error_dict(exc.response, path)
    except httpx.RequestError as exc:
        logger.error("GUB GET %s request error: %s", path, exc)
        return {"error": True, "status": 0, "message": _UNREACHABLE_MESSAGE}


async def gub_post(
    path: str,
    body: dict,
    tool_context: Any | None = None,
) -> dict:
    """
    Make an authenticated POST request to the GUB backend.
    Returns the parsed JSON response, or an error dict on HTTP failure.
    """
    state = _state_of_tool_context(tool_context)
    require_app_scope(state)  # before ANY GUB call, the token exchange included
    scope_headers = _scope_headers(state)
    jwt = await _resolve_gub_jwt(tool_context)

    client = _get_client()
    try:
        resp = await client.post(
            f"{GUB_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {jwt}", **scope_headers},
            json=body,
        )
        resp.raise_for_status()
        return _with_scope_notice(resp.json(), resp, path)
    except httpx.HTTPStatusError as exc:
        logger.warning("GUB POST %s returned %s", path, exc.response.status_code)
        return _error_dict(exc.response, path)
    except httpx.RequestError as exc:
        logger.error("GUB POST %s request error: %s", path, exc)
        return {"error": True, "status": 0, "message": _UNREACHABLE_MESSAGE}


def _error_dict(resp: httpx.Response, path: str) -> dict:
    """A failed call as the tools return it. UB's own error `code` is carried
    when its body has one (e.g. `RUNN_NO_GRANT`), so a caller can tell a
    missing grant from a missing record; the message stays the friendly one."""
    out: dict[str, Any] = {
        "error": True,
        "status": resp.status_code,
        "message": _friendly_error(resp.status_code, path),
    }
    try:
        body = resp.json()
    except ValueError:
        return out
    if isinstance(body, dict) and isinstance(body.get("code"), str):
        out["code"] = body["code"]
    return out


def _friendly_error(status: int, path: str) -> str:
    if status == 401:
        return "Authentication failed — your session may have expired."
    if status == 403:
        return f"Access denied to {path}. The user may not have permission to view this data."
    if status == 404:
        return f"Resource not found at {path}."
    return f"Company records returned an unexpected error ({status}) for {path}."

"""
engine.py — the Agent Engine, over its REST data plane.

`create_session` makes the engine session a new conversation runs in;
`stream_turn` runs one turn and yields the engine's events as dicts.

A turn goes through `async_stream_query`, which hands `state_delta` to ADK's
`runner.run_async`; `temp:` keys in it are applied for the turn and stripped
before the session is saved (gcp-universal-backend#162, checked live on the
sandbox engine 2026-11-09).

The engine answers 200 even when the run dies inside it, so an empty stream
is a failure signal, and the caller treats "no answer event" as one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .errors import ServiceError
from .gcp import ServiceToken

logger = logging.getLogger(__name__)

#: The caller's UB token for the turn (read by gub_agent/tools/_client.py).
UB_TOKEN_KEY = "temp:gub_jwt"
#: The person's Google access token for the turn, when they have a grant (#90).
WORKSPACE_TOKEN_KEY = "temp:workspace_token"

TURN_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def _parse_line(line: str) -> dict[str, Any] | None:
    """One event from a stream line: SSE (`data: {...}`) or bare JSON."""
    line = line.strip()
    if not line or line.startswith((":", "event:", "id:", "retry:")):
        return None
    if line.startswith("data:"):
        line = line[5:].strip()
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("engine.stream: unparseable line (%d chars)", len(line))
        return None
    return event if isinstance(event, dict) else None


class Engine:
    def __init__(
        self, resource: str, region: str, http: httpx.AsyncClient, token: ServiceToken
    ) -> None:
        self._base = f"https://{region}-aiplatform.googleapis.com/v1/{resource}"
        self._http = http
        self._token = token

    async def create_session(self, user_id: str) -> str:
        resp = await self._http.post(
            f"{self._base}:query",
            headers=await self._token.headers(),
            json={"class_method": "create_session", "input": {"user_id": user_id}},
        )
        if resp.status_code != 200:
            logger.error("engine.create_session failed: status=%s", resp.status_code)
            raise ServiceError(
                502, "ENGINE_UNAVAILABLE", "The query agent could not start a conversation."
            )
        output = resp.json().get("output") or {}
        session_id = output.get("id") or output.get("session_id")
        if not session_id:
            raise ServiceError(
                502, "ENGINE_UNAVAILABLE", "The query agent could not start a conversation."
            )
        return str(session_id)

    async def stream_turn(
        self,
        user_id: str,
        session_id: str,
        message: str,
        state_delta: dict[str, str],
    ) -> AsyncIterator[dict[str, Any]]:
        body = {
            "class_method": "async_stream_query",
            "input": {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "state_delta": state_delta,
            },
        }
        async with self._http.stream(
            "POST",
            f"{self._base}:streamQuery",
            params={"alt": "sse"},
            headers=await self._token.headers(),
            json=body,
            timeout=TURN_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                logger.error("engine.stream failed: status=%s", resp.status_code)
                raise ServiceError(
                    502, "ENGINE_UNAVAILABLE", "The query agent could not be reached."
                )
            async for line in resp.aiter_lines():
                event = _parse_line(line)
                if event is not None:
                    yield event

"""
gcp.py — calls to Google APIs as the service's own account.

One shared httpx client, and the account's access token from Application
Default Credentials, refreshed off the event loop when it nears expiry.
"""

from __future__ import annotations

import asyncio

import google.auth
import google.auth.transport.requests
import httpx

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class ServiceToken:
    def __init__(self) -> None:
        self._creds = None
        self._lock = asyncio.Lock()

    def _refresh(self) -> str:
        if self._creds is None:
            self._creds, _ = google.auth.default(scopes=SCOPES)
        if not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())
        return self._creds.token

    async def headers(self) -> dict[str, str]:
        async with self._lock:
            token = await asyncio.to_thread(self._refresh)
        return {"Authorization": f"Bearer {token}"}


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

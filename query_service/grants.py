"""
grants.py — each person's Gemini Enterprise grant, and the per-turn token.

A grant is one Google refresh token per person, issued to the Gemini
Enterprise project's client for one scope, and reused by every app in the
collection (#93 writes it). It is stored encrypted with this service's KMS
key; the service account can encrypt and decrypt with that key and nothing
else.

Per turn, `mint_access_token` turns the grant into a short-lived access
token for the engine's `temp:` state. The access token is never stored.
"""

from __future__ import annotations

import base64
import logging
from typing import Protocol

import asyncpg
import httpx

from .config import Settings
from .gcp import ServiceToken

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
KMS_URL = "https://cloudkms.googleapis.com/v1"


class Cipher(Protocol):
    async def encrypt(self, plaintext: bytes) -> bytes: ...
    async def decrypt(self, ciphertext: bytes) -> bytes: ...


class KmsCipher:
    """Cloud KMS symmetric encrypt / decrypt over REST."""

    def __init__(self, key: str, http: httpx.AsyncClient, token: ServiceToken) -> None:
        self._key = key
        self._http = http
        self._token = token

    async def _call(self, verb: str, field: str, value: bytes, out: str) -> bytes:
        resp = await self._http.post(
            f"{KMS_URL}/{self._key}:{verb}",
            headers=await self._token.headers(),
            json={field: base64.b64encode(value).decode()},
        )
        resp.raise_for_status()
        return base64.b64decode(resp.json()[out])

    async def encrypt(self, plaintext: bytes) -> bytes:
        return await self._call("encrypt", "plaintext", plaintext, "ciphertext")

    async def decrypt(self, ciphertext: bytes) -> bytes:
        return await self._call("decrypt", "ciphertext", ciphertext, "plaintext")


async def put_grant(
    pool: asyncpg.Pool,
    cipher: Cipher,
    person_id: str,
    refresh_token: str,
    scopes: list[str],
    client_id: str,
) -> None:
    """Store (or replace) the person's grant. A new grant clears the reuse
    notices: the person connected again, so each app tells them again."""
    ciphertext = await cipher.encrypt(refresh_token.encode())
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("DELETE FROM workspace_grants WHERE person_id = $1", person_id)
        await conn.execute(
            "INSERT INTO workspace_grants"
            " (person_id, refresh_token_ciphertext, scopes, client_id)"
            " VALUES ($1, $2, $3, $4)",
            person_id,
            ciphertext,
            scopes,
            client_id,
        )


async def delete_grant(pool: asyncpg.Pool, person_id: str) -> bool:
    status = await pool.execute("DELETE FROM workspace_grants WHERE person_id = $1", person_id)
    return status.endswith(" 1")


async def mint_access_token(
    pool: asyncpg.Pool,
    cipher: Cipher,
    http: httpx.AsyncClient,
    settings: Settings,
    person_id: str,
) -> str | None:
    """A Google access token for this turn from the person's grant; None when
    they have no grant, the service has no client configured, or Google
    refuses (the grant was revoked). The reason goes to the log, never the
    token or Google's response body."""
    if not (settings.google_client_id and settings.google_client_secret):
        return None
    row = await pool.fetchrow(
        "SELECT refresh_token_ciphertext, client_id FROM workspace_grants WHERE person_id = $1",
        person_id,
    )
    if row is None:
        return None
    if row["client_id"] != settings.google_client_id:
        logger.warning("grant.mint skipped: grant issued to another client")
        return None
    refresh_token = (await cipher.decrypt(row["refresh_token_ciphertext"])).decode()
    resp = await http.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
        },
    )
    if resp.status_code != 200:
        error = (
            resp.json().get("error")
            if resp.headers.get("content-type", "").startswith("application/json")
            else None
        )
        logger.warning("grant.mint failed: status=%s error=%s", resp.status_code, error)
        return None
    return resp.json().get("access_token")

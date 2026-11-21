"""
grants.py — each person's Gemini Enterprise grant, and the per-turn token.

A grant is one Google refresh token per person, issued to the Gemini
Enterprise project's client for one scope, and reused by every app in the
collection (#93 writes it). It is stored encrypted with this service's KMS
key; the service account can encrypt and decrypt with that key and nothing
else.

Per turn, `mint_access_token` turns the grant into a short-lived access
token for the engine's `temp:` state. The access token is never stored.

`revoke_grant` is disconnect: Google revokes the refresh token, then the row
goes. `claim_reuse_notice` decides the one-time "Using your existing Gemini
Enterprise connection" line (#93).
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
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
KMS_URL = "https://cloudkms.googleapis.com/v1"


class Cipher(Protocol):
    """`aad` binds a ciphertext to its use: one made for a cookie never
    decrypts as a grant. Grants use none, as they always have."""

    async def encrypt(self, plaintext: bytes, aad: bytes = b"") -> bytes: ...
    async def decrypt(self, ciphertext: bytes, aad: bytes = b"") -> bytes: ...


class KmsCipher:
    """Cloud KMS symmetric encrypt / decrypt over REST. KMS ciphertexts are
    authenticated, so a changed one fails to decrypt."""

    def __init__(self, key: str, http: httpx.AsyncClient, token: ServiceToken) -> None:
        self._key = key
        self._http = http
        self._token = token

    async def _call(self, verb: str, field: str, value: bytes, out: str, aad: bytes) -> bytes:
        body = {field: base64.b64encode(value).decode()}
        if aad:
            body["additionalAuthenticatedData"] = base64.b64encode(aad).decode()
        resp = await self._http.post(
            f"{KMS_URL}/{self._key}:{verb}",
            headers=await self._token.headers(),
            json=body,
        )
        resp.raise_for_status()
        return base64.b64decode(resp.json()[out])

    async def encrypt(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        return await self._call("encrypt", "plaintext", plaintext, "ciphertext", aad)

    async def decrypt(self, ciphertext: bytes, aad: bytes = b"") -> bytes:
        return await self._call("decrypt", "ciphertext", ciphertext, "plaintext", aad)


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


async def grant_status(pool: asyncpg.Pool, person_id: str) -> dict | None:
    """What the person may see about their own grant: when, and for what.
    Never the token."""
    row = await pool.fetchrow(
        "SELECT scopes, connected_at FROM workspace_grants WHERE person_id = $1", person_id
    )
    if row is None:
        return None
    return {"scopes": list(row["scopes"]), "connected_at": row["connected_at"].isoformat()}


async def revoke_grant(
    pool: asyncpg.Pool, cipher: Cipher, http: httpx.AsyncClient, person_id: str
) -> bool:
    """Disconnect: ask Google to revoke the refresh token (which also ends
    every access token minted from it), then delete the row. The row goes
    even when Google cannot be reached or already forgot the token: the
    person asked to disconnect, and without the row nothing is minted again.
    Returns whether there was a grant."""
    row = await pool.fetchrow(
        "SELECT refresh_token_ciphertext FROM workspace_grants WHERE person_id = $1", person_id
    )
    if row is None:
        return False
    try:
        refresh_token = (await cipher.decrypt(row["refresh_token_ciphertext"])).decode()
        resp = await http.post(REVOKE_URL, data={"token": refresh_token})
        if resp.status_code != 200:
            # 400 invalid_token: Google already revoked it (the person did so
            # in their account settings). Either way it is gone.
            logger.warning("grant.revoke: status=%s", resp.status_code)
    except Exception:  # noqa: BLE001 — the row is deleted whatever Google says
        logger.exception("grant.revoke failed; deleting the grant anyway")
    return await delete_grant(pool, person_id)


async def claim_reuse_notice(pool: asyncpg.Pool, person_id: str, app_id: str | None) -> bool:
    """Whether this answer tells the person their connection is being reused.

    The grant is one per person and serves every app in the collection. The
    first app to use it says nothing (the person just connected for it). The
    first time a second or later app uses it, that app says so once. Each app
    is recorded the first time it uses the grant, so it never says it again;
    a new grant (`put_grant`) clears the record. A token with no app (UB
    itself) is no app, and never shows it."""
    if app_id is None:
        return False
    async with pool.acquire() as conn, conn.transaction():
        # Serialise one person's apps, so two first uses at once cannot both
        # think they are first.
        locked = await conn.fetchval(
            "SELECT 1 FROM workspace_grants WHERE person_id = $1 FOR UPDATE", person_id
        )
        if locked is None:
            return False
        inserted = await conn.fetchval(
            "INSERT INTO workspace_grant_notices (person_id, app_id) VALUES ($1, $2)"
            " ON CONFLICT DO NOTHING RETURNING 1",
            person_id,
            app_id,
        )
        if inserted is None:
            return False  # this app has already used it
        others = await conn.fetchval(
            "SELECT count(*) FROM workspace_grant_notices WHERE person_id = $1 AND app_id <> $2",
            person_id,
            app_id,
        )
        return others > 0


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

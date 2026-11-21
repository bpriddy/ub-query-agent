"""
Fixtures for the query agent service (query_service, #92).

The database is real Postgres: TEST_DATABASE_URL (CI runs a postgres:15
service container). Without it these tests skip. Each test gets a fresh
`public` schema with the migrations applied.

UB tokens are real RS256 JWTs signed with a key made here; the verifier
checks them with the real code path, only its key lookup is local. The
engine, the KMS cipher and Google's token endpoint are fakes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import asyncpg
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from query_service.app import create_app
from query_service.auth import UbTokenVerifier
from query_service.config import Settings
from query_service.db import migrate
from query_service.errors import ServiceError
from tests.unit.test_format_gate import _valid_dict

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

UB = "https://ub.test"
APP = "holocron"
CLIENT_ID = "client-1.apps.googleusercontent.com"
PUBLIC = "https://qa.test"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 — pytest hook signature
    if DATABASE_URL:
        return
    skip = pytest.mark.skip(reason="set TEST_DATABASE_URL to run the query service tests")
    for item in items:
        if "tests/service/" in str(item.fspath):
            item.add_marker(skip)


_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def ub_token(person: str = "person-1", app: str | None = APP, **over: Any) -> str:
    now = int(time.time())
    claims = {
        "sub": person,
        "iss": UB,
        "aud": [app, UB] if app else UB,
        "iat": now,
        "exp": now + 900,
        "email": f"{person}@anomaly.com",
        **over,
    }
    return jwt.encode(claims, _KEY, algorithm="RS256", headers={"kid": "k1"})


class _LocalKeys:
    def get_signing_key_from_jwt(self, _token: str) -> Any:
        return SimpleNamespace(key=_KEY.public_key())


def verifier() -> UbTokenVerifier:
    v = UbTokenVerifier(f"{UB}/auth/jwks", UB, UB)
    v._jwks = _LocalKeys()  # type: ignore[assignment]
    return v


def settings(**over: Any) -> Settings:
    base = {
        "engine": "projects/p/locations/us-central1/reasoningEngines/1",
        "ub_base_url": UB,
        "ub_issuer": UB,
        "ub_audience": UB,
        "database_url": DATABASE_URL,
        "cloud_sql_instance": None,
        "db_name": None,
        "db_iam_user": None,
        "grant_kms_key": "projects/p/locations/l/keyRings/r/cryptoKeys/k",
        "google_client_id": CLIENT_ID,
        "google_client_secret": "client-secret",
        "public_base_url": PUBLIC,
    }
    base.update(over)
    return Settings(**base)


def ask_event(meta: dict[str, Any]) -> dict[str, Any]:
    return {"author": "ask", "partial": True, "custom_metadata": meta}


def answer(**over: Any) -> dict[str, Any]:
    """A valid `AskResponse` as the `/ask` turn would stream it."""
    body = {
        **_valid_dict(),
        "sources": {s: {"status": "skipped"} for s in ("ub", "runn", "workspace")},
        "conversation_id": "engine-session-id",
    }
    body.update(over)
    return body


class FakeEngine:
    """Records what the service sent; plays back a scripted turn."""

    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.turns: list[dict[str, Any]] = []
        self.script: list[dict[str, Any]] = [
            ask_event({"progress": {"step": "understand", "text": "Reading the question"}}),
            {"author": "gub_agent", "content": {"parts": [{"text": "TOOL RESULT secret-row"}]}},
            ask_event({"ask_answer": answer()}),
        ]
        self.fail_with: Exception | None = None

    async def create_session(self, user_id: str) -> str:
        sid = f"engine-{len(self.sessions) + 1}"
        self.sessions.append(sid)
        return sid

    async def stream_turn(
        self, user_id: str, session_id: str, message: str, state_delta: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        self.turns.append(
            {"user_id": user_id, "session_id": session_id, "message": message, "state": state_delta}
        )
        for event in self.script:
            yield event
        if self.fail_with is not None:
            raise self.fail_with


class FakeCipher:
    """Reversible, never contains the plaintext, and like KMS refuses a
    ciphertext that was changed or made for another `aad`."""

    @staticmethod
    def _tag(body: bytes, aad: bytes) -> bytes:
        return hashlib.sha256(b"k" + aad + body).digest()[:8]

    async def encrypt(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        body = bytes(b ^ 0x5A for b in plaintext)
        return b"enc:" + self._tag(body, aad) + body

    async def decrypt(self, ciphertext: bytes, aad: bytes = b"") -> bytes:
        assert ciphertext.startswith(b"enc:")
        tag, body = ciphertext[4:12], ciphertext[12:]
        if tag != self._tag(body, aad):
            raise ValueError("decryption failed")
        return bytes(b ^ 0x5A for b in body)


ASSIST = "https://www.googleapis.com/auth/discoveryengine.assist.readwrite"


class GoogleTokenEndpoint:
    """Google's token and revoke endpoints, and UB's exchange and logout, as
    the service calls them. Records every call."""

    def __init__(self) -> None:
        self.status = 200
        self.calls: list[dict[str, str]] = []
        self.code_calls: list[dict[str, str]] = []
        self.revoked: list[str] = []
        self.revoke_status = 200
        self.ub_calls: list[tuple[str, dict[str, Any]]] = []
        self.ub_status = 200
        # What the authorization-code exchange returns.
        self.granted_scope = f"openid email {ASSIST}"
        self.refresh_token: str | None = "1//connected-refresh-token"
        self.person = "person-1"

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://oauth2.googleapis.com/token":
            form = dict(httpx.QueryParams(request.content.decode()))
            if form.get("grant_type") == "authorization_code":
                self.code_calls.append(form)
                if form["code"] != "good-code":
                    return httpx.Response(400, json={"error": "invalid_grant"})
                body = {"access_token": "ya29.consent-access-token", "scope": self.granted_scope}
                if self.refresh_token:
                    body["refresh_token"] = self.refresh_token
                return httpx.Response(200, json=body)
            self.calls.append(form)
            if self.status != 200:
                return httpx.Response(self.status, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "ya29.minted-google-token"})
        if url == "https://oauth2.googleapis.com/revoke":
            self.revoked.append(dict(httpx.QueryParams(request.content.decode()))["token"])
            return httpx.Response(self.revoke_status)
        if url == f"{UB}/auth/google/access-token-exchange":
            body = json.loads(request.content)
            self.ub_calls.append(("exchange", body))
            if self.ub_status != 200:
                return httpx.Response(self.ub_status, json={"code": "UNAUTHORIZED"})
            return httpx.Response(
                200,
                json={
                    "accessToken": ub_token(person=self.person, app=None),
                    "refreshToken": "ub-refresh-1",
                },
            )
        if url == f"{UB}/auth/logout":
            self.ub_calls.append(("logout", json.loads(request.content)))
            return httpx.Response(204)
        raise AssertionError(f"unexpected call: {request.method} {url}")


@pytest.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    p = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    async with p.acquire() as conn:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    await migrate(p)
    yield p
    await p.close()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def google() -> GoogleTokenEndpoint:
    return GoogleTokenEndpoint()


@pytest.fixture
async def service(pool, engine, google) -> AsyncIterator[SimpleNamespace]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(google.handler))
    app = create_app(
        settings(),
        engine=engine,  # type: ignore[arg-type]
        cipher=FakeCipher(),
        verifier=verifier(),
        pool=pool,
        http=http,
    )
    async with app.router.lifespan_context(app):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC)
        yield SimpleNamespace(client=client, app=app, pool=pool, engine=engine, google=google)
        await client.aclose()
    await http.aclose()


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def ask(svc, token: str | None, **body: Any) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return await svc.client.post(
        "/ask", headers=headers, json={"question": "how is chevy?", **body}
    )


__all__ = ["ServiceError"]

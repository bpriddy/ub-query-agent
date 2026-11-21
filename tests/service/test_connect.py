"""
The connect page (ub-query-agent#93).

Pinned here:

  * start sends the person to Google with the existing client, the assist
    scope, offline access and a state bound to a sealed cookie;
  * the callback refuses a missing, changed or foreign state, stores the grant
    under the person UB names (sign in with UB), records only the assist
    scope, and closes the UB session it opened;
  * a consent without the assist scope or without a refresh token stores
    nothing;
  * "manage" signs in without asking for the grant and changes nothing;
  * disconnect (page or API) revokes the token at Google and deletes the
    grant, even when Google refuses; a cross-site POST is refused;
  * the reuse notice: none in the first app, once in a second app, never
    again there, and again after a reconnect;
  * WORKSPACE_NOT_CONNECTED carries this service's connect link;
  * without the client, the public URL or the key, the page says it is not
    set up.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from query_service.app import create_app
from query_service.connect import ASSIST_SCOPE, STATE_COOKIE
from query_service.grants import put_grant

from .conftest import (
    CLIENT_ID,
    PUBLIC,
    FakeCipher,
    answer,
    ask,
    ask_event,
    parse_sse,
    settings,
    ub_token,
    verifier,
)


async def _start(service, intent: str | None = None) -> dict[str, list[str]]:
    url = "/connect/start" + (f"?intent={intent}" if intent else "")
    resp = await service.client.get(url)
    assert resp.status_code == 302
    assert STATE_COOKIE in resp.cookies
    return parse_qs(urlsplit(resp.headers["location"]).query)


async def _connect(service, code: str = "good-code") -> httpx.Response:
    query = await _start(service)
    return await service.client.get(
        "/connect/callback", params={"code": code, "state": query["state"][0]}
    )


async def _grant(service, person: str = "person-1"):
    return await service.pool.fetchrow(
        "SELECT * FROM workspace_grants WHERE person_id = $1", person
    )


# ── start ────────────────────────────────────────────────────────────────────


async def test_start_asks_google_for_the_assist_scope_with_the_existing_client(service):
    resp = await service.client.get("/connect/start")
    target = urlsplit(resp.headers["location"])
    query = parse_qs(target.query)

    assert f"{target.scheme}://{target.netloc}{target.path}" == (
        "https://accounts.google.com/o/oauth2/v2/auth"
    )
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"] == [f"{PUBLIC}/connect/callback"]
    assert query["scope"][0].split() == ["openid", "email", ASSIST_SCOPE]
    assert query["access_type"] == ["offline"] and query["prompt"] == ["consent"]
    assert "include_granted_scopes" not in query
    # The state is not readable from the cookie: it is sealed.
    assert query["state"][0] not in resp.cookies[STATE_COOKIE]
    cookie = resp.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
    assert "path=/connect" in cookie


async def test_manage_signs_in_without_asking_for_the_grant(service):
    query = await _start(service, "manage")
    assert ASSIST_SCOPE not in query["scope"][0]
    assert "access_type" not in query


# ── the callback ─────────────────────────────────────────────────────────────


async def test_a_consent_stores_the_grant_under_the_person_ub_names(service):
    service.google.person = "person-ub-7"
    resp = await _connect(service)

    assert resp.status_code == 303
    assert resp.headers["location"] == f"{PUBLIC}/connect?result=connected"
    row = await _grant(service, "person-ub-7")
    assert row["scopes"] == [ASSIST_SCOPE]
    assert row["client_id"] == CLIENT_ID
    assert b"connected-refresh-token" not in row["refresh_token_ciphertext"]
    # The code went to Google with the client and the same redirect URI.
    [code_call] = service.google.code_calls
    assert code_call["client_id"] == CLIENT_ID
    assert code_call["redirect_uri"] == f"{PUBLIC}/connect/callback"
    # Sign in with UB: UB got the Google token, and its session was closed.
    assert service.google.ub_calls == [
        ("exchange", {"accessToken": "ya29.consent-access-token"}),
        ("logout", {"refreshToken": "ub-refresh-1"}),
    ]
    # The page now shows the person connected, with Disconnect.
    page = await service.client.get("/connect?result=connected")
    assert "Connected since" in page.text and "Disconnect" in page.text
    assert "person-ub-7@anomaly.com" in page.text


@pytest.mark.parametrize("state", [None, "not-the-state"])
async def test_a_missing_or_wrong_state_is_refused(service, state):
    await _start(service)
    params = {"code": "good-code"} | ({"state": state} if state else {})
    resp = await service.client.get("/connect/callback", params=params)
    assert resp.headers["location"].endswith("result=expired")
    assert service.google.code_calls == []
    assert await _grant(service) is None


async def test_a_forged_or_foreign_state_cookie_is_refused(service):
    query = await _start(service)
    state = query["state"][0]
    # A cookie made for another purpose (the session cookie's aad) or changed.
    for bad in ["garbage", service.client.cookies[STATE_COOKIE][:-2] + "AA"]:
        service.client.cookies.set(STATE_COOKIE, bad, domain="qa.test", path="/connect")
        resp = await service.client.get(
            "/connect/callback", params={"code": "good-code", "state": state}
        )
        assert resp.headers["location"].endswith("result=expired")
    assert service.google.code_calls == []


async def test_a_cancelled_consent_stores_nothing(service):
    query = await _start(service)
    resp = await service.client.get(
        "/connect/callback", params={"error": "access_denied", "state": query["state"][0]}
    )
    assert resp.headers["location"].endswith("result=denied")
    assert await _grant(service) is None


@pytest.mark.parametrize("missing", ["scope", "refresh_token"])
async def test_a_consent_without_the_assist_scope_or_a_refresh_token_stores_nothing(
    service, missing
):
    if missing == "scope":
        service.google.granted_scope = "openid email"
    else:
        service.google.refresh_token = None
    resp = await _connect(service)
    assert resp.headers["location"].endswith("result=incomplete")
    assert await _grant(service) is None


async def test_ub_refusing_the_person_stores_nothing(service):
    service.google.ub_status = 403
    resp = await _connect(service)
    assert resp.headers["location"].endswith("result=failed")
    assert await _grant(service) is None


async def test_a_bad_code_stores_nothing(service):
    resp = await _connect(service, code="bad-code")
    assert resp.headers["location"].endswith("result=failed")
    assert await _grant(service) is None


async def test_manage_changes_no_grant(service):
    await put_grant(service.pool, FakeCipher(), "person-1", "1//kept", [ASSIST_SCOPE], CLIENT_ID)
    query = await _start(service, "manage")
    resp = await service.client.get(
        "/connect/callback", params={"code": "good-code", "state": query["state"][0]}
    )
    assert resp.headers["location"] == f"{PUBLIC}/connect"
    row = await _grant(service)
    assert await FakeCipher().decrypt(row["refresh_token_ciphertext"]) == b"1//kept"


# ── disconnect ───────────────────────────────────────────────────────────────


async def test_disconnect_revokes_at_google_and_deletes_the_grant(service):
    await _connect(service)
    resp = await service.client.post("/connect/disconnect", headers={"Origin": PUBLIC})
    assert resp.headers["location"].endswith("result=disconnected")
    assert service.google.revoked == ["1//connected-refresh-token"]
    assert await _grant(service) is None
    page = await service.client.get("/connect")
    assert "Not connected" in page.text


async def test_disconnect_deletes_the_grant_even_when_google_refuses(service):
    await _connect(service)
    service.google.revoke_status = 400
    await service.client.post("/connect/disconnect")
    assert await _grant(service) is None


async def test_a_cross_site_disconnect_is_refused(service):
    await _connect(service)
    resp = await service.client.post("/connect/disconnect", headers={"Origin": "https://evil.test"})
    assert resp.status_code == 403
    assert await _grant(service) is not None


async def test_disconnect_without_a_sign_in_changes_nothing(service):
    await put_grant(service.pool, FakeCipher(), "person-1", "1//r", [ASSIST_SCOPE], CLIENT_ID)
    resp = await service.client.post("/connect/disconnect")
    assert resp.headers["location"].endswith("result=expired")
    assert await _grant(service) is not None


async def test_the_api_shows_and_deletes_the_persons_own_connection(service):
    headers = {"Authorization": f"Bearer {ub_token()}"}
    assert (await service.client.get("/connection", headers=headers)).json() == {
        "connected": False,
        "connected_at": None,
        "connect_url": f"{PUBLIC}/connect",
    }
    await put_grant(service.pool, FakeCipher(), "person-1", "1//r", [ASSIST_SCOPE], CLIENT_ID)
    assert (await service.client.get("/connection", headers=headers)).json()["connected"]

    resp = await service.client.delete("/connection", headers=headers)
    assert resp.json() == {"disconnected": True}
    assert service.google.revoked == ["1//r"]
    assert await _grant(service) is None
    assert (await service.client.delete("/connection")).status_code == 401


# ── the reuse notice and the not-connected link ──────────────────────────────


def _workspace_ok(service) -> None:
    sources = {s: {"status": "skipped"} for s in ("ub", "runn")} | {"workspace": {"status": "ok"}}
    service.engine.script = [ask_event({"ask_answer": answer(sources=sources)})]


async def _notices(service, app: str) -> list[dict]:
    events = parse_sse((await ask(service, ub_token(app=app))).text)
    return events[-1][1].get("notices") or []


async def test_the_reuse_notice_shows_once_in_a_second_app(service):
    await put_grant(service.pool, FakeCipher(), "person-1", "1//r", [ASSIST_SCOPE], CLIENT_ID)
    _workspace_ok(service)

    assert await _notices(service, "holocron") == []  # the first app says nothing
    assert await _notices(service, "holocron") == []
    [notice] = await _notices(service, "ava")  # the second app, once
    assert notice == {
        "code": "WORKSPACE_CONNECTION_REUSED",
        "message": "Using your existing Gemini Enterprise connection.",
    }
    assert await _notices(service, "ava") == []
    assert len(await _notices(service, "studio")) == 1  # a third app, once too

    # A reconnect starts over: each app says it again.
    await put_grant(service.pool, FakeCipher(), "person-1", "1//r2", [ASSIST_SCOPE], CLIENT_ID)
    assert await _notices(service, "ava") == []
    assert len(await _notices(service, "holocron")) == 1


async def test_no_notice_when_workspace_did_not_answer(service):
    await put_grant(service.pool, FakeCipher(), "person-1", "1//r", [ASSIST_SCOPE], CLIENT_ID)
    _workspace_ok(service)
    await _notices(service, "holocron")
    service.engine.script = [ask_event({"ask_answer": answer()})]  # workspace skipped
    assert await _notices(service, "ava") == []
    _workspace_ok(service)
    assert len(await _notices(service, "ava")) == 1  # not yet claimed by ava


async def test_not_connected_carries_this_services_connect_link(service):
    sources = {s: {"status": "skipped"} for s in ("ub", "runn")} | {
        "workspace": {
            "status": "error",
            "error": {"code": "WORKSPACE_NOT_CONNECTED", "message": "Connect.", "link": None},
        }
    }
    service.engine.script = [ask_event({"ask_answer": answer(sources=sources)})]
    events = parse_sse((await ask(service, ub_token())).text)
    assert events[-1][1]["sources"]["workspace"]["error"]["link"] == f"{PUBLIC}/connect"


# ── not set up ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "over", [{"public_base_url": None}, {"google_client_secret": None}, {"grant_kms_key": None}]
)
async def test_without_its_config_the_page_says_it_is_not_set_up(pool, engine, google, over):
    # No grant key means no cipher: the service builds one only from the key.
    cipher = None if "grant_kms_key" in over else FakeCipher()
    http = httpx.AsyncClient(transport=httpx.MockTransport(google.handler))
    app = create_app(
        settings(**over), engine=engine, cipher=cipher, verifier=verifier(), pool=pool, http=http
    )
    async with app.router.lifespan_context(app):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC)
        assert (await client.get("/connect")).status_code == 503
        resp = await client.get("/connect/start")
        assert resp.status_code == 503 and resp.json()["code"] == "CONNECT_NOT_CONFIGURED"
        await client.aclose()
    await http.aclose()


# ── served through UB (the connect proxy) ────────────────────────────────────


async def test_ub_serving_the_public_origin_is_accepted(service):
    resp = await service.client.get("/connect/start", headers={"x-ub-connect-origin": PUBLIC})
    assert resp.status_code == 302
    query = parse_qs(urlsplit(resp.headers["location"]).query)
    assert query["redirect_uri"] == [f"{PUBLIC}/connect/callback"]


@pytest.mark.parametrize("path", ["/connect", "/connect/start", "/connect/callback"])
async def test_ub_serving_another_origin_is_refused(service, path):
    resp = await service.client.get(
        path, headers={"x-ub-connect-origin": "https://ub-other.example"}
    )
    assert resp.status_code == 503
    assert STATE_COOKIE not in resp.cookies
    if path != "/connect":
        assert resp.json()["code"] == "CONNECT_ORIGIN_MISMATCH"


async def test_urls_never_come_from_forwarded_host_headers(service):
    resp = await service.client.get(
        "/connect/start",
        headers={"host": "evil.example", "x-forwarded-host": "evil.example"},
    )
    query = parse_qs(urlsplit(resp.headers["location"]).query)
    assert query["redirect_uri"] == [f"{PUBLIC}/connect/callback"]

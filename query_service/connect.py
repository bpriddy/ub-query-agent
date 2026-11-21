"""
connect.py — the connect page: a person connects Gemini Enterprise once, for
every app in the collection (ub-query-agent#93).

The flow, in the person's browser:

    GET  /connect             the page: connected or not, and the buttons
    GET  /connect/start       Google consent, with the Gemini Enterprise client
    GET  /connect/callback    Google sends the person back here
    POST /connect/disconnect  revoke and delete the grant

**Sign in with UB.** Who the person is comes from UB, never from Google
directly: the callback hands the Google access token to UB's
`/auth/google/access-token-exchange`, the same exchange the chat bot uses with
the same client, and the UB token UB returns is checked against UB's keys like
any caller's (auth.py). Its subject is the person the grant is stored under,
which is the person `/ask` mints for (D28). The UB session that exchange
opens is closed at once (`/auth/logout`): the page needs one answer, not a
session.

**Scopes.** `discoveryengine.assist.readwrite` is the grant: it lets the
query agent call Gemini Enterprise's streamAssist as the person, nothing
else. `openid email` come with it because UB's exchange reads the person's
address from Google. Only the assist scope is recorded on the grant. "Manage"
(see the grant, disconnect) signs in with `openid email` alone and changes
nothing.

**State.** Two short-lived cookies, each a KMS ciphertext of a small JSON
(KMS ciphertexts are authenticated, so no new secret is needed to sign them;
`aad` keeps a cookie from ever decrypting as a grant or as the other cookie):
the OAuth state (10 minutes, CSRF) and, after a sign-in, who is managing the
page (15 minutes). Both are HttpOnly, Secure, SameSite=Lax, so a cross-site
POST to disconnect carries neither; the POST also checks Origin.

**UB-facing API.** `GET /connection` and `DELETE /connection` do the same for
an app through UB, with the person's UB token like `/ask`.

**Chat bot users** reconnect here: their grants are not copied (see
README, "Chat bot grants").
"""

from __future__ import annotations

import base64
import hmac
import html
import json
import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import grants
from .auth import Caller
from .config import Settings
from .errors import ServiceError

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
ASSIST_SCOPE = "https://www.googleapis.com/auth/discoveryengine.assist.readwrite"
IDENTITY_SCOPES = ("openid", "email")

STATE_COOKIE = "qa_connect_state"
SESSION_COOKIE = "qa_connect_session"
STATE_TTL_S = 600
SESSION_TTL_S = 900
_STATE_AAD = b"ub-query-agent:connect-state"
_SESSION_AAD = b"ub-query-agent:connect-session"

INTENTS = ("connect", "manage")

router = APIRouter()


# ── settings and cookies ─────────────────────────────────────────────────────


def _ready(request: Request) -> Settings:
    """The connect page needs the client, the public URL and the KMS key."""
    s: Settings = request.app.state.settings
    if not (
        s.google_client_id
        and s.google_client_secret
        and s.public_base_url
        and request.app.state.cipher is not None
    ):
        raise ServiceError(
            503, "CONNECT_NOT_CONFIGURED", "The connect page is not set up on this service."
        )
    return s


def _cookie_path(s: Settings) -> str:
    return (urlsplit(s.public_base_url or "").path.rstrip("/") or "") + "/connect"


async def _seal(request: Request, payload: dict[str, Any], aad: bytes) -> str:
    raw = await request.app.state.cipher.encrypt(json.dumps(payload).encode(), aad)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


async def _open(request: Request, name: str, aad: bytes, ttl: int) -> dict[str, Any] | None:
    value = request.cookies.get(name)
    if not value:
        return None
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        payload = json.loads(await request.app.state.cipher.decrypt(raw, aad))
    except Exception:  # noqa: BLE001 — a bad or foreign cookie is no cookie
        return None
    if not isinstance(payload, dict) or time.time() - float(payload.get("iat", 0)) > ttl:
        return None
    return payload


def _set_cookie(resp: Response, s: Settings, name: str, value: str, max_age: int) -> None:
    resp.set_cookie(
        name,
        value,
        max_age=max_age,
        path=_cookie_path(s),
        secure=True,
        httponly=True,
        samesite="lax",
    )


def _clear_cookie(resp: Response, s: Settings, name: str) -> None:
    resp.delete_cookie(name, path=_cookie_path(s), secure=True, httponly=True, samesite="lax")


def _page_url(s: Settings, result: str | None = None) -> str:
    return f"{s.connect_url}" + (f"?{urlencode({'result': result})}" if result else "")


# ── the page ─────────────────────────────────────────────────────────────────

_RESULTS = {
    "connected": "Gemini Enterprise is connected. Every app in the collection can now "
    "include your Workspace in answers.",
    "disconnected": "Gemini Enterprise is disconnected. Answers will not include your "
    "Workspace until you connect again.",
    "denied": "Nothing was connected: Google sign-in was cancelled.",
    "incomplete": "Nothing was connected: Google did not grant access to Gemini Enterprise. "
    "Try again and allow it.",
    "failed": "Something went wrong and nothing was changed. Try again.",
    "expired": "That sign-in took too long or was already used. Try again.",
}


def _render(s: Settings, *, body: str, status: int = 200) -> HTMLResponse:
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect Gemini Enterprise</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#5f6368; --accent:#1a73e8; --line:#dadce0; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16171a; --fg:#e8eaed; --muted:#9aa0a6; --accent:#8ab4f8; --line:#3c4043; }}
}}
body {{ background:var(--bg); color:var(--fg); margin:0;
  font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:34rem; margin:3rem auto; padding:0 16px; }}
h1 {{ font-size:1.4rem; margin:0 0 1rem; }}
p {{ margin:0 0 1rem; }} .muted {{ color:var(--muted); }}
.note {{ border-left:3px solid var(--accent); padding:.25rem .75rem; margin:0 0 1.25rem; }}
a.button, button {{ display:inline-block; font:inherit; padding:.55rem 1rem;
  border-radius:6px; border:1px solid var(--line); background:transparent;
  color:var(--fg); text-decoration:none; cursor:pointer; margin:0 .5rem .5rem 0; }}
a.primary {{ background:var(--accent); border-color:var(--accent); color:var(--bg); }}
form {{ display:inline; }}
</style></head>
<body><main>
<h1>Gemini Enterprise for UB answers</h1>
{body}
</main></body></html>"""
    return HTMLResponse(page, status_code=status, headers={"Cache-Control": "no-store"})


@router.get("/connect")
async def page(request: Request, result: str | None = None) -> HTMLResponse:
    try:
        s = _ready(request)
    except ServiceError as err:
        return _render(
            request.app.state.settings, body=f"<p>{html.escape(err.message)}</p>", status=err.status
        )
    parts: list[str] = []
    if result in _RESULTS:
        parts.append(f'<p class="note">{html.escape(_RESULTS[result])}</p>')
    parts.append(
        "<p>Connect once, and answers in every app that uses UB can include what "
        "you can see in Google Workspace (Drive, Gmail, Calendar), read through "
        "Gemini Enterprise as you. You can disconnect at any time.</p>"
    )
    session = await _open(request, SESSION_COOKIE, _SESSION_AAD, SESSION_TTL_S)
    if session is None:
        parts.append(
            '<p><a class="button primary" href="connect/start">Connect Gemini Enterprise</a>'
            '<a class="button" href="connect/start?intent=manage">'
            "Already connected? Sign in to manage</a></p>"
        )
        return _render(s, body="\n".join(parts))

    status = await grants.grant_status(request.app.state.pool, session["person"])
    who = html.escape(str(session.get("email") or "you"))
    if status is None:
        parts.append(f'<p class="muted">Signed in as {who}. Not connected.</p>')
        parts.append(
            '<p><a class="button primary" href="connect/start">Connect Gemini Enterprise</a></p>'
        )
    else:
        since = html.escape(status["connected_at"][:10])
        parts.append(f'<p class="muted">Signed in as {who}. Connected since {since}.</p>')
        parts.append(
            '<p><a class="button" href="connect/start">Reconnect</a>'
            '<form method="post" action="connect/disconnect">'
            '<button type="submit">Disconnect</button></form></p>'
        )
    return _render(s, body="\n".join(parts))


# ── Google consent ───────────────────────────────────────────────────────────


@router.get("/connect/start")
async def start(request: Request, intent: str = "connect") -> Response:
    s = _ready(request)
    if intent not in INTENTS:
        intent = "connect"
    nonce = secrets.token_urlsafe(24)
    scopes = [*IDENTITY_SCOPES, ASSIST_SCOPE] if intent == "connect" else list(IDENTITY_SCOPES)
    params = {
        "client_id": s.google_client_id,
        "redirect_uri": f"{s.public_base_url}/connect/callback",
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": nonce,
    }
    if intent == "connect":
        # A refresh token, every time: prompt=consent makes Google send one
        # even when the person granted this client before. No
        # include_granted_scopes: only the scopes asked for here.
        params |= {"access_type": "offline", "prompt": "consent"}
    if s.google_hosted_domain:
        params["hd"] = s.google_hosted_domain  # a hint for the account chooser only
    resp = RedirectResponse(f"{AUTHORIZE_URL}?{urlencode(params)}", status_code=302)
    sealed = await _seal(
        request, {"nonce": nonce, "iat": time.time(), "intent": intent}, _STATE_AAD
    )
    _set_cookie(resp, s, STATE_COOKIE, sealed, STATE_TTL_S)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _back(s: Settings, result: str | None) -> RedirectResponse:
    resp = RedirectResponse(_page_url(s, result), status_code=303)
    _clear_cookie(resp, s, STATE_COOKIE)
    return resp


@router.get("/connect/callback")
async def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> Response:
    s = _ready(request)
    sealed = await _open(request, STATE_COOKIE, _STATE_AAD, STATE_TTL_S)
    if sealed is None or not state or not hmac.compare_digest(str(sealed["nonce"]), state):
        logger.warning("connect.callback: state refused")
        return _back(s, "expired")
    if error or not code:
        logger.info("connect.callback: google error=%s", error)
        return _back(s, "denied" if error == "access_denied" else "failed")
    intent = sealed.get("intent", "connect")

    http = request.app.state.http
    resp = await http.post(
        grants.TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": s.google_client_id,
            "client_secret": s.google_client_secret,
            "redirect_uri": f"{s.public_base_url}/connect/callback",
        },
    )
    if resp.status_code != 200:
        logger.warning("connect.callback: code exchange status=%s", resp.status_code)
        return _back(s, "failed")
    tokens = resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        return _back(s, "failed")

    caller = await _ub_person(request, s, access_token)
    if caller is None:
        return _back(s, "failed")

    result: str | None = None  # manage: signed in, nothing changed
    if intent == "connect":
        granted = set(str(tokens.get("scope") or "").split())
        refresh_token = tokens.get("refresh_token")
        if ASSIST_SCOPE not in granted or not refresh_token:
            # The person unticked Gemini Enterprise on the consent screen.
            result = "incomplete"
        else:
            await grants.put_grant(
                request.app.state.pool,
                request.app.state.cipher,
                caller.person_id,
                refresh_token,
                [ASSIST_SCOPE],
                s.google_client_id,
            )
            logger.info("connect.grant stored")
            result = "connected"

    out = _back(s, result)
    session = await _seal(
        request,
        {"person": caller.person_id, "email": caller.email, "iat": time.time()},
        _SESSION_AAD,
    )
    _set_cookie(out, s, SESSION_COOKIE, session, SESSION_TTL_S)
    return out


async def _ub_person(request: Request, s: Settings, google_access_token: str) -> Caller | None:
    """Sign in with UB: UB decides who holds this Google token. Then close the
    UB session the exchange opened."""
    http = request.app.state.http
    resp = await http.post(
        f"{s.ub_base_url}/auth/google/access-token-exchange",
        json={"accessToken": google_access_token},
    )
    if resp.status_code != 200:
        logger.warning("connect.ub_exchange: status=%s", resp.status_code)
        return None
    body = resp.json()
    try:
        caller = await request.app.state.verifier.caller(f"Bearer {body.get('accessToken', '')}")
    except ServiceError as err:
        logger.warning("connect.ub_exchange: token refused (%s)", err.code)
        return None
    refresh = body.get("refreshToken")
    if refresh:
        try:
            await http.post(f"{s.ub_base_url}/auth/logout", json={"refreshToken": refresh})
        except Exception:  # noqa: BLE001 — the session expires on its own
            logger.warning("connect.ub_logout failed")
    return caller


# ── disconnect ───────────────────────────────────────────────────────────────


@router.post("/connect/disconnect")
async def disconnect(request: Request, origin: str | None = Header(default=None)) -> Response:
    s = _ready(request)
    allowed = "{0.scheme}://{0.netloc}".format(urlsplit(s.public_base_url or ""))
    if origin is not None and origin != allowed:
        raise ServiceError(403, "BAD_ORIGIN", "This request did not come from the connect page.")
    session = await _open(request, SESSION_COOKIE, _SESSION_AAD, SESSION_TTL_S)
    if session is None:
        return RedirectResponse(_page_url(s, "expired"), status_code=303)
    await grants.revoke_grant(
        request.app.state.pool, request.app.state.cipher, request.app.state.http, session["person"]
    )
    logger.info("connect.grant revoked")
    return RedirectResponse(_page_url(s, "disconnected"), status_code=303)


# ── the same, for an app through UB ──────────────────────────────────────────


@router.get("/connection")
async def connection(request: Request, authorization: str | None = Header(default=None)):
    caller: Caller = await request.app.state.verifier.caller(authorization)
    s: Settings = request.app.state.settings
    status = await grants.grant_status(request.app.state.pool, caller.person_id)
    return JSONResponse(
        {
            "connected": status is not None,
            "connected_at": status["connected_at"] if status else None,
            "connect_url": s.connect_url,
        },
        headers={"Cache-Control": "no-store"},
    )


@router.delete("/connection")
async def delete_connection(request: Request, authorization: str | None = Header(default=None)):
    caller: Caller = await request.app.state.verifier.caller(authorization)
    if request.app.state.cipher is None:
        raise ServiceError(503, "CONNECT_NOT_CONFIGURED", "Grants are not set up on this service.")
    had = await grants.revoke_grant(
        request.app.state.pool, request.app.state.cipher, request.app.state.http, caller.person_id
    )
    return JSONResponse({"disconnected": had}, headers={"Cache-Control": "no-store"})

"""
app.py — the HTTP surface: `POST /ask` and `GET /healthz`.

`POST /ask` answers with a Server-Sent Events stream, in this order:

    event: conversation  {"conversation_id": "…", "new": true}
    event: progress      {"step": "lookup", "text": "Reading a campaign"}   (0 or more)
    event: answer        {AskResponse}                                      (exactly 1)

The answer is always last and always sent, even when the turn fails: then it
is `kind="error"` with the reason. Errors before the stream starts (no token,
a bad body, someone else's conversation, the engine down) are plain JSON
`{"code", "message"}` with an HTTP status.

Only events the `/ask` turn authors (`gub_agent/agents/ask.py`) are relayed.
The engine's other events (model text, tool calls and results) never leave
this service.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from . import conversations
from .auth import Caller, UbTokenVerifier
from .config import Settings
from .db import create_pool, migrate
from .engine import UB_TOKEN_KEY, WORKSPACE_TOKEN_KEY, Engine
from .errors import ServiceError
from .gcp import ServiceToken, new_client
from .grants import Cipher, KmsCipher, mint_access_token

logger = logging.getLogger("query_service")

ASK_AUTHOR = "ask"
PROGRESS_KEY = "progress"
ASK_ANSWER_KEY = "ask_answer"
SOURCES = ("ub", "runn", "workspace")


class ContextIn(BaseModel):
    type: Literal[1, 2]
    id: str | None = Field(default=None, max_length=128)


class AskIn(BaseModel):
    """What UB sends. There is no person field: the person is the token's (D28)."""

    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)
    actor: str | None = Field(default=None, min_length=1, max_length=64)
    context: ContextIn | None = None


def error_answer(code: str, message: str, conversation_id: str | None) -> dict[str, Any]:
    """An `AskResponse` with kind="error" (gub_agent/schemas/ask.py)."""
    return {
        "kind": "error",
        "headline": "The question could not be answered",
        "error": {"code": code, "message": message},
        "sources": {s: {"status": "skipped"} for s in SOURCES},
        "conversation_id": conversation_id,
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _ask_metadata(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("author") != ASK_AUTHOR:
        return None
    meta = event.get("custom_metadata") or event.get("customMetadata")
    return meta if isinstance(meta, dict) else None


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    cipher: Cipher | None = None,
    verifier: UbTokenVerifier | None = None,
    pool: Any = None,
    http: httpx.AsyncClient | None = None,
) -> FastAPI:
    """The app. Tests pass fakes for the engine, the cipher, the verifier, the
    pool and the HTTP client; on Cloud Run everything is built from the
    environment at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or Settings.from_env()
        client = http or new_client()
        token = ServiceToken()
        app.state.settings = s
        app.state.http = client
        app.state.pool = pool or await create_pool(s)
        applied = await migrate(app.state.pool)
        if applied:
            logger.info("migrations applied: %s", ", ".join(applied))
        app.state.engine = engine or Engine(s.engine, s.engine_region, client, token)
        app.state.cipher = cipher or (
            KmsCipher(s.grant_kms_key, client, token) if s.grant_kms_key else None
        )
        app.state.verifier = verifier or UbTokenVerifier(
            f"{s.ub_base_url}/auth/jwks", s.ub_issuer, s.ub_audience
        )
        try:
            yield
        finally:
            if http is None:
                await client.aclose()
            if pool is None:
                await app.state.pool.close()

    app = FastAPI(title="ub-query-agent", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, err: ServiceError) -> JSONResponse:
        return JSONResponse(err.body(), status_code=err.status)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    async def ask(request: Request, authorization: str | None = Header(default=None)):
        state = request.app.state
        caller: Caller = await state.verifier.caller(authorization)
        try:
            body = AskIn.model_validate(await request.json())
        except (ValidationError, ValueError) as err:
            raise ServiceError(400, "BAD_REQUEST", "The request body is not valid.") from err

        s: Settings = state.settings
        conversation = None
        if body.conversation_id:
            conversation = await conversations.find_resumable(
                state.pool, caller, body.conversation_id, s.conversation_idle_seconds
            )
        if conversation is None:
            session_id = await state.engine.create_session(caller.person_id)
            context = (
                conversations.Context(type=body.context.type, id=body.context.id)
                if body.context
                else None
            )
            conversation = await conversations.create(
                state.pool, caller, session_id, body.actor, context
            )

        state_delta = {UB_TOKEN_KEY: caller.token}
        if state.cipher is not None:
            google_token = await mint_access_token(
                state.pool, state.cipher, state.http, s, caller.person_id
            )
            if google_token:
                state_delta[WORKSPACE_TOKEN_KEY] = google_token

        stream = _relay(state.engine, state.pool, caller, conversation, body.question, state_delta)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


async def _relay(
    engine: Engine,
    pool: Any,
    caller: Caller,
    conversation: conversations.Conversation,
    question: str,
    state_delta: dict[str, str],
) -> AsyncIterator[bytes]:
    started = time.monotonic()
    outcome = "answer"
    progress = 0
    yield _sse("conversation", {"conversation_id": conversation.id, "new": conversation.new})
    answer: dict[str, Any] | None = None
    try:
        async for event in engine.stream_turn(
            caller.person_id, conversation.engine_session_id, question, state_delta
        ):
            meta = _ask_metadata(event)
            if meta is None:
                continue
            if PROGRESS_KEY in meta:
                progress += 1
                yield _sse("progress", meta[PROGRESS_KEY])
            elif ASK_ANSWER_KEY in meta:
                answer = dict(meta[ASK_ANSWER_KEY])
    except ServiceError as err:
        outcome = err.code
        answer = error_answer(err.code, err.message, conversation.id)
    except Exception:  # noqa: BLE001 — the caller always gets an answer event
        logger.exception("ask.relay failed")
        outcome = "INTERNAL"
        answer = error_answer("INTERNAL", "The query agent failed.", conversation.id)

    if answer is None:
        outcome = "NO_ANSWER"
        answer = error_answer(
            "NO_ANSWER", "The query agent finished without an answer.", conversation.id
        )
    # The engine names its own session; the caller only ever sees ours.
    answer["conversation_id"] = conversation.id
    if answer.get("kind") == "error" and outcome == "answer":
        outcome = (answer.get("error") or {}).get("code", "error")
    yield _sse("answer", answer)

    await conversations.touch(pool, conversation.id)
    # Ids and counts only: never the question, the answer or a token.
    logger.info(
        json.dumps(
            {
                "event": "ask.turn",
                "conversation_id": conversation.id,
                "new": conversation.new,
                "app_id": caller.app_id,
                "outcome": outcome,
                "progress": progress,
                "workspace_token": WORKSPACE_TOKEN_KEY in state_delta,
                "ms": int((time.monotonic() - started) * 1000),
            }
        )
    )


# Cloud Run reads stdout; one line per record, the record is the message.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
app = create_app()

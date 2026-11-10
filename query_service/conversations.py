"""
conversations.py — which conversation a turn belongs to.

A conversation is resumed only by its owner: the same person in the same app.
The app is part of ownership because an app's presentation scope (D15)
narrows what its turns can read, and the engine session carries the earlier
turns' tool results; resuming it in another app would show that app records
its scope withholds. Actor and context are recorded at creation, never
compared (D28).

An unknown id and someone else's id get the same answer, so an id says
nothing about whether it exists. An expired conversation (idle longer than the
limit) is not an error: the turn starts a new one, and the answer carries the
new id.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

import asyncpg

from .auth import Caller
from .errors import ServiceError


@dataclass(frozen=True)
class Context:
    type: Literal[1, 2]
    id: str | None = None


@dataclass(frozen=True)
class Conversation:
    id: str
    engine_session_id: str
    new: bool


async def find_resumable(
    pool: asyncpg.Pool,
    caller: Caller,
    conversation_id: str,
    idle_seconds: int,
) -> Conversation | None:
    """The caller's conversation if it may be resumed; None when it expired.
    Raises CONVERSATION_NOT_FOUND when it is not the caller's (or no such id)."""
    try:
        key = uuid.UUID(conversation_id)
    except ValueError:
        key = None
    row = None
    if key is not None:
        row = await pool.fetchrow(
            "SELECT id, engine_session_id,"
            " last_active_at > now() - make_interval(secs => $4) AS live"
            " FROM conversations"
            " WHERE id = $1 AND person_id = $2 AND app_id IS NOT DISTINCT FROM $3",
            key,
            caller.person_id,
            caller.app_id,
            idle_seconds,
        )
    if row is None:
        raise ServiceError(404, "CONVERSATION_NOT_FOUND", "No such conversation.")
    if not row["live"]:
        return None
    await touch(pool, str(row["id"]))
    return Conversation(id=str(row["id"]), engine_session_id=row["engine_session_id"], new=False)


async def create(
    pool: asyncpg.Pool,
    caller: Caller,
    engine_session_id: str,
    actor: str | None,
    context: Context | None,
) -> Conversation:
    row = await pool.fetchrow(
        "INSERT INTO conversations"
        " (person_id, app_id, actor, context_type, context_id, engine_session_id)"
        " VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        caller.person_id,
        caller.app_id,
        actor,
        context.type if context else None,
        context.id if context else None,
        engine_session_id,
    )
    return Conversation(id=str(row["id"]), engine_session_id=engine_session_id, new=True)


async def touch(pool: asyncpg.Pool, conversation_id: str) -> None:
    await pool.execute(
        "UPDATE conversations SET last_active_at = now() WHERE id = $1",
        uuid.UUID(conversation_id),
    )

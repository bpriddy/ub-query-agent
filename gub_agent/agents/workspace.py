"""
workspace.py — Gemini Enterprise over the person's Workspace, as a `/ask`
source (gub-agent#90; UB `docs/ASK.md`).

Ported from the chat bot's `src/agent/streamassist.ts` (gub-gchat-bot, being
sunset). One `streamAssist` call per turn, run as the person: the query agent
service mints a short-lived Google access token from the person's stored grant
and hands it over in `temp:workspace_token`, which ADK applies for the turn and
never saves (#162). This module only reads it.

What comes back is the assistant's answer text plus its grounding: passages of
the text, each with the Workspace items (mail, docs, Chat, events) it was read
from. Each grounded passage becomes one fact with a `ref` to its item. The
answer's prose stays the interpreter's until the blend (#91) writes one answer
from both.

The Gemini Enterprise session is kept per conversation, in the engine
session's state (`workspace_session`: a resource name, not a secret), so it
follows #92's conversation and its expiry. A session the engine rejects gets
one cold retry inside the same deadline, as in the bot.

Offsets in the grounding are not used here: a fact quotes the passage the
engine echoed (`segment.text`), which needs none. The bot's offset handling
(UTF-8 bytes against UTF-16, chunk-local against absolute) matters only for
marking up prose, which is #91's.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from ..schemas.ask import AskFact, Ref, SourceError

logger = logging.getLogger(__name__)

TOKEN_STATE_KEY = "temp:workspace_token"
SESSION_STATE_KEY = "workspace_session"
EVIDENCE_PREFIX = "workspace"

# One call, cold retry included (the bot's STREAMASSIST_TIMEOUT_MS). The turn's
# own soft deadline (config.WORKSPACE_SOFT_DEADLINE_S) is usually what ends it.
HARD_TIMEOUT_S = 90.0
MAX_FACTS = 12
MAX_VALUE_CHARS = 500


@dataclass(frozen=True)
class Passage:
    text: str
    refs: list[Ref]


@dataclass
class WorkspaceResult:
    text: str = ""
    session: str | None = None
    skipped: bool = False
    unauthorized_connectors: list[str] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)


class StreamAssistError(Exception):
    """streamAssist answered non-2xx, or timed out (status 408, TIMEOUT)."""

    def __init__(self, status: int, reason: str | None, detail: str) -> None:
        super().__init__(f"streamAssist {status}{f' ({reason})' if reason else ''}")
        self.status = status
        self.reason = reason
        self.detail = detail


def is_session_rejection(err: StreamAssistError) -> bool:
    """The engine refused the session we asked it to continue. The status
    alone is not enough: a malformed body is a 400 too, and retrying that
    cold would double every turn's latency."""
    return err.status in (400, 403, 404) and bool(
        re.search("session", f"{err.reason or ''} {err.detail}", re.I)
    )


def assist_url(engine: str) -> str:
    """`engine` is the Gemini Enterprise engine's resource name,
    projects/…/locations/{loc}/collections/…/engines/{id}. v1alpha: the narrow
    `discoveryengine.assist.readwrite` scope and `toolsSpec` need it."""
    match = re.search(r"/locations/([^/]+)/", engine)
    location = match.group(1) if match else "global"
    host = (
        "discoveryengine.googleapis.com"
        if location == "global"
        else f"{location}-discoveryengine.googleapis.com"
    )
    return f"https://{host}/v1alpha/{engine}/assistants/default_assistant:streamAssist"


def data_store_specs(engine: str, data_store_ids: list[str]) -> list[dict[str, str]]:
    """The raw API needs each connected data store named (the web UI does this
    itself); without them the assistant has no source and answers "set up
    your connector"."""
    collection = engine.split("/engines/", 1)[0]
    return [{"dataStore": f"{collection}/dataStores/{ds}"} for ds in data_store_ids]


async def stream_assist(
    *,
    engine: str,
    data_store_ids: list[str],
    token: str,
    query: str,
    session: str | None,
    client: httpx.AsyncClient | None = None,
) -> tuple[WorkspaceResult, bool]:
    """Ask the person's assistant. Returns the result and whether a stored
    session was rejected (and so retried cold). Raises StreamAssistError."""
    deadline = time.monotonic() + HARD_TIMEOUT_S
    owned = client is None
    client = client or httpx.AsyncClient()

    async def attempt(with_session: str | None) -> WorkspaceResult:
        return await _attempt(client, engine, data_store_ids, token, query, with_session, deadline)

    try:
        if not session:
            return await attempt(None), False
        try:
            return await attempt(session), False
        except StreamAssistError as err:
            if not is_session_rejection(err):
                raise
            return await attempt(None), True
    finally:
        if owned:
            await client.aclose()


async def _attempt(
    client: httpx.AsyncClient,
    engine: str,
    data_store_ids: list[str],
    token: str,
    query: str,
    session: str | None,
    deadline: float,
) -> WorkspaceResult:
    body: dict[str, Any] = {
        "query": {"text": query},
        "toolsSpec": {
            "vertexAiSearchSpec": {"dataStoreSpecs": data_store_specs(engine, data_store_ids)}
        },
    }
    if session:
        body["session"] = session
    remaining = max(0.0, deadline - time.monotonic())
    acc = _Accumulator()
    splitter = JsonArraySplitter()
    try:
        async with client.stream(
            "POST",
            assist_url(engine),
            headers={"Authorization": f"Bearer {token}"},
            json=body,
            timeout=httpx.Timeout(remaining),
        ) as resp:
            if resp.status_code >= 400:
                raw = (await resp.aread()).decode("utf-8", "replace")
                raise StreamAssistError(resp.status_code, error_reason(raw), raw[:500])
            async for piece in resp.aiter_text():
                for element in splitter.push(piece):
                    acc.fold(element)
                if time.monotonic() > deadline:
                    raise httpx.ReadTimeout("streamAssist hard deadline")
        for element in splitter.flush():
            acc.fold(element)
    except httpx.TimeoutException as err:
        raise StreamAssistError(408, "TIMEOUT", f"no answer in {HARD_TIMEOUT_S:.0f}s") from err
    return acc.result()


class JsonArraySplitter:
    """Splits a streamed JSON array (`[ {…}, {…} ]`) into its top-level
    elements as text arrives. Tracks brace depth outside string literals, so
    brackets inside the answer text never confuse it. A bare object (no array
    around it) is one element."""

    def __init__(self) -> None:
        self._buf = ""
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._started = False
        self._start = -1
        self._scan_from = 0

    def push(self, piece: str) -> list[Any]:
        self._buf += piece
        out: list[Any] = []
        buf = self._buf
        for i in range(self._scan_from, len(buf)):
            ch = buf[i]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif ch == "\\":
                    self._escaped = True
                elif ch == '"':
                    self._in_string = False
                continue
            if ch == '"':
                self._in_string = True
                continue
            if not self._started:
                if ch == "[":
                    self._started = True
                    continue
                if ch != "{":
                    continue
                self._started = True
            if ch in "{[":
                if self._depth == 0:
                    self._start = i
                self._depth += 1
            elif ch in "}]":
                if self._depth == 0:
                    continue  # the array's own closing bracket
                self._depth -= 1
                if self._depth == 0 and self._start >= 0:
                    try:
                        out.append(json.loads(buf[self._start : i + 1]))
                    except json.JSONDecodeError:
                        pass  # one bad element must not poison the stream
                    self._start = -1
        if self._depth == 0 and self._start < 0:
            self._buf = ""
            self._scan_from = 0
        else:
            self._buf = buf[self._start :]
            self._start = 0
            self._scan_from = len(self._buf)
        return out

    def flush(self) -> list[Any]:
        tail = self._buf.strip()
        self._buf = ""
        if not tail:
            return []
        try:
            parsed = json.loads(tail)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else [parsed]


def ref_type(data_store_id: str | None, uri: str | None) -> str:
    """What kind of Workspace item a reference is. A store id ends in its
    connector type; the uri's host is the fallback. Most specific first, as
    the bot's `kindOf`, so `gchat_…_google_chat` is not read as Drive."""
    ds = (data_store_id or "").lower()
    for needle, kind in (
        ("calendar", "event"),
        ("groups", "group"),
        ("sites", "site"),
        ("keep", "note"),
        ("people", "person"),
        ("mail", "mail"),
        ("chat", "chat"),
        ("drive", "doc"),
    ):
        if needle in ds:
            return kind
    host = (urlparse(uri or "").hostname or "").removeprefix("www.")
    return {
        "mail.google.com": "mail",
        "chat.google.com": "chat",
        "drive.google.com": "doc",
        "docs.google.com": "doc",
        "calendar.google.com": "event",
        "groups.google.com": "group",
        "sites.google.com": "site",
        "keep.google.com": "note",
        "contacts.google.com": "person",
    }.get(host, "other")


def _to_ref(raw: Any) -> Ref | None:
    meta = (raw or {}).get("documentMetadata") if isinstance(raw, dict) else None
    if not isinstance(meta, dict):
        return None
    document = meta.get("document") if isinstance(meta.get("document"), str) else None
    uri = meta.get("uri") if isinstance(meta.get("uri"), str) else None
    if not document and not uri:
        return None
    store = re.search(r"/dataStores/([^/]+)(?:/|$)", document or "")
    title = meta.get("title") if isinstance(meta.get("title"), str) and meta["title"] else None
    return Ref(
        type=ref_type(store.group(1) if store else None, uri),
        id=(document or uri or "").rsplit("/", 1)[-1],
        title=title,
        uri=uri,
    )


class _Accumulator:
    """Folds parsed streamAssist chunks into a result. Never raises on an odd
    shape: a malformed grounding costs that passage, never the answer."""

    def __init__(self) -> None:
        self.text = ""
        self.session: str | None = None
        self.skipped = False
        self.unauthorized: list[str] = []
        self.passages: list[Passage] = []
        # Every reference seen so far, in order: live turns showed segment
        # reference indices running past their own chunk's references, so a
        # chunk-only lookup is not enough (the bot's `allRefs`).
        self._all_refs: list[Any] = []

    def fold(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        session = (chunk.get("sessionInfo") or {}).get("session")
        if isinstance(session, str) and session:
            self.session = session
        answer = chunk.get("answer") or {}
        if answer.get("assistSkippedReasons"):
            self.skipped = True
        for reply in answer.get("replies") or []:
            content = (reply or {}).get("groundedContent") or {}
            text = (content.get("content") or {}).get("text")
            reply_text = text if isinstance(text, str) else ""
            self.text += reply_text
            grounding = content.get("textGroundingMetadata")
            if isinstance(grounding, dict):
                try:
                    self._fold_grounding(grounding, reply_text)
                except Exception:  # noqa: BLE001 — see the class docstring
                    logger.warning("workspace: malformed grounding skipped")
        for err in chunk.get("connectorAuthErrors") or []:
            connector = (err or {}).get("dataConnector")
            if isinstance(connector, str):
                self.unauthorized.append(connector)

    def _fold_grounding(self, grounding: dict, reply_text: str) -> None:
        refs = grounding.get("references") if isinstance(grounding.get("references"), list) else []
        base = len(self._all_refs)
        self._all_refs.extend(refs)
        for seg in grounding.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            passage = seg.get("text") if isinstance(seg.get("text"), str) else None
            if not passage:
                start, end = _index(seg.get("startIndex")), _index(seg.get("endIndex"))
                if start is None or end is None or end <= start:
                    continue
                passage = reply_text[start:end]
            resolved: list[Ref] = []
            for raw_index in seg.get("referenceIndices") or []:
                i = _index(raw_index)
                if i is None:
                    resolved = []
                    break
                raw = (
                    refs[i]
                    if i < len(refs)
                    else self._all_refs[base + i]
                    if base + i < len(self._all_refs)
                    else self._all_refs[i]
                    if i < len(self._all_refs)
                    else None
                )
                ref = _to_ref(raw)
                if ref is None:
                    # Partial attribution is worse than none: a passage from
                    # mail and a doc must not read as the doc alone.
                    resolved = []
                    break
                resolved.append(ref)
            if passage.strip() and resolved:
                self.passages.append(Passage(text=passage.strip(), refs=resolved))

    def result(self) -> WorkspaceResult:
        return WorkspaceResult(
            text=self.text.strip(),
            session=self.session,
            skipped=self.skipped,
            unauthorized_connectors=list(dict.fromkeys(self.unauthorized)),
            passages=self.passages,
        )


def _index(raw: Any) -> int | None:
    """An int64 off the wire: a string, sometimes a number; absent is zero."""
    if raw is None:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def error_reason(raw: str) -> str | None:
    """The Discovery Engine reason (LICENSE_INACTIVE, USER_UNAUTHENTICATED, …)
    from an error body."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    obj = parsed[0] if isinstance(parsed, list) and parsed else parsed
    error = obj.get("error") if isinstance(obj, dict) else None
    if not isinstance(error, dict):
        return None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and detail.get("reason"):
            return str(detail["reason"])
    status = error.get("status")
    return str(status) if status else None


def facts_of(result: WorkspaceResult) -> list[AskFact]:
    """One fact per grounded passage and item, each quoting the passage and
    naming the item it was read from. Repeats are dropped; at most MAX_FACTS."""
    facts: list[AskFact] = []
    seen: set[tuple[str, str]] = set()
    for passage in result.passages:
        value = passage.text[:MAX_VALUE_CHARS]
        for ref in passage.refs:
            key = (ref.id, value)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                AskFact(
                    evidence_id=f"{EVIDENCE_PREFIX}:{len(facts) + 1}",
                    value=value,
                    source="workspace",
                    ref=ref,
                )
            )
            if len(facts) == MAX_FACTS:
                return facts
    return facts


def not_connected(connect_url: str | None) -> SourceError:
    return SourceError(
        code="WORKSPACE_NOT_CONNECTED",
        message="Connect Gemini Enterprise to include your Workspace in answers.",
        link=connect_url or None,
    )


def source_error(err: StreamAssistError, app_url: str | None) -> SourceError:
    """What a failed call tells the caller. The person's own access gaps say
    what to do; anything else names the engine's reason."""
    if err.reason == "USER_UNAUTHENTICATED":
        return SourceError(
            code="WORKSPACE_SETUP_REQUIRED",
            message="Open Gemini Enterprise once to finish setting it up, then ask again.",
            link=app_url or None,
        )
    if err.reason == "LICENSE_INACTIVE":
        return SourceError(
            code="WORKSPACE_NO_LICENSE",
            message="You don't have a Gemini Enterprise licence.",
        )
    if err.status == 401:
        return SourceError(
            code="WORKSPACE_TOKEN_REJECTED",
            message="Google refused the Workspace connection; reconnect Gemini Enterprise.",
        )
    return SourceError(
        code="WORKSPACE_UNAVAILABLE",
        message=f"Gemini Enterprise answered {err.status}"
        + (f" ({err.reason})" if err.reason else "")
        + ".",
    )

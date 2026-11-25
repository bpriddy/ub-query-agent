"""
Test doubles shared across the suite.

MockGub is a real localhost HTTP/1.1 keep-alive server (not a transport mock),
so tests exercise the actual network path: connection pooling, status codes,
and request bodies. FakeToolContext stands in for the ADK ToolContext that
Gemini Enterprise injects at runtime. `invocation_ctx` builds a REAL ADK
InvocationContext over InMemorySessionService, for the agents whose behaviour
is control flow over session state and events (the gates, the dispatcher).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types


class MockGub:
    """Keep-alive HTTP server; (method, path) -> (status, json payload) with an
    optional third slot of extra response headers.

    The payload is whatever GUB answers with, NOT necessarily a dict: several
    org endpoints answer with a bare JSON array. The header slot exists for the
    signals the client reads off the response rather than the body — today the
    tenant-10 `x-account-scope-filtered` flag, which is the only way to
    exercise the bare-array wrapping end to end.

    `connections` counts TCP connections (for pooling assertions);
    `requests` records every (method, path) hit (for call-count assertions);
    `targets` records the same hits with the query string still attached, for
    the assertions that are about what was SENT rather than how often;
    `bodies` records each request's JSON body (None when it had none), in the
    same order.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple] = {}
        self.connections = 0
        self.requests: list[tuple[str, str]] = []
        self.targets: list[tuple[str, str]] = []
        self.bodies: list[Any] = []
        self._server: asyncio.Server | None = None
        self.port: int | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode()
                request_line, *header_lines = head.split("\r\n")
                method, target, _ = request_line.split(" ")
                path = target.split("?")[0]
                content_length = 0
                for line in header_lines:
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1])
                raw = await reader.readexactly(content_length) if content_length else b""

                self.requests.append((method, path))
                self.targets.append((method, target))
                self.bodies.append(json.loads(raw) if raw else None)
                status, payload, *rest = self.routes.get(
                    (method, path), (404, {"detail": "not found"})
                )
                extra = rest[0] if rest else {}
                body = json.dumps(payload).encode()
                writer.write(
                    (
                        f"HTTP/1.1 {status} X\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\n"
                        + "".join(f"{name}: {value}\r\n" for name, value in extra.items())
                        + "Connection: keep-alive\r\n\r\n"
                    ).encode()
                    + body
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()


class FakeState(dict):
    """ADK State stand-in: dict plus the to_dict() the auth code calls."""

    def to_dict(self) -> dict:
        return dict(self)


class FakeToolContext:
    """ToolContext stand-in carrying session state (e.g. an injected OAuth token)."""

    def __init__(self, **state) -> None:
        self.state = FakeState(state)


async def invocation_ctx(
    *,
    state: dict | None = None,
    invocation_id: str = "inv-1",
    user_text: str | None = None,
    events: list[Event] | None = None,
) -> InvocationContext:
    """A real InvocationContext: session state seeded, optional prior events
    appended (so `state_delta`s are committed the way the runner commits
    them), and `user_content` set to the question under test."""
    service = InMemorySessionService()
    session = await service.create_session(app_name="gub", user_id="u", state=state or {})
    for event in events or []:
        await service.append_event(session, event)
    content = (
        genai_types.Content(role="user", parts=[genai_types.Part(text=user_text)])
        if user_text is not None
        else None
    )
    return InvocationContext(
        session_service=service,
        invocation_id=invocation_id,
        agent=BaseAgent(name="host"),
        session=session,
        user_content=content,
    )


# ── resourcing's query syntax (resourcing-aggregate service/src/query.ts) ─────

RESOURCING_OPS = {"ne", "in", "gt", "gte", "lt", "lte", "contains", "is_null"}


def resourcing_read(
    rows: list[dict], params: dict[str, str], *, reserved: tuple[str, ...] = ()
) -> tuple[int, dict]:
    """Filter, order and page `rows` as resourcing does for a view read:
    `col=v`, `col.<op>=v`, `order`, `limit` (≤ 1000, default 200), `offset`.
    Answers `(status, {data, meta: {limit, offset, nextOffset}})`, or a 400
    with resourcing's code when a parameter is not one it takes. `reserved`
    names a route's own parameters (capacity's `from`, `to`)."""
    columns = set().union(*(r.keys() for r in rows)) if rows else set()
    out = list(rows)
    for key, value in params.items():
        if key in ("order", "fields", "limit", "offset") or key in reserved:
            continue
        column, _, op = key.partition(".")
        op = op or "eq"
        if (columns and column not in columns) or (op != "eq" and op not in RESOURCING_OPS):
            return 400, {"code": "RESOURCING_BAD_QUERY", "message": f"bad parameter {key!r}"}
        out = [r for r in out if _resourcing_match(r.get(column), op, value)]
    for part in reversed([p for p in params.get("order", "").split(",") if p]):
        name = part.lstrip("-")
        out.sort(key=lambda r: (r.get(name) is None, str(r.get(name))), reverse=part[0] == "-")
    limit = min(int(params.get("limit", "200")), 1000)
    offset = int(params.get("offset", "0"))
    page = out[offset : offset + limit]
    more = offset + limit < len(out)
    return 200, {
        "data": page,
        "meta": {"limit": limit, "offset": offset, "nextOffset": offset + limit if more else None},
    }


def _resourcing_match(value: Any, op: str, arg: str) -> bool:
    if op == "is_null":
        return (value is None) == (arg == "true")
    if value is None:
        return False
    text = str(value).lower()
    if op == "eq":
        return text == arg.lower()
    if op == "ne":
        return text != arg.lower()
    if op == "in":
        return text in {a.lower() for a in arg.split(",")}
    if op == "contains":
        return arg.lower() in text
    a, b = str(value), arg
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        a, b = float(value), float(arg)  # type: ignore[assignment]
    return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]

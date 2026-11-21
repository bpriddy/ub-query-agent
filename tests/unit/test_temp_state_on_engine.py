"""
The per-turn Google token in `temp:` state, on the Agent Engine's own path
(gcp-universal-backend#162, ub-query-agent#90).

The query agent service sends each turn through the engine's
`async_stream_query` with `state_delta = {"temp:workspace_token": …}`
(query_service/engine.py). On the engine that is the Vertex AI `AdkApp`
template over `VertexAiSessionService`, with the pinned ADK. This test runs
exactly that stack in process: the real `AdkApp`, the real
`VertexAiSessionService` and the real `AskTurn`. Only the Agent Engine
sessions API (the remote store) and the Gemini Enterprise call are fakes.

Pinned here:

  * the agent READS the `temp:` token during the turn: the Workspace call is
    made with it (the part #162 left open);
  * the token reaches the remote session store in no request: not in any
    appended event, not in the stored state (the part checked live on the
    sandbox engine, 2026-11-09);
  * an ordinary key in the same delta IS stored, so the fake store is really
    recording what ADK sends;
  * the next turn has no token unless the service sends one again;
  * the service writes the key the agent reads.
"""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.cloud.aiplatform import initializer
from vertexai.agent_engines.templates.adk import AdkApp

from gub_agent import config
from gub_agent.agents import workspace
from gub_agent.agents.ask import ASK_ANSWER_KEY, AskTurn
from query_service.engine import WORKSPACE_TOKEN_KEY

from .test_ask_contract import _tree
from .test_workspace_source import _body

ENGINE_ID = "1234567890"
TOKEN = "ya29.turn-only-google-token-162"
GE_ENGINE = "projects/p/locations/global/collections/default_collection/engines/dev-ge"


class _FakeSessionsApi:
    """The Agent Engine sessions API, as `VertexAiSessionService` calls it.
    It keeps what a real store would keep: the session, its state (merged
    from each appended event's state_delta, as the server does) and every
    event, and records every request body."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events = SimpleNamespace(append=self._append, list=self._list)

    @staticmethod
    def _now() -> datetime.datetime:
        return datetime.datetime.now(tz=datetime.UTC)

    def _session_obj(self, sid: str) -> SimpleNamespace:
        s = self.sessions[sid]
        return SimpleNamespace(
            name=f"reasoningEngines/{ENGINE_ID}/sessions/{sid}",
            user_id=s["user_id"],
            session_state=dict(s["state"]),
            update_time=s["update_time"],
        )

    async def create(self, *, name: str, user_id: str, config: dict[str, Any]) -> Any:
        self.requests.append({"create": {"name": name, "user_id": user_id, "config": config}})
        sid = str(len(self.sessions) + 1)
        self.sessions[sid] = {
            "user_id": user_id,
            "state": dict(config.get("session_state") or {}),
            "events": [],
            "update_time": self._now(),
        }
        return SimpleNamespace(response=self._session_obj(sid))

    async def get(self, *, name: str) -> Any:
        return self._session_obj(name.rsplit("/", 1)[-1])

    async def _append(self, *, name: str, author, invocation_id, timestamp, config) -> None:
        self.requests.append({"append": {"name": name, "author": author, "config": config}})
        s = self.sessions[name.rsplit("/", 1)[-1]]
        s["state"].update((config.get("actions") or {}).get("state_delta") or {})
        s["events"].append(
            SimpleNamespace(
                name=f"{name}/events/{len(s['events']) + 1}",
                raw_event=config.get("raw_event"),
                invocation_id=invocation_id,
                author=author,
                timestamp=timestamp,
            )
        )
        s["update_time"] = self._now()

    async def _list(self, *, name: str, **_kwargs: Any):
        async def gen():
            for event in self.sessions[name.rsplit("/", 1)[-1]]["events"]:
                yield event

        return gen()

    def wire(self) -> str:
        """Everything ADK sent the store, and everything the store holds."""
        held = [
            {"state": s["state"], "events": [e.raw_event for e in s["events"]]}
            for s in self.sessions.values()
        ]
        return json.dumps(self.requests, default=str) + json.dumps(held, default=str)


class _FakeClient:
    def __init__(self, api: _FakeSessionsApi) -> None:
        self.agent_engines = SimpleNamespace(sessions=api)

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


@pytest.fixture
def store(monkeypatch) -> _FakeSessionsApi:
    api = _FakeSessionsApi()
    monkeypatch.setattr(VertexAiSessionService, "_get_api_client", lambda self: _FakeClient(api))
    # AdkApp reads the project from Vertex's global config, which would fall
    # back to Application Default Credentials; pin it so no credential is read.
    monkeypatch.setattr(type(initializer.global_config), "project", property(lambda _s: "p"))
    monkeypatch.setattr(
        type(initializer.global_config), "location", property(lambda _s: "us-central1")
    )
    # AdkApp.set_up writes these into the process environment; keep them out
    # of the other tests.
    for key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_AGENT_ENGINE_LOCATION",
        "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS",
    ):
        monkeypatch.delenv(key, raising=False)
    return api


@pytest.fixture
def ge(monkeypatch) -> list[dict[str, Any]]:
    """Gemini Enterprise, recording the token each call was made with."""
    monkeypatch.setattr(config, "WORKSPACE_ENGINE", GE_ENGINE)
    monkeypatch.setattr(config, "WORKSPACE_SOFT_DEADLINE_S", 5.0)
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any):
        calls.append(kwargs)
        acc = workspace._Accumulator()
        for element in workspace.JsonArraySplitter().push(_body()):
            acc.fold(element)
        return acc.result(), False

    monkeypatch.setattr(workspace, "stream_assist", fake)
    return calls


def _app() -> AdkApp:
    app = AdkApp(
        agent=AskTurn(name="ask", sub_agents=[_tree()]),
        app_name=ENGINE_ID,
        session_service_builder=lambda: VertexAiSessionService(
            project="p", location="us-central1", agent_engine_id=ENGINE_ID
        ),
    )
    app.set_up()
    return app


async def _turn(app: AdkApp, session_id: str, state_delta: dict[str, Any] | None) -> dict:
    """One turn exactly as query_service/engine.py sends it."""
    kwargs = {"state_delta": state_delta} if state_delta is not None else {}
    events = [
        e
        async for e in app.async_stream_query(
            message="how is chevy?", user_id="person-1", session_id=session_id, **kwargs
        )
    ]
    [answer] = [
        e["custom_metadata"][ASK_ANSWER_KEY]
        for e in events
        if ASK_ANSWER_KEY in (e.get("custom_metadata") or {})
    ]
    return answer


async def test_the_agent_reads_the_temp_token_and_the_store_never_sees_it(store, ge):
    app = _app()
    session = await app.async_create_session(user_id="person-1")

    answer = await _turn(
        app, session["id"], {workspace.TOKEN_STATE_KEY: TOKEN, "probe_persisted": "kept"}
    )

    # Read during the turn: the Workspace call ran as the person.
    assert [c["token"] for c in ge] == [TOKEN]
    assert answer["sources"]["workspace"]["status"] == "ok"
    # The ordinary key was stored, so the store records what ADK sends...
    assert store.sessions["1"]["state"]["probe_persisted"] == "kept"
    assert store.sessions["1"]["state"][workspace.SESSION_STATE_KEY] == f"{GE_ENGINE}/sessions/s1"
    # ...and the token is in no request and nothing stored.
    assert TOKEN not in store.wire()
    assert workspace.TOKEN_STATE_KEY not in store.wire()


async def test_the_next_turn_has_no_token_unless_it_is_sent_again(store, ge):
    app = _app()
    session = await app.async_create_session(user_id="person-1")
    await _turn(app, session["id"], {workspace.TOKEN_STATE_KEY: TOKEN})

    answer = await _turn(app, session["id"], None)

    assert len(ge) == 1  # no second call: there was no token to make it with
    assert answer["sources"]["workspace"]["status"] == "error"
    assert answer["sources"]["workspace"]["error"]["code"] == "WORKSPACE_NOT_CONNECTED"
    assert TOKEN not in store.wire()


def test_the_service_sends_the_key_the_agent_reads():
    assert WORKSPACE_TOKEN_KEY == workspace.TOKEN_STATE_KEY
    assert WORKSPACE_TOKEN_KEY.startswith("temp:")

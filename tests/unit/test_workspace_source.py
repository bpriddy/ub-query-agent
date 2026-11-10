"""
Workspace as a `/ask` source (gub-agent#90): Gemini Enterprise's streamAssist,
called as the person, beside the root.

Pinned here:

  * the stream: a JSON array split as it arrives, whatever the chunking and
    whatever brackets the answer text holds;
  * grounding: each passage becomes a fact naming the item it was read from;
    a passage whose references cannot all be resolved is left out;
  * the call: a rejected session is retried once, cold; the engine's reasons
    map to codes the person can act on;
  * a turn: not configured → skipped; no token → WORKSPACE_NOT_CONNECTED with
    the connect link; connected → workspace facts in the answer and the
    Gemini Enterprise session kept for the next turn, the token never; too
    slow → timeout.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agents import workspace
from gub_agent.agents.ask import ASK_ANSWER_KEY, PROGRESS_KEY, AskTurn
from gub_agent.schemas.ask import AskResponse

from .test_ask_contract import _tree

ENGINE = "projects/p/locations/global/collections/default_collection/engines/dev-ge"
DRIVE = (
    "projects/p/locations/global/collections/default_collection/dataStores/gdrive_1_google_drive"
)
MAIL = "projects/p/locations/global/collections/default_collection/dataStores/gmail_2_google_mail"
TOKEN = "ya29.turn-only-google-token"


def _ref(store: str, doc: str, title: str, uri: str) -> dict:
    return {
        "documentMetadata": {
            "document": f"{store}/branches/0/documents/{doc}",
            "title": title,
            "uri": uri,
        }
    }


CHUNKS = [
    {"sessionInfo": {"session": f"{ENGINE}/sessions/s1"}},
    {
        "answer": {
            "replies": [
                {
                    "groundedContent": {
                        "content": {"text": "Launch is Dec 2 {per the timeline}. "},
                        "textGroundingMetadata": {
                            "segments": [
                                {
                                    "startIndex": "0",
                                    "endIndex": "17",
                                    "text": "Launch is Dec 2",
                                    "referenceIndices": [0],
                                }
                            ],
                            "references": [
                                _ref(
                                    DRIVE, "doc1", "Holiday timeline", "https://docs.google.com/d/1"
                                )
                            ],
                        },
                    }
                }
            ]
        }
    },
    {
        "answer": {
            "state": "SUCCEEDED",
            "replies": [
                {
                    "groundedContent": {
                        "content": {"text": "Ada confirmed it [by mail]."},
                        "textGroundingMetadata": {
                            "segments": [
                                {"text": "Ada confirmed it", "referenceIndices": [0]},
                                # Index 7 resolves nowhere: the passage is dropped.
                                {"text": "orphan passage", "referenceIndices": [7]},
                            ],
                            "references": [
                                _ref(MAIL, "m9", "Re: launch", "https://mail.google.com/x")
                            ],
                        },
                    }
                }
            ],
        }
    },
]


def _body() -> str:
    return json.dumps(CHUNKS)


# ── the stream ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [1, 7, 64, 100_000])
def test_the_array_splits_into_its_elements_however_it_arrives(size):
    splitter = workspace.JsonArraySplitter()
    body = _body()
    out = []
    for i in range(0, len(body), size):
        out.extend(splitter.push(body[i : i + size]))
    out.extend(splitter.flush())
    assert out == CHUNKS


def test_a_bare_object_is_one_element():
    splitter = workspace.JsonArraySplitter()
    assert splitter.push('{"a": "]["}') == [{"a": "]["}]


# ── the call ─────────────────────────────────────────────────────────────────


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_a_call_returns_text_session_and_grounded_passages():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=_body())

    result, rejected = await workspace.stream_assist(
        engine=ENGINE,
        data_store_ids=["gdrive_1_google_drive"],
        token=TOKEN,
        query="when is launch?",
        session=None,
        client=_client(handler),
    )

    assert seen["auth"] == f"Bearer {TOKEN}"
    assert seen["url"] == (
        f"https://discoveryengine.googleapis.com/v1alpha/{ENGINE}"
        "/assistants/default_assistant:streamAssist"
    )
    assert seen["body"]["toolsSpec"]["vertexAiSearchSpec"]["dataStoreSpecs"] == [
        {"dataStore": DRIVE}
    ]
    assert "session" not in seen["body"]
    assert not rejected
    assert result.session == f"{ENGINE}/sessions/s1"
    assert result.text == "Launch is Dec 2 {per the timeline}. Ada confirmed it [by mail]."
    assert [p.text for p in result.passages] == ["Launch is Dec 2", "Ada confirmed it"]

    facts = workspace.facts_of(result)
    assert [(f.evidence_id, f.source, f.ref.type, f.ref.id, f.ref.title) for f in facts] == [
        ("workspace:1", "workspace", "doc", "doc1", "Holiday timeline"),
        ("workspace:2", "workspace", "mail", "m9", "Re: launch"),
    ]
    assert facts[0].ref.uri == "https://docs.google.com/d/1"


async def test_a_rejected_session_is_retried_once_cold():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "session" in body:
            return httpx.Response(
                404, json={"error": {"status": "NOT_FOUND", "message": "Session not found"}}
            )
        return httpx.Response(200, text=_body())

    result, rejected = await workspace.stream_assist(
        engine=ENGINE,
        data_store_ids=[],
        token=TOKEN,
        query="q",
        session=f"{ENGINE}/sessions/old",
        client=_client(handler),
    )
    assert rejected
    assert [("session" in b) for b in bodies] == [True, False]
    assert result.session == f"{ENGINE}/sessions/s1"


async def test_a_bad_request_is_not_mistaken_for_a_rejected_session():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT"}})

    with pytest.raises(workspace.StreamAssistError) as err:
        await workspace.stream_assist(
            engine=ENGINE,
            data_store_ids=[],
            token=TOKEN,
            query="q",
            session=f"{ENGINE}/sessions/old",
            client=_client(handler),
        )
    assert err.value.reason == "INVALID_ARGUMENT"
    assert calls == [1]


@pytest.mark.parametrize(
    ("status", "reason", "code", "has_link"),
    [
        (401, "USER_UNAUTHENTICATED", "WORKSPACE_SETUP_REQUIRED", True),
        (403, "LICENSE_INACTIVE", "WORKSPACE_NO_LICENSE", False),
        (401, None, "WORKSPACE_TOKEN_REJECTED", False),
        (503, "UNAVAILABLE", "WORKSPACE_UNAVAILABLE", False),
    ],
)
def test_the_engines_reasons_map_to_codes_the_person_can_act_on(status, reason, code, has_link):
    err = workspace.source_error(workspace.StreamAssistError(status, reason, ""), "https://ge/app")
    assert err.code == code
    assert (err.link is not None) == has_link


def test_a_reason_is_read_from_the_error_details():
    raw = json.dumps(
        [{"error": {"status": "PERMISSION_DENIED", "details": [{"reason": "LICENSE_INACTIVE"}]}}]
    )
    assert workspace.error_reason(raw) == "LICENSE_INACTIVE"


@pytest.mark.parametrize(
    ("store", "uri", "kind"),
    [
        ("gchat_1_google_chat", None, "chat"),
        ("google-calendar_1_google_calendar", None, "event"),
        ("gmail_1_google_mail", None, "mail"),
        (None, "https://www.drive.google.com/x", "doc"),
        (None, "https://example.com", "other"),
    ],
)
def test_an_items_type_comes_from_its_store_then_its_host(store, uri, kind):
    assert workspace.ref_type(store, uri) == kind


# ── a turn ───────────────────────────────────────────────────────────────────


@pytest.fixture
def ge(monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_ENGINE", ENGINE)
    monkeypatch.setattr(config, "WORKSPACE_CONNECT_URL", "https://qa/connect")
    monkeypatch.setattr(config, "WORKSPACE_SOFT_DEADLINE_S", 5.0)
    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        splitter = workspace.JsonArraySplitter()
        acc = workspace._Accumulator()
        for element in splitter.push(_body()):
            acc.fold(element)
        return acc.result(), False

    monkeypatch.setattr(workspace, "stream_assist", fake)
    return calls


async def _turn(state_delta: dict | None = None, session_id: str | None = None, runner=None):
    runner = runner or InMemoryRunner(
        agent=AskTurn(name="ask", sub_agents=[_tree()]), app_name="gub"
    )
    if session_id is None:
        session = await runner.session_service.create_session(app_name="gub", user_id="u")
        session_id = session.id
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
    streamed = [
        e
        async for e in runner.run_async(
            user_id="u", session_id=session_id, new_message=message, state_delta=state_delta
        )
    ]
    stored = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session_id
    )
    answer = AskResponse.model_validate(streamed[-1].custom_metadata[ASK_ANSWER_KEY])
    progress = [
        e.custom_metadata[PROGRESS_KEY]
        for e in streamed
        if PROGRESS_KEY in (e.custom_metadata or {})
    ]
    return answer, progress, stored, runner


async def test_not_configured_workspace_is_skipped(monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_ENGINE", "")
    answer, _, _, _ = await _turn({workspace.TOKEN_STATE_KEY: TOKEN})
    assert answer.sources["workspace"].status == "skipped"


async def test_no_token_is_not_connected_with_the_connect_link(ge):
    answer, progress, _, _ = await _turn()
    report = answer.sources["workspace"]
    assert report.status == "error"
    assert report.error.code == "WORKSPACE_NOT_CONNECTED"
    assert report.error.link == "https://qa/connect"
    assert answer.kind == "answer"  # a Workspace failure never fails the answer
    assert ge == []
    assert {"step": "source", "source": "workspace", "status": "error"}.items() <= progress[
        -1
    ].items()


async def test_a_connected_turn_carries_workspace_facts_and_keeps_the_session(ge):
    answer, progress, stored, runner = await _turn({workspace.TOKEN_STATE_KEY: TOKEN})

    assert ge[0]["token"] == TOKEN
    assert ge[0]["query"] == "how is chevy?"
    assert ge[0]["session"] is None
    assert answer.sources["workspace"].status == "ok"
    ws = [f for f in answer.facts if f.source == "workspace"]
    assert [f.ref.title for f in ws] == ["Holiday timeline", "Re: launch"]
    assert {f.evidence_id for f in ws} <= set(answer.citations)
    assert progress[1] == {"step": "lookup", "text": "Searching your Workspace"}
    assert {"step": "source", "text": "Found 2 items in Workspace"}.items() <= progress[-1].items()

    assert stored.state[workspace.SESSION_STATE_KEY] == f"{ENGINE}/sessions/s1"
    saved = json.dumps([e.model_dump(mode="json") for e in stored.events]) + json.dumps(
        stored.state
    )
    assert TOKEN not in saved

    # The next turn of the conversation continues the same Gemini Enterprise session.
    await _turn({workspace.TOKEN_STATE_KEY: TOKEN}, session_id=stored.id, runner=runner)
    assert ge[1]["session"] == f"{ENGINE}/sessions/s1"


async def test_a_slow_workspace_is_a_timeout_and_is_cancelled(ge, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_SOFT_DEADLINE_S", 0.05)
    cancelled = asyncio.Event()

    async def slow(**kwargs):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(workspace, "stream_assist", slow)
    answer, progress, _, _ = await _turn({workspace.TOKEN_STATE_KEY: TOKEN})
    await asyncio.sleep(0)
    assert answer.sources["workspace"].status == "timeout"
    assert cancelled.is_set()
    assert progress[-1]["text"] == "Workspace didn't answer in time"


async def test_a_workspace_error_is_reported_and_the_answer_stands(ge, monkeypatch):
    async def refused(**kwargs):
        raise workspace.StreamAssistError(401, "USER_UNAUTHENTICATED", "")

    monkeypatch.setattr(workspace, "stream_assist", refused)
    answer, _, _, _ = await _turn({workspace.TOKEN_STATE_KEY: TOKEN})
    assert answer.kind == "answer"
    assert answer.sources["workspace"].error.code == "WORKSPACE_SETUP_REQUIRED"

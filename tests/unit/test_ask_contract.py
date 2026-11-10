"""
The `/ask` contract (gub-agent#88): `AskResponse`, `ProgressEvent`, and the
`AskTurn` that streams them.

Pinned here:

  * the shape: every source reported, `error` exactly on kind="error", a
    workspace fact carries a ref, only workspace claims are dropped, a source's
    error exactly on status="error";
  * assembly: a fact's source and record type come from the tool behind its
    evidence id, records are the cited UB records once each, with the name
    the evidence carried;
  * a turn on a real Runner: progress lines first, the answer last, both
    partial and authored `ask`; the wrapped root's events pass through; the
    stored session is the same as the bare root's;
  * failure: the caller still gets an answer event, kind="error", with the
    reason.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from pydantic import ValidationError

from gub_agent.agents.ask import (
    ASK_ANSWER_KEY,
    ASK_AUTHOR,
    PROGRESS_KEY,
    AskTurn,
    SourceLog,
    build_response,
)
from gub_agent.agents.evidence_index import evidence_index, record_evidence, reset_evidence_index
from gub_agent.agents.format_gate import FormatGate
from gub_agent.config import AGENT_NAME
from gub_agent.schemas.answer import AnswerPayload
from gub_agent.schemas.ask import AskResponse, ProgressEvent, SourceReport

from .test_format_gate import _scripted, _valid_dict

ALL_SKIPPED = {s: {"status": "skipped"} for s in ("ub", "runn", "workspace")}


def _answer(**over) -> dict:
    base = {**_valid_dict(), "sources": {**ALL_SKIPPED, "ub": {"status": "ok"}}}
    base.update(over)
    return base


# ── the shape ────────────────────────────────────────────────────────────────


def test_a_formatter_payload_plus_sources_is_a_valid_answer():
    response = AskResponse.model_validate(_answer())
    assert response.kind == "answer"
    assert response.facts[0].source == "ub"


def test_every_source_is_reported():
    with pytest.raises(ValidationError, match="missing: workspace"):
        AskResponse.model_validate(
            _answer(sources={"ub": {"status": "ok"}, "runn": {"status": "ok"}})
        )


def test_kind_error_carries_the_reason_and_nothing_else():
    error = {
        "code": "NO_ACCESS",
        "message": "You have no grant on account a1",
        "record": {"type": "account", "id": "a1"},
    }
    response = AskResponse.model_validate(
        {
            "kind": "error",
            "headline": "No access to that account",
            "error": error,
            "sources": ALL_SKIPPED,
        }
    )
    assert response.error.record.id == "a1"
    with pytest.raises(ValidationError, match="kind is"):
        AskResponse.model_validate({"kind": "error", "headline": "x", "sources": ALL_SKIPPED})
    with pytest.raises(ValidationError, match="kind is"):
        AskResponse.model_validate(_answer(error=error))
    with pytest.raises(ValidationError, match="no blocks or citations"):
        AskResponse.model_validate(_answer(kind="error", error=error))


def test_a_workspace_fact_names_the_item_it_came_from():
    fact = {"evidence_id": "gemini:1", "value": "launch Dec 2", "source": "workspace"}
    with pytest.raises(ValidationError, match="needs a ref"):
        AskResponse.model_validate(_answer(citations=["gemini:1"], facts=[fact]))
    fact["ref"] = {"type": "doc", "id": "1AbC", "title": "Holiday timeline"}
    AskResponse.model_validate(_answer(citations=["gemini:1"], facts=[fact]))


def test_only_workspace_claims_are_dropped():
    dropped = [
        {"value": "launch Nov 28", "conflicts_with": {"source": "ub", "field": "launchDate"}}
    ]
    ok = _answer(
        sources={
            **ALL_SKIPPED,
            "ub": {"status": "ok"},
            "workspace": {"status": "ok", "dropped": dropped},
        }
    )
    assert AskResponse.model_validate(ok).sources["workspace"].dropped[0].value == "launch Nov 28"
    bad = _answer(sources={**ALL_SKIPPED, "ub": {"status": "ok", "dropped": dropped}})
    with pytest.raises(ValidationError, match="only workspace"):
        AskResponse.model_validate(bad)


def test_a_source_error_is_carried_exactly_on_status_error():
    SourceReport(
        status="error", error={"code": "RUNN_NO_GRANT", "message": "needs the Runn reader grant"}
    )
    with pytest.raises(ValidationError):
        SourceReport(status="error")
    with pytest.raises(ValidationError):
        SourceReport(status="ok", error={"code": "X", "message": "y"})


def test_progress_lines_are_short_and_a_source_step_names_its_source():
    ProgressEvent(step="source", text="Checked Runn", source="runn", status="ok")
    with pytest.raises(ValidationError):
        ProgressEvent(step="source", text="Checked Runn")
    with pytest.raises(ValidationError):
        ProgressEvent(step="lookup", text="x", source="ub", status="ok")
    with pytest.raises(ValidationError):
        ProgressEvent(step="lookup", text="x" * 161)


# ── assembly ─────────────────────────────────────────────────────────────────


def _index(invocation_id: str) -> dict:
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    record_evidence(
        SimpleNamespace(name="get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "Q3 push", "status": "live"},
    )
    return evidence_index(invocation_id)


def test_facts_and_records_come_from_the_tools_behind_the_evidence():
    payload = AnswerPayload.model_validate(_valid_dict())  # cites get_campaign:c1:status
    log = SourceLog()
    log.called("get_campaign")
    log.answered("get_campaign", {"id": "c1"})
    response = build_response(payload, _index("inv-build"), log, "conv-1")

    [fact] = response.facts
    assert (fact.source, fact.entity_type, fact.entity_id, fact.value) == (
        "ub",
        "campaign",
        "c1",
        "live",
    )
    assert [r.model_dump(exclude_none=True) for r in response.records] == [
        {"type": "campaign", "id": "c1", "name": "Q3 push"}
    ]
    assert {s: r.status for s, r in response.sources.items()} == {
        "ub": "ok",
        "runn": "skipped",
        "workspace": "skipped",
    }
    assert response.conversation_id == "conv-1"
    # The formatter's prose is carried untouched.
    assert response.headline == payload.headline and response.blocks == payload.blocks


# ── a turn, on a real Runner ─────────────────────────────────────────────────

TEXT = "chevy is in good shape — the Q3 push is live."


class _Executor(BaseAgent):
    """The executor's part: a tool call, its evidence, then its prose."""

    async def _run_async_impl(self, ctx):
        reset_evidence_index(ctx)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name="get_campaign", args={"id": "c1"}
                        )
                    )
                ],
            ),
        )
        record_evidence(
            SimpleNamespace(name="get_campaign"),
            {},
            ctx,
            {"id": "c1", "name": "Q3 push", "status": "live", "budget": 1200000},
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=TEXT)]),
        )


class _Broken(BaseAgent):
    async def _run_async_impl(self, ctx):
        raise RuntimeError("UB answered 503")
        yield  # pragma: no cover


def _tree() -> SequentialAgent:
    return SequentialAgent(
        name="gub_root",
        sub_agents=[
            _Executor(name=AGENT_NAME),
            FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])]),
        ],
    )


async def _run(root: BaseAgent) -> tuple[list[Event], list[Event], dict]:
    runner = InMemoryRunner(agent=root, app_name="gub")
    session = await runner.session_service.create_session(app_name="gub", user_id="u")
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
    streamed = [
        e async for e in runner.run_async(user_id="u", session_id=session.id, new_message=message)
    ]
    stored = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session.id
    )
    return streamed, list(stored.events), dict(stored.state)


def _stored_shape(events: list[Event]) -> list[tuple]:
    return [(e.author, e.content.model_dump() if e.content else None) for e in events]


def _ours(events: list[Event]) -> list[Event]:
    return [e for e in events if e.author == ASK_AUTHOR]


async def test_a_turn_streams_progress_then_one_answer():
    streamed, _, _ = await _run(AskTurn(name="ask", sub_agents=[_tree()]))
    ours = _ours(streamed)

    assert all(e.partial for e in ours)
    progress = [
        e.custom_metadata[PROGRESS_KEY] for e in ours if PROGRESS_KEY in (e.custom_metadata or {})
    ]
    assert progress == [
        {"step": "understand", "text": "Reading the question"},
        {"step": "lookup", "text": "Reading a campaign"},
        {"step": "source", "text": "Found 1 record in UB", "source": "ub", "status": "ok"},
    ]
    assert streamed[0] is ours[0]  # the first thing the caller sees
    last = streamed[-1]
    assert last is ours[-1] and ASK_ANSWER_KEY in last.custom_metadata
    answer = AskResponse.model_validate(last.custom_metadata[ASK_ANSWER_KEY])
    assert answer.kind == "answer"
    assert answer.records[0].id == "c1"
    # No tool arguments anywhere in what this agent streams.
    assert '"id": "c1"' not in json.dumps([e.custom_metadata for e in ours][:-1])


async def test_the_wrapped_root_streams_and_stores_as_it_did_bare():
    bare_stream, bare_stored, bare_state = await _run(_tree())
    ask_stream, ask_stored, ask_state = await _run(AskTurn(name="ask", sub_agents=[_tree()]))

    passed = [e for e in ask_stream if e.author != ASK_AUTHOR]
    assert [e.author for e in passed] == [e.author for e in bare_stream]
    assert _stored_shape(ask_stored) == _stored_shape(bare_stored)
    assert _ours(ask_stored) == []
    assert ask_state == bare_state


async def test_a_failed_turn_still_ends_with_an_answer_giving_the_reason():
    streamed, _, _ = await _run(AskTurn(name="ask", sub_agents=[_Broken(name="gub_root")]))
    last = streamed[-1]
    answer = AskResponse.model_validate(last.custom_metadata[ASK_ANSWER_KEY])
    assert answer.kind == "error"
    assert answer.error.code == "INTERNAL"
    assert "UB answered 503" in answer.error.message


def test_every_tool_has_a_progress_phrase():
    from gub_agent.agents.ask import TOOL_PHRASES
    from gub_agent.tools import ALL_TOOLS

    assert {tool.__name__ for tool in ALL_TOOLS} <= set(TOOL_PHRASES)

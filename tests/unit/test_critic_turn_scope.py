"""
The critic's deterministic facts are about THIS turn (thread-topics).

`_executor_made_tool_call` and `_last_executor_text` read session events, and
session events are never trimmed: a thread keeps every turn it has ever had.
Unfiltered, both scans reached back into earlier turns —

- a tool call from ANY earlier turn made the critic read "TOOL CALL THIS TURN:
  yes", which switches off its "you answered from memory" guard for the rest of
  the session;
- a turn whose executor produced no text read the PREVIOUS answer instead, so
  the format gate could ship it as this turn's and CriticGate could skip the
  critic because an earlier turn had abstained.

Both now filter on the current invocation id. Built on a REAL ADK
InvocationContext (tests/helpers.py), because the invocation id is the thing
under test and a fake would supply whatever the test assumed.
"""

from __future__ import annotations

import json

from google.adk.agents import BaseAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agents.critic import (
    CriticGate,
    _critic_instruction,
    _executor_made_tool_call,
    _last_executor_text,
)
from gub_agent.config import AGENT_NAME
from tests.helpers import invocation_ctx

EARLIER = "inv-earlier"
NOW = "inv-now"


def _text(author: str, text: str, inv: str) -> Event:
    role = "user" if author == "user" else "model"
    return Event(
        invocation_id=inv,
        author=author,
        content=genai_types.Content(role=role, parts=[genai_types.Part(text=text)]),
    )


def _call(inv: str) -> Event:
    return Event(
        invocation_id=inv,
        author=AGENT_NAME,
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(function_call=genai_types.FunctionCall(name="find", args={}))],
        ),
    )


def _earlier_turn_with_a_tool_call() -> list[Event]:
    return [_text("user", "how is chevy doing?", EARLIER), _call(EARLIER)]


# ── "was a tool called this turn" ────────────────────────────────────────────


async def test_an_earlier_turns_tool_call_is_not_this_turns():
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            *_earlier_turn_with_a_tool_call(),
            _text(AGENT_NAME, "draft from an earlier turn", EARLIER),
            _text("user", "and how many campaigns are live?", NOW),
            _text(AGENT_NAME, "There are 12 live campaigns.", NOW),  # from memory
        ],
    )
    assert _executor_made_tool_call(ctx) is False
    # …and the critic is told so, which is what keeps its guard armed.
    assert "TOOL CALL THIS TURN: no" in _critic_instruction(ReadonlyContext(ctx))


async def test_this_turns_tool_call_still_counts():
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[*_earlier_turn_with_a_tool_call(), _text("user", "and Q3?", NOW), _call(NOW)],
    )
    assert _executor_made_tool_call(ctx) is True
    assert "TOOL CALL THIS TURN: yes" in _critic_instruction(ReadonlyContext(ctx))


# ── the executor's text ──────────────────────────────────────────────────────


async def test_the_last_executor_text_is_this_turns_or_nothing():
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            _text(AGENT_NAME, "the earlier answer", EARLIER),
            _text("user", "and Q3?", NOW),
            _call(NOW),  # the executor died after its call: no text this turn
        ],
    )
    # Not "the earlier answer": the format gate treats "" as "emit nothing",
    # which is the honest trace of a failed run.
    assert _last_executor_text(ctx) == ""

    ctx_ok = await invocation_ctx(
        invocation_id=NOW,
        events=[_text(AGENT_NAME, "the earlier answer", EARLIER), _text(AGENT_NAME, "Q3 is", NOW)],
    )
    assert _last_executor_text(ctx_ok) == "Q3 is"


class _RecordingCritic(BaseAgent):
    ran: list = []

    async def _run_async_impl(self, ctx):
        self.ran.append(ctx.invocation_id)
        yield Event(invocation_id=ctx.invocation_id, author=self.name)


async def test_an_earlier_abstention_does_not_skip_this_turns_critic():
    """CriticGate's deterministic pass is for THIS turn's NO_COMPANY_RECORDS.
    An earlier turn's must not buy the current draft a free "sufficient"."""
    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(name="critic_gate", sub_agents=[critic])
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            _text(AGENT_NAME, "NO_COMPANY_RECORDS", EARLIER),
            _text("user", "and the other account?", NOW),
            _call(NOW),
        ],
    )

    [e async for e in gate.run_async(ctx)]

    assert critic.ran == [NOW]


async def test_this_turns_abstention_still_skips_the_critic():
    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(name="critic_gate", sub_agents=[critic])
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            _text(AGENT_NAME, "the earlier answer", EARLIER),
            _text(AGENT_NAME, "NO_COMPANY_RECORDS", NOW),
        ],
    )

    events = [e async for e in gate.run_async(ctx)]

    assert critic.ran == []
    assert events[0].actions.state_delta["critic_verdict"]["sufficient"] is True


# ── an abstain payload with no tool call is a draft from memory ─────────────


async def _gate_on_abstain_payload(events: list[Event]) -> tuple[_RecordingCritic, list[Event]]:
    """The gate right after a format gate that emitted an abstain payload this
    pass — as an event, which is where the gate reads it from."""
    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(
        name="critic_gate", sub_agents=[critic], payload_authors=("format_gate", "formatter")
    )
    payload = Event(
        invocation_id=NOW,
        author="format_gate",
        actions=EventActions(
            state_delta={"answer_payload": {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}}
        ),
    )
    ctx = await invocation_ctx(invocation_id=NOW, events=[*events, payload])
    return critic, [e async for e in gate.run_async(ctx)]


async def test_an_abstain_payload_with_no_tool_call_is_sent_back_in_code():
    """The live 2026-09-24 case: the executor copied its previous answer
    without querying and the format gate found no evidence and abstained.

    The gate sends it back for a re-query itself. Not via the critic LLM:
    live, the critic read the earlier turns' identical answers and passed the
    copy both times it was asked."""
    critic, events = await _gate_on_abstain_payload(
        [
            *_earlier_turn_with_a_tool_call(),
            _text(AGENT_NAME, "Here is what's new: 3 new hires, 12 live campaigns.", EARLIER),
            _text("user", "whats new", NOW),
            _text(AGENT_NAME, "Here is what's new: 3 new hires, 12 live campaigns.", NOW),
        ]
    )

    assert critic.ran == []  # an EARLIER turn's call does not count as looking
    [event] = events
    verdict = event.actions.state_delta["critic_verdict"]
    assert verdict["sufficient"] is False  # the escalator does not exit: retry
    assert "NO tool call" in verdict["feedback"]
    # Authored as the critic, verdict as text: the executor's retry reads its
    # feedback from "[critic] said: …", and the bot restarts the streamed pass
    # on a sufficient=false from author "critic".
    assert event.author == "critic"
    assert json.loads(event.content.parts[0].text) == verdict


async def test_a_second_from_memory_abstain_in_the_same_turn_passes():
    """The retry answered from memory too (a question about the conversation
    itself): the abstain ships, once, instead of a second send-back."""
    sent_back = Event(
        invocation_id=NOW,
        author="critic",
        actions=EventActions(state_delta={"critic_verdict": {"sufficient": False}}),
    )
    critic, events = await _gate_on_abstain_payload(
        [
            _text("user", "what did I just ask?", NOW),
            _text(AGENT_NAME, "You asked what's new.", NOW),
            sent_back,
            _text(AGENT_NAME, "You asked what's new.", NOW),
        ]
    )

    assert critic.ran == []
    assert events[0].actions.state_delta["critic_verdict"]["sufficient"] is True


async def test_an_earlier_turns_send_back_does_not_count():
    earlier = Event(
        invocation_id=EARLIER,
        author="critic",
        actions=EventActions(state_delta={"critic_verdict": {"sufficient": False}}),
    )
    critic, events = await _gate_on_abstain_payload(
        [earlier, _text("user", "whats new", NOW), _text(AGENT_NAME, "Copied.", NOW)]
    )

    assert events[0].actions.state_delta["critic_verdict"]["sufficient"] is False


async def test_an_abstain_payload_after_a_tool_call_still_skips_the_critic(monkeypatch):
    # One pass, as before #102: the searched abstain is not sent back to confirm.
    monkeypatch.setattr(config, "CONFIRM_ABSTENTION", False)
    critic, events = await _gate_on_abstain_payload(
        [
            _text("user", "how is the acme account?", NOW),
            _call(NOW),
            _text(AGENT_NAME, "GUB has no account by that name.", NOW),
        ]
    )

    assert critic.ran == []
    assert events[0].actions.state_delta["critic_verdict"]["sufficient"] is True


async def test_the_bare_marker_skips_the_critic_without_a_tool_call():
    """The executor's own "nothing to look up" — no tool call needed."""
    critic, events = await _gate_on_abstain_payload(
        [_text("user", "what's the weather?", NOW), _text(AGENT_NAME, "NO_COMPANY_RECORDS", NOW)]
    )

    assert critic.ran == []
    assert events[0].actions.state_delta["critic_verdict"]["sufficient"] is True


async def test_the_executor_retry_sees_the_send_back_feedback():
    """Nothing injects state into the executor's prompt, so the feedback only
    reaches the retry if ADK renders the send-back event into its request.
    Built through ADK's own content builder, as the executor's retry sees it."""
    from google.adk.flows.llm_flows.contents import _get_contents

    critic, [send_back] = await _gate_on_abstain_payload(
        [_text("user", "whats new", NOW), _text(AGENT_NAME, "Copied from memory.", NOW)]
    )
    events = [
        _text("user", "whats new", NOW),
        _text(AGENT_NAME, "Copied from memory.", NOW),
        send_back,
    ]

    contents = _get_contents(None, events, AGENT_NAME)
    rendered = " ".join(p.text or "" for c in contents for p in (c.parts or []))
    assert "You made NO tool call this turn" in rendered


# ── the re-query's "already sent back" fact is about THIS turn too ───────────


def _sent_back(inv: str, author: str = "critic") -> Event:
    verdict = {"sufficient": False, "reason": "from memory", "feedback": "query GUB"}
    return Event(
        invocation_id=inv,
        author=author,
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=json.dumps(verdict))]
        ),
        actions=EventActions(state_delta={"critic_verdict": verdict}),
    )


async def test_an_earlier_turns_send_back_is_not_this_turns():
    from gub_agent.agents.critic import _sent_back_this_turn

    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[_sent_back(EARLIER), _text(AGENT_NAME, "from memory", NOW)],
    )
    assert _sent_back_this_turn(ctx) is False
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[_sent_back(EARLIER), _text(AGENT_NAME, "from memory", NOW), _sent_back(NOW)],
    )
    assert _sent_back_this_turn(ctx) is True


async def test_a_send_back_counts_only_under_the_critics_own_name():
    from gub_agent.agents.critic import _sent_back_this_turn

    ctx = await invocation_ctx(invocation_id=NOW, events=[_sent_back(NOW, author="other")])
    assert _sent_back_this_turn(ctx) is False
    assert _sent_back_this_turn(ctx, "other") is True

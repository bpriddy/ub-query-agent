"""
The critic beside the format gate (CRITIC_PARALLEL) — agent.py:build_deep_agent.

The deep loop used to run [executor, format_gate, critic_gate, escalator] one
after another, although the critic judges the executor's tool coverage and
never reads the formatter's payload. The parallel wiring runs the critic LLM
while the formatter renders, holds its events, and lets a resolver named
`critic_gate` make CriticGate's decisions after the join.

What has to stay true, and is pinned here on a REAL ADK runner (InMemoryRunner
over the real LoopAgent / ParallelAgent / CriticResolver / escalator, with
stand-ins only for the three model-backed agents):

- the two really overlap, and the critic's branch still sees the executor's
  work (and not the formatter's — ADK's branch isolation);
- the loop exits on the escalator's escalate, and an insufficient verdict still
  buys the executor a second pass that can read the feedback;
- the abstain pass and the from-memory re-query are decided on THIS pass's
  payload — never on a stale one left in state by an earlier turn — and a held
  verdict they overrule never reaches the stream, the state or the bot;
- the stream keeps its order and its authors: the last payload event is the
  format gate's, the verdict follows it;
- the loop's last iteration makes no critic MODEL call (counted on a stub
  model under a real LlmAgent), and still ends on the escalator with a
  verdict authored `critic`, as the bot's per-pass count expects;
- CRITIC_PARALLEL=0 builds the serial tree it replaced, and makes the same
  decisions as the parallel one on the same turns;
- a critic call whose verdict the pass has already settled in code (an abstain
  payload) is cancelled instead of waited for, and one whose verdict is needed
  is not: the stalled-critic tests at the end, whose critic waits on an event
  that only a LATER step of the turn sets, so the old wiring deadlocks on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, ParallelAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event, EventActions
from google.adk.flows.llm_flows.contents import _get_contents
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agent import build_deep_agent
from gub_agent.agents import critic as critic_module
from gub_agent.agents.critic import (
    CriticGate,
    CriticResolver,
    CriticVerdict,
    EscalateIfSufficient,
    SpeculativeCritic,
)
from gub_agent.agents.format_gate import FormatGate
from gub_agent.config import AGENT_NAME, GEMINI_MODEL
from gub_agent.models import VendorRouter, bind_model_call

REPO = Path(__file__).resolve().parents[2]

ANSWER = {"kind": "answer", "headline": "chevy is live", "citations": ["org_query:a1"]}
ABSTAIN = {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}
SUFFICIENT = {
    "info_sufficient": True,
    "answer_satisfies": True,
    "sufficient": True,
    "reason": "LLM: covered",
    "feedback": "",
}
INSUFFICIENT = {
    "info_sufficient": False,
    "answer_satisfies": False,
    "sufficient": False,
    "reason": "LLM: the Q3 campaign was never read",
    "feedback": "call get_campaign for the Q3 push",
}
FINAL_PASS_REASON = (
    "Deterministic pass: final loop iteration, no retry left to request (no critic LLM run)."
)


@pytest.fixture(autouse=True)
def _formatter_always_runs(monkeypatch):
    """The tests below describe the pipeline with the from-memory shortcut OFF,
    the code default; the deployed value is 1. The shortcut's own tests set it
    themselves, so a shell that sources the deploy env cannot turn the rest red."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", False, raising=False)


@pytest.fixture(autouse=True)
def _skip_final_pass(monkeypatch):
    """The default, pinned: most tests here describe the last-pass skip, and the
    one about its rollback turns it off itself — whatever the environment says."""
    monkeypatch.setattr(config, "CRITIC_SKIP_FINAL_PASS", True)


# ── stand-ins for the model-backed agents ────────────────────────────────────


def _text_event(ctx, author: str, text: str, *, partial: bool = False) -> Event:
    return Event(
        invocation_id=ctx.invocation_id,
        author=author,
        branch=ctx.branch,
        partial=partial or None,
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
    )


class _Executor(BaseAgent):
    """Per pass: an optional tool call + its result, then a text answer. The
    tool call is `org_query` for True, or the tool named by a string."""

    passes: list = []  # [(made_tool_call, text)], one per pass
    runs: int = 0
    pass_started: list = []  # asyncio.Event per pass, set when that pass begins

    async def _run_async_impl(self, ctx):
        tool_call, text = self.passes[min(self.runs, len(self.passes) - 1)]
        if self.runs < len(self.pass_started):
            self.pass_started[self.runs].set()
        self.runs += 1
        if tool_call:
            tool = tool_call if isinstance(tool_call, str) else "org_query"
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(id="call-1", name=tool, args={})
                        )
                    ],
                ),
            )
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                id="call-1", name=tool, response={"total": 12}
                            )
                        )
                    ],
                ),
            )
        if text:
            yield _text_event(ctx, self.name, text)


class _Formatter(BaseAgent):
    """The format gate's formatter: its payload event carries the branch ADK
    gave it, exactly as an LlmAgent's does."""

    payloads: list = []  # one per pass; None = the gate emits nothing; a tuple = several events
    runs: int = 0
    started: asyncio.Event | None = None  # set on entry
    done: asyncio.Event | None = None  # set once its payload is committed
    wait_for: asyncio.Event | None = None
    timeout: float = 2.0
    delay: float = 0.0  # rendering time, before the payload is made

    async def _run_async_impl(self, ctx):
        payload = self.payloads[min(self.runs, len(self.payloads) - 1)]
        self.runs += 1
        if self.started is not None:
            self.started.set()
        if self.wait_for is not None:
            await asyncio.wait_for(self.wait_for.wait(), timeout=self.timeout)
        if self.delay:
            await asyncio.sleep(self.delay)
        for emitted in payload if isinstance(payload, tuple) else (payload,):
            if emitted is None:
                continue
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=json.dumps(emitted))]
                ),
                actions=EventActions(state_delta={"answer_payload": emitted}),
            )
        # Resumed only after the runner consumed (and appended) the event.
        if self.done is not None:
            self.done.set()


class _Gate(BaseAgent):
    """format_gate's shape: runs its formatter, relays what it emits."""

    async def _run_async_impl(self, ctx):
        async for event in self.sub_agents[0].run_async(ctx):
            yield event


class _Critic(BaseAgent):
    """The critic LLM: a streamed partial, then the verdict as text + state,
    the way an LlmAgent with output_key emits it. Records what its branch saw."""

    verdicts: list = []  # one per call
    ran: list = []
    saw: list = []  # the rendered request contents, per call
    session_authors: list = []  # who was in the session when it looked
    raises: Exception | None = None
    started: asyncio.Event | None = None
    wait_for: asyncio.Event | None = None
    timeout: float = 2.0

    async def _run_async_impl(self, ctx):
        self.ran.append(ctx.invocation_id)
        if self.started is not None:
            self.started.set()
        if self.wait_for is not None:
            await asyncio.wait_for(self.wait_for.wait(), timeout=self.timeout)
        self.session_authors.append([e.author for e in ctx.session.events])
        contents = _get_contents(ctx.branch, ctx.session.events, self.name)
        self.saw.append(
            " ".join(
                str(part.text or part.function_call or part.function_response or "")
                for content in contents
                for part in content.parts or []
            )
        )
        if self.raises is not None:
            raise self.raises
        verdict = self.verdicts[min(len(self.ran) - 1, len(self.verdicts) - 1)]
        yield _text_event(ctx, self.name, '{"info_suff', partial=True)
        event = _text_event(ctx, self.name, json.dumps(verdict))
        event.actions = EventActions(state_delta={"critic_verdict": verdict})
        yield event


def _pipeline(
    *,
    parallel: bool,
    executor_passes: list,
    payloads: list,
    verdicts: list | None = None,
    critic_raises: Exception | None = None,
    critic: BaseAgent | None = None,
    gate_callback=None,
) -> tuple[LoopAgent, _Executor, _Formatter, _Critic]:
    executor = _Executor(name=AGENT_NAME, passes=executor_passes, runs=0)
    formatter = _Formatter(name="formatter", payloads=payloads, runs=0)
    gate = _Gate(name="format_gate", sub_agents=[formatter], after_agent_callback=gate_callback)
    if critic is None:
        critic = _Critic(
            name="critic",
            verdicts=verdicts or [SUFFICIENT],
            ran=[],
            saw=[],
            session_authors=[],
            raises=critic_raises,
        )
    escalator = EscalateIfSufficient(name="loop_escalator")
    loop = build_deep_agent(executor, gate, critic, escalator, parallel=parallel)
    return loop, executor, formatter, critic


async def _run(
    loop: LoopAgent,
    *,
    question: str = "how is chevy?",
    state=None,
    prior=None,
    run_config: RunConfig | None = None,
) -> tuple[list[Event], dict, list[Event]]:
    """One turn through a real runner: (streamed events, final state, session events).
    `run_config` is the caller's: the Chat bots run SSE, so a model call streams."""
    runner = InMemoryRunner(agent=loop, app_name="gub")
    session = await runner.session_service.create_session(
        app_name="gub", user_id="u", state=state or {}
    )
    for event in prior or []:
        await runner.session_service.append_event(session, event)
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
    options = {} if run_config is None else {"run_config": run_config}
    streamed = [
        event
        async for event in runner.run_async(
            user_id="u", session_id=session.id, new_message=message, **options
        )
    ]
    final = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session.id
    )
    return streamed, dict(final.state), final.events


def _complete(events: list[Event]) -> list[Event]:
    return [event for event in events if not event.partial]


def _authors(events: list[Event]) -> list[str]:
    return [event.author for event in _complete(events)]


# ── the tree ─────────────────────────────────────────────────────────────────


def test_the_default_tree_runs_the_critic_beside_the_format_gate():
    from gub_agent.agent import deep_agent, root_agent
    from gub_agent.agents.critic import critic_agent
    from gub_agent.agents.format_gate import format_gate

    if not config.CRITIC_PARALLEL:
        pytest.skip("CRITIC_PARALLEL=0 in this environment — the serial tree is pinned below")
    names = [agent.name for agent in deep_agent.sub_agents]
    assert names == [AGENT_NAME, "format_and_critic", "critic_gate", "loop_escalator"]
    fork = deep_agent.sub_agents[1]
    assert isinstance(fork, ParallelAgent)
    assert fork.sub_agents[0] is format_gate
    assert isinstance(fork.sub_agents[1], SpeculativeCritic)
    assert fork.sub_agents[1].sub_agents == [critic_agent]
    resolver = deep_agent.sub_agents[2]
    assert isinstance(resolver, CriticResolver)
    assert resolver.payload_authors == ("format_gate", "formatter")
    assert deep_agent.max_iterations == 2
    # The critic is IN the tree: the runner resolves every event author with
    # find_sub_agent, and an author it cannot find is a warning per event per
    # turn (agents/_agent_router.py).
    assert root_agent.find_agent("critic") is critic_agent


_SERIAL_TREE = [
    "LoopAgent",
    "gub_pipeline",
    [
        ["LlmAgent", AGENT_NAME, []],
        ["FormatGate", "format_gate", [["LlmAgent", "formatter", []]]],
        ["CriticGate", "critic_gate", [["LlmAgent", "critic", []]]],
        ["EscalateIfSufficient", "loop_escalator", []],
    ],
]

_DUMP = """
import json
from gub_agent import agent
from gub_agent.agents import critic
from gub_agent.agents.format_gate import format_gate

def tree(a):
    return [type(a).__name__, a.name, [tree(s) for s in a.sub_agents]]

deep = agent.deep_agent
print(json.dumps({
    "root": [a.name for a in agent.root_agent.sub_agents],
    "deep": tree(deep),
    "max_iterations": deep.max_iterations,
    "same_objects": [
        deep.sub_agents[0] is agent.executor_agent,
        deep.sub_agents[1] is format_gate,
        deep.sub_agents[2].sub_agents[0] is critic.critic_agent,
        deep.sub_agents[3] is critic.escalator_agent,
    ],
}))
"""


def test_critic_parallel_0_builds_the_serial_tree_it_replaced():
    """The rollback. Read at import, so it is checked in a fresh interpreter:
    the tree must be today's serial one, built from the same module objects."""
    # SPECULATIVE_DEEP=0 too, so the root is today's as well; the speculative
    # root is pinned in tests/unit/test_speculation.py.
    env = {**os.environ, "CRITIC_PARALLEL": "0", "SPECULATIVE_DEEP": "0", "PYTHONPATH": str(REPO)}
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _DUMP],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    dumped = json.loads(out.stdout.strip().splitlines()[-1])
    assert dumped["root"] == ["sandbox_echo", "router", "dispatcher"]
    assert dumped["deep"] == _SERIAL_TREE
    assert dumped["max_iterations"] == 2
    assert dumped["same_objects"] == [True, True, True, True]


def test_the_builder_makes_both_shapes():
    parallel, *_ = _pipeline(parallel=True, executor_passes=[(True, "x")], payloads=[ANSWER])
    assert [type(a).__name__ for a in parallel.sub_agents] == [
        "_Executor",
        "ParallelAgent",
        "CriticResolver",
        "EscalateIfSufficient",
    ]
    assert [a.name for a in parallel.sub_agents[1].sub_agents] == [
        "format_gate",
        "critic_speculation",
    ]

    serial, *_ = _pipeline(parallel=False, executor_passes=[(True, "x")], payloads=[ANSWER])
    assert [type(a).__name__ for a in serial.sub_agents] == [
        "_Executor",
        "_Gate",
        "CriticGate",
        "EscalateIfSufficient",
    ]
    assert [a.name for a in serial.sub_agents] == [
        AGENT_NAME,
        "format_gate",
        "critic_gate",
        "loop_escalator",
    ]
    gate = serial.sub_agents[2]
    assert type(gate) is CriticGate  # the gate itself, not the resolver
    assert [a.name for a in gate.sub_agents] == ["critic"]


# ── concurrency and branch isolation ─────────────────────────────────────────


async def test_the_critic_runs_while_the_formatter_does():
    """Each waits for the other to have STARTED. Run one after the other, the
    first would time out waiting; overlapped, both finish."""
    loop, _, formatter, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )
    formatter.started, critic.started = asyncio.Event(), asyncio.Event()
    formatter.wait_for, critic.wait_for = critic.started, formatter.started

    streamed, state, _ = await _run(loop)

    assert critic.ran and formatter.runs == 1
    assert state["critic_verdict"]["sufficient"] is True
    assert streamed[-1].author == "loop_escalator" and streamed[-1].actions.escalate


async def test_the_serial_tree_does_not_overlap_them():
    """Control for the test above: the same barrier deadlocks the serial tree,
    so the overlap is the wiring's doing and not the stand-ins'."""
    loop, _, formatter, critic = _pipeline(
        parallel=False, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )
    formatter.started, critic.started = asyncio.Event(), asyncio.Event()
    formatter.wait_for, critic.wait_for = critic.started, formatter.started
    formatter.timeout = critic.timeout = 0.3

    with pytest.raises(TimeoutError):
        await _run(loop)


async def test_the_critics_branch_sees_the_executor_but_not_the_formatter():
    """ParallelAgent isolates history per branch and shares state. The critic
    must still see the executor's call, result and text (the evidence it
    judges); the formatter's JSON, on the sibling branch, it does not see —
    and never needed (prompts/critic.py calls it pipeline plumbing)."""
    loop, _, formatter, critic = _pipeline(
        parallel=True, executor_passes=[(True, "There are 12 live campaigns.")], payloads=[ANSWER]
    )
    # The critic looks only after the formatter's payload has been committed.
    formatter.done = critic.wait_for = asyncio.Event()

    await _run(loop)

    [seen] = critic.saw
    assert "There are 12 live campaigns." in seen
    assert "org_query" in seen  # the call and its result
    assert "chevy is live" not in seen  # the formatter's payload…
    assert "formatter" in critic.session_authors[0]  # …which WAS in the session


# ── loop control ─────────────────────────────────────────────────────────────


async def test_a_sufficient_verdict_ends_the_loop_after_one_pass():
    loop, executor, _, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    streamed, state, _ = await _run(loop)

    assert executor.runs == 1 and len(critic.ran) == 1
    assert state["critic_verdict"] == SUFFICIENT
    assert [e.author for e in streamed if e.actions.escalate] == ["loop_escalator"]


async def test_an_insufficient_verdict_buys_a_second_pass_that_reads_the_feedback():
    loop, executor, formatter, critic = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns."), (True, "Q3 push is live.")],
        payloads=[ANSWER, ANSWER],
        verdicts=[INSUFFICIENT, SUFFICIENT],
    )

    streamed, state, session_events = await _run(loop)

    assert executor.runs == 2 and formatter.runs == 2
    # The retry reads the feedback from "[critic] said: …" — nothing injects
    # state into its prompt — so the held verdict must be in its history.
    verdict_at = next(i for i, e in enumerate(session_events) if e.author == "critic")
    assert session_events[verdict_at + 1].author == AGENT_NAME  # pass 2 starts
    contents = _get_contents(None, session_events[: verdict_at + 1], AGENT_NAME)
    rendered = " ".join(p.text or "" for c in contents for p in c.parts or [])
    assert INSUFFICIENT["feedback"] in rendered
    assert streamed[-1].author == "loop_escalator"
    assert state["critic_verdict"]["sufficient"] is True


# ── the decisions after the join ─────────────────────────────────────────────


async def test_an_abstain_after_a_tool_call_passes_and_the_held_verdict_is_dropped(
    caplog, monkeypatch
):
    """With CONFIRM_ABSTENTION off (its rollback): the speculative critic said
    "insufficient"; CriticGate's rule for an abstain payload after a tool call
    is a pass without the LLM. The dropped verdict must reach neither the
    stream (the bot would count it and arm a pass restart) nor the state (the
    escalator would read it as a retry)."""
    monkeypatch.setattr(config, "CONFIRM_ABSTENTION", False)
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(True, "GUB has no account by that name.")],
        payloads=[ABSTAIN],
        verdicts=[INSUFFICIENT],
    )

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        streamed, state, session_events = await _run(loop)

    assert len(critic.ran) == 1  # it ran — speculatively
    assert "critic" not in [e.author for e in streamed]
    assert "critic" not in [e.author for e in session_events]
    assert state["critic_verdict"]["reason"] == (
        "Deterministic pass: abstain AnswerPayload (no critic LLM run)."
    )
    assert executor.runs == 1
    assert "critic_speculation: verdict discarded" in caplog.text
    assert "tenant=anomaly" in caplog.text


CONFIRM_PREFIX = "Deterministic: an abstention after a search is confirmed by a second route"


def _verdicts_by_critic(events: list[Event]) -> list[dict]:
    return [
        json.loads(e.content.parts[0].text)
        for e in _complete(events)
        if e.author == "critic" and e.content and e.content.parts
    ]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_a_searched_abstain_is_sent_back_once_to_confirm(parallel):
    """#102: after a search, "none" stands only when a second route agrees.
    Pass 1 abstains and is sent back in code, authored as the critic; pass 2
    abstains again, and that confirmed abstention passes."""
    loop, executor, _, critic = _pipeline(
        parallel=parallel,
        executor_passes=[(True, "GUB has no account by that name."), (True, "Still none.")],
        payloads=[ABSTAIN, ABSTAIN],
        verdicts=[SUFFICIENT],
    )

    streamed, state, _ = await _run(loop, question="is ford a client?")

    assert executor.runs == 2
    sent_back = _verdicts_by_critic(streamed)[0]
    assert sent_back["sufficient"] is False
    assert sent_back["reason"].startswith(CONFIRM_PREFIX)
    assert "different route" in sent_back["feedback"]
    assert state["critic_verdict"]["sufficient"] is True
    assert streamed[-1].author == "loop_escalator"


async def test_a_confirming_pass_that_finds_something_answers():
    """The second route found the record: pass 2 answers, settled as the
    loop's last pass."""
    loop, executor, _, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "No account starts with x."), (True, "Xiaomi.")],
        payloads=[ABSTAIN, ANSWER],
        verdicts=[SUFFICIENT],
    )

    _, state, _ = await _run(loop, question="which accounts start with x?")

    assert executor.runs == 2
    assert state["answer_payload"] == ANSWER
    assert state["critic_verdict"]["reason"] == FINAL_PASS_REASON


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_a_find_files_abstain_stands_at_once(parallel):
    """Only Drive file names are indexed, so an empty find_files has no second
    route: its abstention passes on the first pass, as before."""
    loop, executor, _, _ = _pipeline(
        parallel=parallel,
        executor_passes=[("find_files", "No file by that name.")],
        payloads=[ABSTAIN],
    )

    _, state, _ = await _run(loop, question="do we have the BHAC teaser?")

    assert executor.runs == 1
    assert state["critic_verdict"]["reason"] == (
        "Deterministic pass: abstain AnswerPayload (no critic LLM run)."
    )


async def test_the_marker_after_a_search_is_confirmed_too():
    """The bare NO_COMPANY_RECORDS marker after a tool call is a "none found",
    not the executor's "nothing to look up": it is confirmed like a payload."""
    loop, executor, _, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "NO_COMPANY_RECORDS"), (True, "NO_COMPANY_RECORDS")],
        payloads=[ABSTAIN, ABSTAIN],
    )

    streamed, state, _ = await _run(loop, question="is ford a client?")

    assert executor.runs == 2
    assert _verdicts_by_critic(streamed)[0]["reason"].startswith(CONFIRM_PREFIX)
    assert state["critic_verdict"]["sufficient"] is True


async def test_a_from_memory_abstain_is_sent_back_after_the_join():
    """No tool call and an abstain payload: the draft was written from memory
    and is sent back once, in code, authored as the critic."""
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN, ANSWER],
        verdicts=[SUFFICIENT],
    )

    streamed, state, _ = await _run(loop, question="whats new")

    assert executor.runs == 2
    send_back = [e for e in _complete(streamed) if e.author == "critic"][0]
    verdict = json.loads(send_back.content.parts[0].text)
    assert verdict["sufficient"] is False and "NO tool call" in verdict["feedback"]
    assert streamed[-1].author == "loop_escalator"


def _an_earlier_abstain() -> dict:
    """An earlier turn that ended in an abstain payload: its gate event, and
    the state it left behind."""
    earlier = Event(
        invocation_id="inv-earlier",
        author="format_gate",
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=json.dumps(ABSTAIN))]
        ),
        actions=EventActions(state_delta={"answer_payload": ABSTAIN}),
    )
    return {"state": {"answer_payload": ABSTAIN}, "prior": [earlier]}


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_a_stale_payload_left_in_state_by_an_earlier_turn_is_not_read(parallel):
    """The gate emits nothing when the executor produced no text — on purpose.
    State then still holds the previous turn's abstain payload, and a gate that
    read state would settle THIS turn on it (here: a from-memory re-query —
    two executor passes and a `sufficient: false` the bot restarts its bubble
    on). Both gates read this pass's payload, find none, and ask the critic."""
    loop, executor, _, critic = _pipeline(
        parallel=parallel,
        executor_passes=[(False, "Thanks, noted.")],
        payloads=[None],
        verdicts=[SUFFICIENT],
    )

    streamed, state, _ = await _run(loop, **_an_earlier_abstain())

    assert state["answer_payload"] == ABSTAIN  # still there, still stale
    assert len(critic.ran) == 1
    assert state["critic_verdict"] == SUFFICIENT  # the critic's, not a re-query
    assert executor.runs == 1


async def test_the_first_passes_payload_is_not_the_seconds():
    """Same edge inside one turn: pass 1 abstained from memory and was sent
    back; pass 2's executor called a tool but wrote nothing, so its gate
    emitted nothing. Pass 1's abstain payload is this invocation's, and still
    not this PASS's — pass 2 is settled as the loop's last pass, not as a
    second abstain pass."""
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(False, "Copied from memory."), (True, "")],
        payloads=[ABSTAIN, None],
        verdicts=[SUFFICIENT],
    )

    _, state, _ = await _run(loop, question="whats new")

    assert executor.runs == 2
    assert state["critic_verdict"]["reason"] == FINAL_PASS_REASON
    assert len(critic.ran) == 1  # pass 1's speculation, dropped; none on pass 2


async def test_the_bare_marker_makes_no_critic_call():
    loop, _, _, critic = _pipeline(
        parallel=True, executor_passes=[(False, "NO_COMPANY_RECORDS")], payloads=[ABSTAIN]
    )

    streamed, state, _ = await _run(loop)

    assert critic.ran == []  # known before the fork: no speculative call
    assert state["critic_verdict"]["sufficient"] is True
    assert "critic_gate" in _authors(streamed)


async def test_a_disabled_critic_makes_no_critic_call(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    loop, _, _, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    _, state, _ = await _run(loop, state={"sandbox": {"critic_enabled": False}})

    assert critic.ran == []
    assert state["critic_verdict"]["reason"] == "sandbox: critic disabled"


async def test_a_held_critic_error_fails_only_a_turn_that_needs_the_verdict():
    boom = RuntimeError("critic model failed")

    settled, *_ = _pipeline(
        parallel=True,
        executor_passes=[(True, "GUB has no account by that name.")],
        payloads=[ABSTAIN],
        critic_raises=boom,
    )
    _, state, _ = await _run(settled)  # the abstain pass never needed it
    assert state["critic_verdict"]["sufficient"] is True

    needed, *_ = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns.")],
        payloads=[ANSWER],
        critic_raises=boom,
    )
    with pytest.raises(RuntimeError, match="critic model failed"):
        await _run(needed)


async def test_a_pass_with_nothing_held_falls_back_to_the_serial_order(monkeypatch, caplog):
    """The branch holds a result whenever it had a call to make. If the
    resolver finds none anyway (an evicted entry, a resumed invocation), it
    runs the critic itself, after the payload — never a verdict nobody made."""
    from gub_agent.agents import critic as critic_module

    monkeypatch.setattr(critic_module, "_hold", lambda invocation_id, run: None)
    loop, _, _, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    with caplog.at_level(logging.WARNING, logger="gub_agent.agents.critic"):
        streamed, state, _ = await _run(loop)

    assert len(critic.ran) == 2  # the dropped speculation, then the fallback
    assert state["critic_verdict"] == SUFFICIENT
    assert _authors(streamed)[-2:] == ["critic", "loop_escalator"]
    assert "critic_speculation: nothing held" in caplog.text


# ── the loop's last pass ─────────────────────────────────────────────────────


ANSWER_2 = {"kind": "answer", "headline": "the Q3 push is live", "citations": ["org_query:a2"]}


class _CriticModel(BaseLlm):
    """The critic's MODEL, not a stand-in agent: the real LlmAgent (output
    schema, output_key) runs over it, and every generate call is counted."""

    verdicts: list = []  # one per call
    calls: int = 0

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        yield LlmResponse(
            content=genai_types.Content(
                role="model", parts=[genai_types.Part.from_text(text=json.dumps(verdict))]
            )
        )


def _llm_critic(verdicts: list) -> tuple[LlmAgent, _CriticModel]:
    model = _CriticModel(model="stub-critic", verdicts=verdicts, calls=0)
    critic = LlmAgent(
        name="critic",
        model=model,
        instruction="Judge the executor's tool coverage.",
        output_schema=CriticVerdict,
        output_key="critic_verdict",
    )
    return critic, model


def _critic_verdicts(events: list[Event]) -> list[dict]:
    """What the bot reads per judged pass: every complete `critic` event's text."""
    return [
        json.loads(event.content.parts[0].text)
        for event in _complete(events)
        if event.author == "critic"
    ]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_the_last_pass_makes_no_critic_model_call(parallel):
    """Iteration 2 of max_iterations=2: the critic's verdict could not buy a
    third pass, so its model is never called there. The loop still ends on the
    escalator, the answer is pass 2's, and the verdict the bot reads last is a
    sufficient one authored `critic` — pass 1 judged exactly as before."""
    critic, model = _llm_critic([INSUFFICIENT, SUFFICIENT])
    loop, executor, formatter, _ = _pipeline(
        parallel=parallel,
        executor_passes=[(True, "12 live campaigns."), (True, "The Q3 push is live.")],
        payloads=[ANSWER, ANSWER_2],
        critic=critic,
    )

    streamed, state, _ = await _run(loop)

    assert model.calls == 1  # pass 1 only
    assert executor.runs == 2 and formatter.runs == 2
    assert _authors(streamed) == [
        *[AGENT_NAME] * 3,
        "formatter",
        "critic",  # pass 1: the LLM's verdict
        *[AGENT_NAME] * 3,
        "formatter",
        "critic",  # pass 2: settled in code
        "loop_escalator",
    ]
    first, last = _critic_verdicts(streamed)
    assert first == INSUFFICIENT
    assert last["sufficient"] is True and last["reason"] == FINAL_PASS_REASON
    assert state["critic_verdict"] == last
    assert streamed[-1].actions.escalate
    payload_events = [e for e in _complete(streamed) if e.author in ("formatter", "format_gate")]
    assert json.loads(payload_events[-1].content.parts[0].text) == ANSWER_2


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_critic_skip_final_pass_0_asks_the_model_on_the_last_pass(parallel, monkeypatch):
    """The skip's own rollback, in both wirings: the model judges pass 2 again
    and the verdict the bot reads last is the model's, not one written in code.
    CRITIC_PARALLEL alone could not bring this back — it only picks the tree."""
    monkeypatch.setattr(config, "CRITIC_SKIP_FINAL_PASS", False)
    critic, model = _llm_critic([INSUFFICIENT, SUFFICIENT])
    loop, executor, _, _ = _pipeline(
        parallel=parallel,
        executor_passes=[(True, "12 live campaigns."), (True, "The Q3 push is live.")],
        payloads=[ANSWER, ANSWER_2],
        critic=critic,
    )

    streamed, state, _ = await _run(loop)

    assert model.calls == 2 and executor.runs == 2
    assert _critic_verdicts(streamed) == [INSUFFICIENT, SUFFICIENT]
    assert state["critic_verdict"] == SUFFICIENT
    assert streamed[-1].author == "loop_escalator" and streamed[-1].actions.escalate


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_the_first_pass_still_asks_the_critic_model(parallel):
    critic, model = _llm_critic([SUFFICIENT])
    loop, executor, _, _ = _pipeline(
        parallel=parallel,
        executor_passes=[(True, "12 live campaigns.")],
        payloads=[ANSWER],
        critic=critic,
    )

    streamed, state, _ = await _run(loop)

    assert model.calls == 1 and executor.runs == 1
    assert _critic_verdicts(streamed) == [SUFFICIENT]
    assert state["critic_verdict"] == SUFFICIENT
    assert streamed[-1].author == "loop_escalator" and streamed[-1].actions.escalate


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_a_from_memory_retry_is_not_judged_by_the_model_either(parallel):
    """Pass 1 is sent back in code (the from-memory re-query), pass 2 is the
    last: neither pass is judged by the model. The parallel wiring still STARTS
    pass 1's speculative call — the abstain payload is only known once the gate
    ends — and drops its verdict; pass 2 makes none."""
    critic, model = _llm_critic([INSUFFICIENT])
    loop, executor, _, _ = _pipeline(
        parallel=parallel,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN, ANSWER],
        critic=critic,
    )

    streamed, state, _ = await _run(loop, question="whats new")

    # The speculative call is cancelled when the gate ends on the abstain payload
    # (`settle_speculation`), and here the formatter stand-in answers at once, so
    # whether the stub model was reached before the cancel is a matter of
    # scheduling: the bound is what is pinned, not the count. Serial never calls.
    assert model.calls <= (1 if parallel else 0)
    assert executor.runs == 2
    requery, last = _critic_verdicts(streamed)
    assert requery["sufficient"] is False and "NO tool call" in requery["feedback"]
    assert last["reason"] == FINAL_PASS_REASON
    assert streamed[-1].author == "loop_escalator"


async def test_a_retry_that_abstains_keeps_the_abstain_reason():
    """The abstain pass is decided BEFORE the last-pass check, so a retry that
    abstains after a tool call reads as it always did."""
    loop, _, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns."), (True, "GUB has no such campaign.")],
        payloads=[ANSWER, ABSTAIN],
        verdicts=[INSUFFICIENT],
    )

    _, state, _ = await _run(loop)

    assert len(critic.ran) == 1
    assert state["critic_verdict"]["reason"] == (
        "Deterministic pass: abstain AnswerPayload (no critic LLM run)."
    )


def test_the_deployed_tree_finds_its_loop_bound():
    """`_is_final_pass` walks up to the LoopAgent from wherever the gate sits
    in the tree this process built (serial or parallel), and a gate outside a
    loop never sees a last pass."""
    from types import SimpleNamespace

    from gub_agent.agent import deep_agent
    from gub_agent.agents.critic import _is_final_pass

    def ctx(verdicts: int):
        judged = Event(
            invocation_id="inv",
            author="critic",
            actions=EventActions(state_delta={"critic_verdict": SUFFICIENT}),
        )
        other = Event(invocation_id="inv-earlier", author="critic", actions=judged.actions)
        return SimpleNamespace(
            invocation_id="inv", session=SimpleNamespace(events=[other] + [judged] * verdicts)
        )

    gates = [a for a in deep_agent.sub_agents if a.name == "critic_gate"]
    speculation = deep_agent.find_agent("critic_speculation")
    for agent in [*gates, *([speculation] if speculation else [])]:
        assert _is_final_pass(agent, ctx(0)) is False
        assert _is_final_pass(agent, ctx(1)) is True
    assert _is_final_pass(CriticGate(name="loose"), ctx(1)) is False


# ── the stream the bot reads ─────────────────────────────────────────────────


async def test_the_stream_keeps_its_order_and_its_authors():
    """The bot takes the LAST formatter/format_gate payload and ignores the
    critic's; it restarts its pass on a critic `sufficient: false`. Held and
    released after the join, the critic's events land where they always did:
    after the payload, before the escalator — partial included."""
    loop, *_ = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    streamed, _, _ = await _run(loop)

    authors = [e.author for e in streamed]
    assert authors == [
        AGENT_NAME,  # call
        AGENT_NAME,  # result
        AGENT_NAME,  # answer
        "formatter",  # the payload
        "critic",  # its streamed partial
        "critic",  # the verdict
        "loop_escalator",
    ]
    payload_events = [e for e in streamed if e.author in ("formatter", "format_gate")]
    assert json.loads(payload_events[-1].content.parts[0].text) == ANSWER
    assert streamed[4].partial and not streamed[5].partial


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_a_reloaded_turn_reads_in_the_streamed_order(parallel, tmp_path):
    """The formatter is the slower branch here, so the critic MADE its events
    before the payload they follow in the stream. A session store orders a turn
    by event timestamp (ADK's sqlite service here; the Agent Engine store is
    handed the same field on append), so a relayed event that kept its creation
    time reloads ahead of the payload: [critic, formatter] where the stream had
    [formatter, critic]."""
    loop, _, formatter, _ = _pipeline(
        parallel=parallel, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )
    formatter.delay = 0.05
    service = SqliteSessionService(db_path=str(tmp_path / "sessions.db"))
    runner = Runner(agent=loop, app_name="gub", session_service=service)
    session = await service.create_session(app_name="gub", user_id="u")
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])

    streamed = [
        event
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message)
    ]
    stored = await service.get_session(app_name="gub", user_id="u", session_id=session.id)

    stamps = [event.timestamp for event in _complete(streamed)]
    assert stamps == sorted(stamps)
    assert [event.author for event in stored.events][1:] == _authors(streamed)
    assert _authors(streamed)[-3:] == ["formatter", "critic", "loop_escalator"]


@pytest.mark.parametrize(
    "executor_passes,payloads,verdicts,earlier_abstain",
    [
        pytest.param([(True, "12 live.")], [ANSWER], [SUFFICIENT], False, id="answer-sufficient"),
        pytest.param(
            [(True, "12 live."), (True, "Q3 live.")],
            [ANSWER, ANSWER],
            [INSUFFICIENT, SUFFICIENT],
            False,
            id="retry",
        ),
        pytest.param([(True, "No such account.")], [ABSTAIN], [INSUFFICIENT], False, id="abstain"),
        pytest.param(
            [(False, "Copied."), (True, "12 live.")],
            [ABSTAIN, ANSWER],
            [SUFFICIENT],
            False,
            id="from-memory",
        ),
        pytest.param([(False, "NO_COMPANY_RECORDS")], [ABSTAIN], [SUFFICIENT], False, id="marker"),
        # The gate emits nothing and state still holds the previous turn's
        # abstain payload: neither wiring may read it as this pass's.
        pytest.param([(False, "Thanks, noted.")], [None], [SUFFICIENT], True, id="stale-state"),
    ],
)
async def test_both_wirings_reach_the_same_decisions(
    executor_passes, payloads, verdicts, earlier_abstain
):
    """Same turn, both trees: same passes, same verdict in state, and the same
    complete events in the same order. What differs is only WHEN the critic ran
    and, on an abstain, that the parallel one ran it for nothing."""
    results = []
    for parallel in (False, True):
        loop, executor, _, _ = _pipeline(
            parallel=parallel,
            executor_passes=executor_passes,
            payloads=payloads,
            verdicts=verdicts,
        )
        streamed, state, _ = await _run(loop, **(_an_earlier_abstain() if earlier_abstain else {}))
        results.append(
            (
                executor.runs,
                state["critic_verdict"],
                _authors(streamed),
                [json.dumps(e.actions.state_delta, sort_keys=True) for e in _complete(streamed)],
            )
        )
    assert results[0] == results[1]


# ── a critic call whose verdict the pass has already settled ──────────────────
#
# The ParallelAgent joins only when BOTH branches end. Before the gate's
# after-callback (critic.py:settle_speculation) an abstain payload — which the
# resolver settles in code and whose critic verdict it drops — still waited out
# the critic's model call (tens of seconds on a measured turn). The critic here is
# STALLED on an event that only a later step of the turn sets (or nobody does),
# so under the old wiring these turns deadlock, bounded by `_bounded`'s timeout
# and never by a sleep. No test below synchronises on time.

CLARIFY = {"kind": "clarify", "headline": "Which campaign do you mean?"}
ABSTAIN_PASS = "Deterministic pass: abstain AnswerPayload (no critic LLM run)."


class _StalledCritic(_Critic):
    """A critic that waits (up to `timeout`, far past any test's own bound) and
    counts the cancellations it was handed — the proof that a call was
    abandoned and not merely outlasted."""

    cancelled: int = 0
    timeout: float = 30.0
    stall_once: bool = False  # only the first call waits; a re-run answers at once

    async def _run_async_impl(self, ctx):
        try:
            async for event in super()._run_async_impl(ctx):
                yield event
        except asyncio.CancelledError:
            self.cancelled += 1
            if self.stall_once:
                self.wait_for = None
            raise


def _stalled_pipeline(
    *,
    executor_passes: list,
    payloads: list,
    release: asyncio.Event | None = None,
    parallel: bool = True,
    verdicts: list | None = None,
    gate_callback=None,
):
    """A pipeline whose critic waits on `release` (never set, by default).

    The formatter ends only after the critic call has STARTED: that is when the
    call is a running task the gate's callback can reach. (In production the
    formatter takes seconds; here it would otherwise answer before the sibling
    branch had been scheduled, and the turn would fall back to the old wait.)"""
    critic = _StalledCritic(
        name="critic",
        verdicts=verdicts or [INSUFFICIENT],
        ran=[],
        saw=[],
        session_authors=[],
        started=asyncio.Event(),
        wait_for=release if release is not None else asyncio.Event(),
    )
    loop, executor, formatter, _ = _pipeline(
        parallel=parallel,
        executor_passes=executor_passes,
        payloads=payloads,
        critic=critic,
        gate_callback=gate_callback,
    )
    if parallel:
        formatter.wait_for = critic.started
    return loop, executor, formatter, critic


async def _bounded(loop: LoopAgent, **kwargs):
    """`_run`, with a fail-safe: a turn that waits on the stalled critic fails
    here, as an assertion, instead of hanging the suite."""
    try:
        return await asyncio.wait_for(_run(loop, **kwargs), timeout=3.0)
    except TimeoutError:
        pytest.fail("the turn did not finish: it is waiting for a critic call it does not need")


def _nothing_left_running() -> None:
    """No critic call in the registry, no task of the turn still alive. (The
    registry is read with a default so a turn that never had one — the old
    wiring — is judged by its tasks alone.)"""
    assert getattr(critic_module, "_RUNNING", {}) == {}
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []


async def test_a_stalled_critic_does_not_delay_the_pass_it_was_dropped_from(caplog):
    """The case this exists for: pass 1 is a from-memory draft, the gate abstains, the
    resolver sends it back for a re-query in code — and pass 2 waited for a
    critic call whose verdict was thrown away. The critic here waits for pass 2
    to START, which under the old wiring only comes after the join: a deadlock.
    Now the call is cancelled the moment the gate ends on the abstain payload,
    and pass 2 runs."""
    release = asyncio.Event()
    loop, executor, _, critic = _stalled_pipeline(
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN, ANSWER],
        release=release,
    )
    executor.pass_started = [asyncio.Event(), release]

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        streamed, state, _ = await _bounded(loop, question="whats new")

    assert executor.runs == 2 and release.is_set()  # pass 2 ran
    assert critic.cancelled == 1  # cancelled — not timed out, not answered
    assert critic.saw == []  # it never got as far as a verdict
    requery, last = _critic_verdicts(streamed)
    assert requery["sufficient"] is False and "NO tool call" in requery["feedback"]
    assert last["reason"] == FINAL_PASS_REASON
    assert streamed[-1].author == "loop_escalator"
    assert "call abandoned" in caplog.text
    _nothing_left_running()


async def test_a_stalled_critic_does_not_delay_an_abstain_after_a_tool_call(monkeypatch):
    """The deterministic pass: the executor looked, the gate abstained, the
    resolver passes in code. The critic would never have been read; the turn
    ends without waiting for it (nobody ever releases it here)."""
    # One pass, as before #102: the searched abstain is not sent back to confirm.
    monkeypatch.setattr(config, "CONFIRM_ABSTENTION", False)
    loop, executor, _, critic = _stalled_pipeline(
        executor_passes=[(True, "No account by that name.")], payloads=[ABSTAIN]
    )

    streamed, state, session_events = await _bounded(loop)

    assert executor.runs == 1 and critic.cancelled == 1 and critic.saw == []
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS
    assert "critic" not in [e.author for e in streamed]
    assert "critic" not in [e.author for e in session_events]
    assert streamed[-1].author == "loop_escalator" and streamed[-1].actions.escalate
    _nothing_left_running()


@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param([ANSWER], id="answer"),
        pytest.param([CLARIFY], id="clarify"),
        pytest.param([None], id="no-payload"),
        # One pass, two payload events: the gate abstained, retried and ended on an
        # answer. The callback reads the pass's LAST payload, not its first.
        pytest.param([(ABSTAIN, ANSWER)], id="abstain-then-answer"),
    ],
)
async def test_a_verdict_that_matters_is_still_awaited(payloads, caplog):
    """The critic is released only once the formatter is done. A payload the
    resolver does not settle in code leaves the call running, and its verdict
    is the one the loop acts on."""
    critic = _StalledCritic(
        name="critic",
        verdicts=[SUFFICIENT],
        ran=[],
        saw=[],
        session_authors=[],
        started=asyncio.Event(),
    )
    loop, _, formatter, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns.")],
        payloads=payloads,
        critic=critic,
    )
    formatter.done = asyncio.Event()
    critic.wait_for = formatter.done  # the call ends only after the gate has
    formatter.wait_for = critic.started  # and the gate ends only once it runs

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        streamed, state, _ = await _bounded(loop)

    assert critic.cancelled == 0 and len(critic.saw) == 1  # it ran to its verdict
    assert state["critic_verdict"] == SUFFICIENT  # the critic's, not one made in code
    assert _critic_verdicts(streamed) == [SUFFICIENT]
    assert "call abandoned" not in caplog.text
    assert "verdict discarded" not in caplog.text
    _nothing_left_running()


@pytest.mark.parametrize(
    "executor_passes,payloads",
    [
        pytest.param([(False, "Copied."), (True, "12 live.")], [ABSTAIN, ANSWER], id="from-memory"),
        pytest.param([(True, "No such account.")], [ABSTAIN], id="abstain-after-tool-call"),
    ],
)
async def test_the_stalled_critic_changes_neither_the_events_nor_the_state(
    executor_passes, payloads
):
    """The abandon is invisible to everything downstream: for the same turn the
    serial wiring (which never calls the critic on an abstain) and the parallel
    one produce the same complete events, in the same order, with the same
    state deltas and the same final verdict."""
    results = []
    for parallel in (False, True):
        release = asyncio.Event()
        loop, executor, _, _ = _stalled_pipeline(
            executor_passes=executor_passes,
            payloads=payloads,
            release=release,
            parallel=parallel,
        )
        executor.pass_started = [asyncio.Event(), release]
        streamed, state, session_events = await _bounded(loop, question="whats new")
        results.append(
            (
                executor.runs,
                state["critic_verdict"],
                _authors(streamed),
                [json.dumps(e.actions.state_delta, sort_keys=True) for e in _complete(streamed)],
                [e.author for e in session_events],
            )
        )
    assert results[0] == results[1]
    _nothing_left_running()


async def test_an_abandoned_call_logs_one_line_and_the_discard_line_still_logs(caplog, monkeypatch):
    """Telemetry: ONE new info line, keys in the engine's order with tenant last,
    and the existing `verdict discarded` line exactly where it was. A second
    settle (a callback chained AFTER ours calls it again) finds the run already
    abandoned: no second line, no second cancel — and proves the gate's own
    callback is chained, not replaced."""
    # One pass, as before #102: the searched abstain is not sent back to confirm.
    monkeypatch.setattr(config, "CONFIRM_ABSTENTION", False)
    chained: list[str] = []

    def later(callback_context):
        chained.append(callback_context.invocation_id)
        assert critic_module.settle_speculation(callback_context) is None

    loop, _, _, critic = _stalled_pipeline(
        executor_passes=[(True, "No account by that name.")],
        payloads=[ABSTAIN],
        gate_callback=later,
    )

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        await _bounded(loop)

    messages = [r.getMessage() for r in caplog.records if r.name == "gub_agent.agents.critic"]
    [abandoned] = [m for m in messages if "call abandoned" in m]
    match = re.fullmatch(
        r"critic_speculation: call abandoned \(inv=(\S+)\) critic_ms=(\d+) tenant=anomaly",
        abandoned,
    )
    assert match, abandoned
    [discarded] = [m for m in messages if "critic_speculation: verdict discarded" in m]
    assert f"(inv={match.group(1)})" in discarded and discarded.endswith(ABSTAIN_PASS)
    assert messages.index(abandoned) < messages.index(discarded)
    assert not any("format_gate" in m for m in messages)
    assert len(chained) == 1 and critic.cancelled == 1
    _nothing_left_running()


async def test_cancelling_the_turn_leaves_no_critic_call_behind():
    """The outer cancellation (a turn that ended, or a speculative deep run that
    was thrown away): the call is cancelled and awaited with it — no task, no
    registry entry — whichever of the two branches was still running."""
    stuck = asyncio.Event()
    loop, _, formatter, critic = _stalled_pipeline(
        executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )
    formatter.wait_for = stuck  # the gate is still rendering when the cancel comes
    formatter.timeout = 30.0
    formatter.started = asyncio.Event()

    turn = asyncio.create_task(_run(loop))
    await asyncio.wait_for(critic.started.wait(), timeout=3.0)
    await asyncio.wait_for(formatter.started.wait(), timeout=3.0)
    assert list(critic_module._RUNNING) != []  # the call is registered while it runs
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert critic.cancelled == 1
    _nothing_left_running()


async def test_a_real_critic_model_call_is_cancelled_and_logs_its_status(caplog):
    """What the after-deploy readout counts: an abandoned call whose request was
    in flight writes its `model_call:` line as `agent=critic status=error:cancelled`
    (models.py), once, so abandoned count == cancelled critic calls. A call
    cancelled before its request was sent writes no line, so the counts can
    differ by those."""

    class _Hanging(BaseLlm):
        reached: asyncio.Event | None = None

        @classmethod
        def supported_models(cls):
            return [r".*"]

        async def generate_content_async(self, llm_request, stream=False):
            self.reached.set()
            await asyncio.Event().wait()
            yield LlmResponse()  # unreachable

    model = _Hanging(model=GEMINI_MODEL, reached=asyncio.Event())
    critic = LlmAgent(
        name="critic",
        model=VendorRouter(model=GEMINI_MODEL, gemini=model),  # type: ignore[arg-type]
        instruction="Judge the executor's tool coverage.",
        output_schema=CriticVerdict,
        output_key="critic_verdict",
        before_model_callback=lambda callback_context, llm_request: bind_model_call(
            callback_context
        ),
    )
    loop, executor, formatter, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "No account by that name.")],
        payloads=[ABSTAIN],
        critic=critic,
    )
    formatter.wait_for = model.reached  # the request is on the wire when the gate ends

    with caplog.at_level(logging.INFO):
        _, state, _ = await _bounded(loop)

    calls = [r.getMessage() for r in caplog.records if r.getMessage().startswith("model_call: ")]
    [line] = calls
    assert " agent=critic " in line and " status=error:cancelled " in line
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS
    assert caplog.text.count("critic_speculation: call abandoned") == 1
    _nothing_left_running()


# ── the wiring and the callback itself ───────────────────────────────────────


def test_the_parallel_builder_attaches_the_callback_once_and_chains_what_is_there():
    seen: list = []

    def existing(callback_context):
        seen.append(callback_context)

    loop, *_ = _pipeline(
        parallel=True,
        executor_passes=[(True, "x")],
        payloads=[ANSWER],
        gate_callback=existing,
    )
    gate = loop.sub_agents[1].sub_agents[0]
    assert gate.after_agent_callback == [critic_module.settle_speculation, existing]
    assert loop.sub_agents[1].sub_agents[1].payload_authors == ("format_gate", "formatter")

    # The same gate built into a second loop (the builder can run again over
    # the module's shared gate): no second copy. ADK allows an agent one
    # parent, so the first loop lets go of its agents first.
    executor, fork, _, escalator = loop.sub_agents
    critic = fork.sub_agents[1].sub_agents[0]
    for agent in (executor, gate, critic, escalator):
        agent.parent_agent = None
    build_deep_agent(executor, gate, critic, escalator, parallel=True)
    assert gate.after_agent_callback == [critic_module.settle_speculation, existing]


def test_the_serial_builder_attaches_nothing():
    loop, *_ = _pipeline(parallel=False, executor_passes=[(True, "x")], payloads=[ANSWER])
    assert loop.sub_agents[1].after_agent_callback is None


def test_critic_parallel_0_leaves_the_shared_gate_without_a_callback():
    """The rollback is exact: a fresh interpreter with CRITIC_PARALLEL=0 builds
    the module's format gate with no after-callback at all."""
    env = {**os.environ, "CRITIC_PARALLEL": "0", "SPECULATIVE_DEEP": "0", "PYTHONPATH": str(REPO)}
    code = (
        "from gub_agent import agent\n"
        "from gub_agent.agents.format_gate import format_gate\n"
        "print(format_gate.after_agent_callback)"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.strip().splitlines()[-1] == "None"


def test_the_deployed_gate_carries_the_callback_in_the_parallel_wiring():
    from gub_agent.agents.format_gate import format_gate

    if not config.CRITIC_PARALLEL:
        pytest.skip("CRITIC_PARALLEL=0 in this environment — the serial tree is pinned above")
    assert format_gate.after_agent_callback == [critic_module.settle_speculation]


def test_settling_with_nothing_running_is_a_silent_no_op(caplog):
    """The format gate is shared with the fast path, where no critic call is ever
    registered, and the serial wiring never registers one: the callback finds
    nothing and emits nothing."""
    from types import SimpleNamespace

    with caplog.at_level(logging.INFO):
        result = critic_module.settle_speculation(
            SimpleNamespace(invocation_id="inv-fast", session=None)
        )

    assert result is None
    assert caplog.records == []
    assert critic_module._RUNNING == {}


async def test_a_broken_settle_never_fails_the_turn(caplog):
    """The callback is an optimisation: whatever goes wrong inside it is caught
    and logged, with the invocation id, and the call is left running."""
    from types import SimpleNamespace

    run = critic_module._Speculation(
        authors=("format_gate",), task=asyncio.ensure_future(asyncio.sleep(0))
    )
    critic_module._RUNNING["inv-broken"] = run
    try:
        # No `session`: reading this pass's payload raises inside the callback.
        with caplog.at_level(logging.WARNING, logger="gub_agent.agents.critic"):
            result = critic_module.settle_speculation(SimpleNamespace(invocation_id="inv-broken"))
    finally:
        del critic_module._RUNNING["inv-broken"]
        await run.task

    assert result is None
    assert run.abandoned is False
    assert "critic_speculation: settle failed (inv=inv-broken) (AttributeError)" in caplog.text
    assert "format_gate" not in caplog.text


async def test_a_broken_settle_leaves_the_turn_to_wait_for_the_critic_as_before(caplog):
    """Through a real turn: the gate's callback breaks (reading the pass's payload
    raises for the callback's context only), the warning is logged, the call is
    not abandoned, and the turn ends as it did before the callback existed —
    the critic ran to its verdict, which the resolver then drops."""
    from google.adk.agents.invocation_context import InvocationContext

    real = critic_module._this_pass_payload

    def flaky(ctx, authors):
        if isinstance(ctx, InvocationContext):  # the resolver's own read
            return real(ctx, authors)
        release.set()  # the old wiring's wait ends: the critic is let finish
        raise RuntimeError("broken callback")

    release = asyncio.Event()  # the critic stalls until the callback has broken
    loop, _, _, critic = _stalled_pipeline(
        executor_passes=[(True, "No account by that name.")], payloads=[ABSTAIN], release=release
    )

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(critic_module, "_this_pass_payload", flaky)
            _, state, _ = await _bounded(loop)

    assert "critic_speculation: settle failed" in caplog.text
    assert "(RuntimeError) tenant=anomaly" in caplog.text
    assert "call abandoned" not in caplog.text
    assert critic.cancelled == 0 and len(critic.saw) == 1  # it ran to its verdict
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS  # …which was dropped
    assert "verdict discarded" in caplog.text
    _nothing_left_running()


async def test_the_gate_ending_at_once_still_abandons_the_call(caplog):
    """The production ordering, with no barrier: the gate abstains the moment it
    runs, as it can in a turn, and nothing makes it wait for the critic's task to
    exist. ParallelAgent resumes a branch only after its event was consumed, so
    the call is registered by then; if a change registered it later the callback
    would find nothing and this turn would wait for the stalled critic."""
    critic = _StalledCritic(
        name="critic",
        verdicts=[INSUFFICIENT],
        ran=[],
        saw=[],
        session_authors=[],
        started=asyncio.Event(),
        wait_for=asyncio.Event(),  # never set
    )
    loop, _, formatter, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "No account by that name.")],
        payloads=[ABSTAIN],
        critic=critic,
    )
    assert formatter.wait_for is None  # nothing orders the gate after the critic

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        _, state, _ = await _bounded(loop)

    assert "call abandoned" in caplog.text
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS
    _nothing_left_running()


async def test_an_abandoned_call_the_resolver_still_needs_is_run_serially(caplog):
    """If the callback's predicate and the resolver's decision ever disagree (here
    forced: the resolver is made to leave an abstain payload to the critic), the
    abandoned run holds no verdict, so the resolver falls back to today's serial
    order — the critic runs now, after the payload — and ITS verdict is the one in
    the stream and the state. Nothing is leaked and no verdict is invented."""
    critic = _StalledCritic(
        name="critic",
        verdicts=[SUFFICIENT],
        ran=[],
        saw=[],
        session_authors=[],
        started=asyncio.Event(),
        wait_for=asyncio.Event(),  # never set: only the first call stalls
        stall_once=True,
    )
    loop, _, formatter, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns.")],
        payloads=[ABSTAIN],
        critic=critic,
    )
    formatter.wait_for = critic.started

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CriticGate, "_abstain_verdict", lambda self, ctx, payload: None)
            streamed, state, _ = await _bounded(loop)

    assert "call abandoned" in caplog.text
    assert "critic_speculation: nothing held" in caplog.text
    assert "verdict discarded" not in caplog.text
    assert critic.cancelled == 1 and len(critic.ran) == 2  # abandoned, then run serially
    assert state["critic_verdict"] == SUFFICIENT
    assert _critic_verdicts(streamed) == [SUFFICIENT]
    _nothing_left_running()


# ── a hedged critic call (CRITIC_HEDGE_AFTER_MS, models.VendorRouter) ─────────
#
# The critic's streamed first token can stall like the executor's, so its call
# is sent a second time after a deadline (models.py, "The critic hedge"). The
# tests run the REAL critic LlmAgent over a VendorRouter with the critic's
# deadline set, under the run config the Chat bots use (SSE: the call streams).

SSE = RunConfig(streaming_mode=StreamingMode.SSE)


class _ScriptedCriticModel(BaseLlm):
    """The critic's model for the hedge tests. Request n replays steps[n] (the
    last repeats): "hang", a pause in seconds, or a response. `reached` is set
    once `want` requests are on the wire; `cancelled[n]` says request n was."""

    steps: list = []
    seen: int = 0
    want: int = 1
    reached: asyncio.Event | None = None
    cancelled: list = []

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        n = self.seen
        self.seen += 1
        self.cancelled.append(False)
        if self.reached is not None and self.seen >= self.want:
            self.reached.set()
        try:
            for step in self.steps[min(n, len(self.steps) - 1)]:
                if step == "hang":
                    await asyncio.Event().wait()
                elif isinstance(step, (int, float)):
                    await asyncio.sleep(step)
                else:
                    yield step
        except asyncio.CancelledError:
            self.cancelled[n] = True
            raise


def _verdict_response(verdict: dict) -> LlmResponse:
    return LlmResponse(
        content=genai_types.Content(
            role="model", parts=[genai_types.Part.from_text(text=json.dumps(verdict))]
        )
    )


def _hedged_critic(model: BaseLlm, *, after_ms: int = 50) -> LlmAgent:
    return LlmAgent(
        name="critic",
        model=VendorRouter(
            model=GEMINI_MODEL,
            gemini=model,  # type: ignore[arg-type]
            agent_hedge_after_ms={config.CRITIC_AGENT_NAME: after_ms},
        ),
        instruction="Judge the executor's tool coverage.",
        output_schema=CriticVerdict,
        output_key="critic_verdict",
        before_model_callback=lambda callback_context, llm_request: bind_model_call(
            callback_context
        ),
    )


def _logged(caplog, prefix: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


async def test_a_hedged_critic_call_abandoned_by_e1_stops_both_requests(caplog):
    """E1 cancels a critic whose spare is already out. The cancellation lands in
    the hedge race: both requests are stopped and each logs `error:cancelled`
    (the spare's line ends ` hedge=1`), one `model_hedge:` line reports the fire
    with no winner, the call is abandoned once, the verdict is the code's pass,
    and nothing is left running."""
    model = _ScriptedCriticModel(
        model=GEMINI_MODEL, steps=[["hang"]], want=2, reached=asyncio.Event(), cancelled=[]
    )
    loop, _, formatter, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "No account by that name.")],
        payloads=[ABSTAIN],
        critic=_hedged_critic(model),
    )
    formatter.wait_for = model.reached  # both requests are on the wire when the gate ends

    with caplog.at_level(logging.INFO):
        _, state, _ = await _bounded(loop, run_config=SSE)

    calls = _logged(caplog, "model_call: ")
    assert len(calls) == 2
    assert all(" agent=critic " in c and " status=error:cancelled " in c for c in calls)
    assert sorted(c.endswith(" hedge=1") for c in calls) == [False, True]
    [fired] = _logged(caplog, "model_hedge: ")
    assert " agent=critic fired=1 winner=- " in fired and fired.endswith(" tenant=anomaly")
    assert caplog.text.count("critic_speculation: call abandoned") == 1
    assert _logged(caplog, "hedge_capped:") == []
    assert model.seen == 2 and model.cancelled == [True, True]
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS
    _nothing_left_running()


async def test_a_hedged_critic_verdict_is_relayed_once(caplog):
    """The critic's first request stalls and its spare answers: the pass is
    judged by the spare's verdict, once, and the stalled request is cancelled."""
    model = _ScriptedCriticModel(
        model=GEMINI_MODEL,
        steps=[["hang"], [0.01, _verdict_response(SUFFICIENT)]],
        cancelled=[],
    )
    loop, executor, _, _ = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns.")],
        payloads=[ANSWER],
        critic=_hedged_critic(model),
    )

    with caplog.at_level(logging.INFO):
        streamed, state, _ = await _bounded(loop, run_config=SSE)

    assert model.seen == 2 and model.cancelled == [True, False]
    assert _critic_verdicts(streamed) == [SUFFICIENT]  # one verdict reaches the stream
    assert state["critic_verdict"] == SUFFICIENT and executor.runs == 1
    [fired] = _logged(caplog, "model_hedge: ")
    assert " agent=critic fired=1 winner=hedge " in fired
    assert _logged(caplog, "hedge_capped:") == []
    _nothing_left_running()


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
async def test_the_critic_of_a_second_pass_shares_the_invocations_one_hedge(
    parallel, monkeypatch, caplog
):
    """CRITIC_SKIP_FINAL_PASS=0 makes the critic judge both passes. Both wirings
    (the serial one is CRITIC_PARALLEL=0 and the resolver's serial fallback)
    reach the same VendorRouter as `critic` and draw on the same budget of one
    hedge per invocation: pass 1's stall is rescued, pass 2's waits on its own
    request and says why."""
    monkeypatch.setattr(config, "CRITIC_SKIP_FINAL_PASS", False)
    model = _ScriptedCriticModel(
        model=GEMINI_MODEL,
        steps=[
            ["hang"],  # pass 1, first request: stalls
            [0.01, _verdict_response(INSUFFICIENT)],  # pass 1's spare
            [0.3, _verdict_response(SUFFICIENT)],  # pass 2: slow, but healthy
        ],
        cancelled=[],
    )
    loop, executor, _, _ = _pipeline(
        parallel=parallel,
        executor_passes=[(True, "12 live campaigns."), (True, "The Q3 push is live.")],
        payloads=[ANSWER, ANSWER_2],
        critic=_hedged_critic(model),
    )

    with caplog.at_level(logging.INFO):
        streamed, state, _ = await _bounded(loop, run_config=SSE)

    assert executor.runs == 2
    assert model.seen == 3 and model.cancelled == [True, False, False]  # no second spare
    assert _critic_verdicts(streamed) == [INSUFFICIENT, SUFFICIENT]
    assert state["critic_verdict"] == SUFFICIENT
    [fired] = _logged(caplog, "model_hedge: ")
    assert " agent=critic fired=1 winner=hedge " in fired
    [capped] = _logged(caplog, "hedge_capped:")
    assert re.match(
        r"^hedge_capped: agent=critic after_ms=50 used=1 limit=1 inv=\S+ tenant=anomaly$", capped
    )
    _nothing_left_running()


# ── REQUERY_SKIPS_FORMATTER through the real format gate ─────────────────────
#
# The stand-in `_Gate` above cannot take the shortcut: it is the real
# FormatGate that writes the abstain for a from-memory first pass. These run
# the deployed pair (FormatGate, CriticResolver or CriticGate) on a real runner.

REQUERY_REASON = (
    "Deterministic: no tool call this turn and the format gate abstained, "
    "so the draft was written from memory (no critic LLM run)."
)
REQUERY_FEEDBACK = (
    "You made NO tool call this turn, so your draft repeated earlier answers from memory "
    "and nothing in it could be grounded. Query GUB now for THIS question - do not reuse "
    "prior turns - and answer only from what the tools return."
)


def _real_gate_pipeline(*, parallel: bool, executor_passes: list, payloads: list, delay: float):
    """`_pipeline` with the real FormatGate around the formatter stand-in. The
    formatter renders for `delay` seconds, so a critic call that is going to
    start beside it has started before it ends (as it does in production,
    where the formatter takes seconds)."""
    executor = _Executor(name=AGENT_NAME, passes=executor_passes, runs=0)
    formatter = _Formatter(name="formatter", payloads=payloads, runs=0, delay=delay)
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    critic = _Critic(
        name="critic", verdicts=[INSUFFICIENT], ran=[], saw=[], session_authors=[], raises=None
    )
    loop = build_deep_agent(
        executor, gate, critic, EscalateIfSufficient(name="loop_escalator"), parallel=parallel
    )
    return loop, executor, formatter, critic


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "serial"])
async def test_a_from_memory_first_pass_is_requeried_with_no_formatter_and_no_critic_call(
    parallel, monkeypatch, caplog
):
    """Pass 1 answers from memory: the gate abstains in code, the verdict is
    today's re-query word for word, and pass 2 runs and is formatted — the
    stand-in formatter is called once, for pass 2 only. No critic call is made
    for either pass (pass 2 is the loop's last), so nothing is left to cancel,
    to discard or to wait for."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    loop, executor, formatter, critic = _real_gate_pipeline(
        parallel=parallel,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN],
        delay=0.05,
    )

    with caplog.at_level(logging.INFO):
        streamed, state, session_events = await _run(loop, question="whats new")

    assert formatter.runs == 1  # pass 2's; pass 1's was never called
    assert critic.ran == []  # neither pass: pass 1 is settled in code, pass 2 is the last
    assert executor.runs == 2
    (requery,) = _critic_verdicts(streamed)  # pass 2's pass is the gate's, not the critic's
    assert requery["reason"] == REQUERY_REASON
    assert requery["feedback"] == REQUERY_FEEDBACK
    assert requery["sufficient"] is False and requery["info_sufficient"] is False
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS
    assert streamed[-1].author == "loop_escalator" and streamed[-1].actions.escalate
    # the send-back reads as the critic's, after the gate's own abstain
    order = [e.author for e in _complete(session_events) if e.author in ("format_gate", "critic")]
    assert order[:2] == ["format_gate", "critic"]
    # the shortcut line, once; and nothing about a critic call that was made
    assert len([m for m in caplog.messages if m.startswith("critic_requery: shortcut")]) == 1
    for noise in ("nothing held", "verdict discarded", "call abandoned"):
        assert noise not in caplog.text
    # nothing running, nothing held for the next pass or turn
    assert critic_module._RUNNING == {}
    assert critic_module._SPECULATIONS == {}


async def test_the_shortcut_pass_does_not_wait_for_a_stalled_critic(monkeypatch):
    """The race the speculative critic's skip exists for: the gate ends at
    once, long before a critic task could be registered, and the join must not
    wait out a call (a tenant's p90 is 42 s) whose verdict is discarded. The critic here
    waits for an event nobody sets, so a pass that starts it cannot finish."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    loop, executor, formatter, critic = _real_gate_pipeline(
        parallel=True,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN],
        delay=0.0,
    )
    critic.wait_for, critic.timeout = asyncio.Event(), 30.0
    try:
        await asyncio.wait_for(_run(loop, question="whats new"), timeout=3.0)
    except TimeoutError:
        pytest.fail("the from-memory pass is waiting for a critic call it does not need")
    assert critic.ran == [] and formatter.runs == 1 and executor.runs == 2
    _nothing_left_running()


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "serial"])
async def test_with_the_flag_at_0_the_formatter_and_the_critic_start_as_before(
    parallel, monkeypatch
):
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", False)
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    loop, executor, formatter, critic = _real_gate_pipeline(
        parallel=parallel,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN],
        delay=0.05,
    )

    streamed, _, _ = await _run(loop, question="whats new")

    assert formatter.runs == 2  # pass 1 is formatted too
    # the speculative call is started beside the formatter and cancelled on the
    # abstain (parallel); the serial wiring never reaches its critic
    assert len(critic.ran) == (1 if parallel else 0)
    (requery,) = _critic_verdicts(streamed)
    assert requery["reason"] == REQUERY_REASON and requery["feedback"] == REQUERY_FEEDBACK


async def test_a_pass_with_a_tool_call_still_runs_the_formatter_and_the_critic(monkeypatch):
    """Eligibility is the tool call, not the evidence index: this pass indexed
    nothing, and is still formatted and judged."""
    # One pass, as before #102: the searched abstain is not sent back to confirm.
    monkeypatch.setattr(config, "CONFIRM_ABSTENTION", False)
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    loop, executor, formatter, critic = _real_gate_pipeline(
        parallel=True,
        executor_passes=[(True, "GUB has no account by that name.")],
        payloads=[ABSTAIN],
        delay=0.05,
    )

    _, state, _ = await _run(loop)

    assert formatter.runs == 1 and executor.runs == 1
    assert len(critic.ran) == 1  # started beside the formatter, as before
    assert state["critic_verdict"]["reason"] == ABSTAIN_PASS


def _stated(name: str, key: str) -> list[str]:
    return [
        line.partition("=")[2].strip()
        for line in (REPO / name).read_text().splitlines()
        if line.startswith(f"{key}=")
    ]


def test_both_deploy_env_files_state_the_requery_flag_with_one_value():
    """Stated, not defaulted — the rollback is a line already in front of the
    operator — ON, and the same in both, so a sandbox run measures what
    production runs."""
    prod = _stated("deploy-prod.env", "REQUERY_SKIPS_FORMATTER")
    sandbox = _stated("deploy-sandbox.env", "REQUERY_SKIPS_FORMATTER")
    assert prod == sandbox == ["1"], (prod, sandbox)


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("0", False), ("1", True)], ids=["unset", "0", "1"]
)
def test_the_requery_flag_defaults_to_todays_behaviour(value, expected):
    """Read at import, so each setting gets a fresh interpreter. Unset is the
    code default and the rollback: the formatter is called."""
    env = {k: v for k, v in os.environ.items() if k != "REQUERY_SKIPS_FORMATTER"}
    if value is not None:
        env["REQUERY_SKIPS_FORMATTER"] = value
    env["PYTHONPATH"] = str(REPO)
    out = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            "from gub_agent import config; print(config.REQUERY_SKIPS_FORMATTER)",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.strip().splitlines()[-1] == str(expected)


async def test_a_first_pass_with_no_text_and_no_tool_call_still_starts_the_critic(
    monkeypatch, caplog
):
    """The executor wrote nothing (it died): the gate emits nothing, so the
    critic started beside it is the only verdict there will be and must not be
    skipped as if the pass were a from-memory draft."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    loop, executor, formatter, critic = _real_gate_pipeline(
        parallel=True,
        executor_passes=[(False, ""), (True, "3 hires, 12 live.")],
        payloads=[ANSWER],
        delay=0.05,
    )

    with caplog.at_level(logging.INFO):
        await _run(loop, question="whats new")

    assert len(critic.ran) == 1  # pass 1's speculative call was made (pass 2 is the last)
    assert "critic_speculation: nothing held" not in caplog.text
    assert "critic_requery: shortcut" not in caplog.text


async def test_the_shortcut_pass_leaves_no_trace_in_the_critics_later_requests(monkeypatch):
    """Under the parallel wiring the speculative critic runs on a sibling branch
    of the gate. The gate's code abstain must be hidden from it, as the
    formatter's was: with CRITIC_SKIP_FINAL_PASS=0 the critic judges pass 2,
    and its request must carry no `[format_gate] said` payload of pass 1."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    monkeypatch.setattr(config, "CRITIC_SKIP_FINAL_PASS", False)
    loop, executor, formatter, critic = _real_gate_pipeline(
        parallel=True,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ANSWER],
        delay=0.05,
    )

    _, _, session_events = await _run(loop, question="whats new")

    assert len(critic.ran) == 1 and formatter.runs == 1  # only pass 2 was judged and formatted
    assert "[format_gate] said" not in critic.saw[0]
    assert "NO_COMPANY_RECORDS" not in critic.saw[0]
    abstains = [
        e
        for e in session_events
        if e.author == "format_gate" and e.content and "abstain" in e.content.parts[0].text
    ]
    assert abstains and all(e.branch for e in abstains)  # in the session, on the gate's branch

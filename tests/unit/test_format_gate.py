"""
agents.format_gate — the deterministic checks and the retry machinery.

Pure-function halves (gate_problems, compose_brief, template_payload) are
tested directly; the gate's control flow (feedback event → formatter re-run →
template after two failed retries; deterministic abstention with NO formatter
LLM run) runs against a scripted formatter stand-in on a real ADK
InvocationContext, the way test_sandbox.py drives CriticGate.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.utils._schema_utils import validate_schema
from google.genai import types as genai_types
from pydantic import ValidationError

from gub_agent import config
from gub_agent.agents import format_gate as format_gate_module
from gub_agent.agents.evidence_index import (
    evidence_index,
    formatter_brief,
    record_evidence,
    reset_evidence_index,
    set_answer_draft,
)
from gub_agent.agents.format_gate import (
    MAX_FORMAT_ATTEMPTS,
    FormatGate,
    _ContractScan,
    _explain_validation,
    compose_brief,
    gate_problems,
    repair_citations,
    template_payload,
)
from gub_agent.agents.formatter import formatter_agent
from gub_agent.config import AGENT_NAME, GEMINI_MODEL
from gub_agent.models import VendorRouter
from gub_agent.schemas import AnswerPayload
from gub_agent.schemas.answer import HEADLINE_MAX_WORDS, count_words

INV = "inv-1"


@pytest.fixture(autouse=True)
def _formatter_always_runs(monkeypatch):
    """This file describes the gate with the from-memory shortcut OFF, the code
    default. The deployed value is 1: the tests of the shortcut below set it
    themselves, and a shell that sources the deploy env must not turn the others
    red."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", False, raising=False)


@pytest.fixture(autouse=True)
def _no_evidence_rows_event(monkeypatch):
    """The gate's event lists below are pinned without the evidence-rows event
    (EVIDENCE_ROWS_EVENT, tests/unit/test_evidence_rows_event.py), whatever the
    process env says — the deploy env turns it on."""
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)


def _seed_index(invocation_id: str = INV) -> dict:
    """A believable turn: an account row, a campaign row, an aggregate total."""
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    record_evidence(
        SimpleNamespace(name="org_query"),
        {},
        ctx,
        {
            "results": [{"id": "a1", "name": "chevy", "status": "active"}],
            "total": 1,
        },
    )
    record_evidence(
        SimpleNamespace(name="get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "Q3 push", "status": "live", "budget": 1200000},
    )
    return evidence_index(invocation_id)


def _payload(**over) -> AnswerPayload:
    base = {
        "kind": "answer",
        "headline": "chevy is in good shape",
        "blocks": [{"kind": "text", "text": "the Q3 push is live with budget 1200000."}],
        "citations": ["get_campaign:c1:status"],
        "facts": [
            {
                "evidence_id": "get_campaign:c1:status",
                "entity_id": "c1",
                "field": "status",
                "value": "live",
            }
        ],
    }
    base.update(over)
    return AnswerPayload.model_validate(base)


# ── gate_problems, pure ───────────────────────────────────────────────────────


async def test_a_grounded_payload_passes():
    assert gate_problems(_payload(), _seed_index()) == []


async def test_unknown_citation_fails_hard():
    index = _seed_index()
    payload = _payload(
        citations=["org_query:nope"],
        facts=[{"evidence_id": "org_query:nope", "value": "x"}],
    )
    problems = gate_problems(payload, index)
    assert len(problems) == 1
    assert "unknown citation" in problems[0]
    assert "org_query:nope" in problems[0]


async def test_ungrounded_number_is_named():
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "the Q3 push is live with budget 999."}]),
        _seed_index(),
    )
    assert any("ungrounded number" in p and "999" in p for p in problems)


async def test_grounded_numbers_survive_reformatting_commas():
    """1,200,000 in the answer grounds against 1200000 in the evidence."""
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "the Q3 push budget is 1,200,000."}]),
        _seed_index(),
    )
    assert problems == []


async def test_ungrounded_entity_is_the_critics_old_hard_fail():
    """'Chevrolet' when the tool returned 'chevy' — the exact case the critic
    used to hard-fail on, now caught in code."""
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "latest from Chevrolet is strong."}]),
        _seed_index(),
    )
    assert any("ungrounded entity" in p and "Chevrolet" in p for p in problems)


async def test_possessives_and_plurals_are_the_same_entity():
    """The largest class of FALSE `ungrounded entity` rejections in the
    2026-09-10 engine logs: the evidence carries "chevy", the answer writes
    "Chevy's", and the turn burned a formatter attempt over an apostrophe."""
    index = _seed_index()  # evidence says "chevy", "Q3 push"
    for text in ("chevy's quarter improved.", "the Q3 pushes are live."):
        problems = gate_problems(_payload(blocks=[{"kind": "text", "text": text}]), index)
        assert not any("ungrounded entity" in p for p in problems), f"{text!r} -> {problems}"


async def test_a_sentence_opener_does_not_drag_a_grounded_name_down():
    """ "While Chevy...", "Although Budweiser...", "Two Chevrolet..." were all
    rejected in live turns although the entity itself was grounded — only the
    opening word was missing from the tool results."""
    index = _seed_index()
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "While chevy held, the Q3 push stayed live."}]),
        index,
    )
    assert not any("ungrounded entity" in p for p in problems), problems


async def test_trimming_an_opener_does_not_smuggle_a_fabricated_name():
    """The hole the trim above opens, closed: once the opener is gone what is
    left is a name and must still be grounded."""
    problems = gate_problems(
        _payload(
            blocks=[{"kind": "text", "text": "While Tesla dominates, the Q3 push stayed live."}]
        ),
        _seed_index(),
    )
    assert any("ungrounded entity" in p and "Tesla" in p for p in problems), problems


async def test_grounded_entity_and_sentence_openers_pass():
    problems = gate_problems(
        _payload(
            headline="chevy is healthy",
            blocks=[
                {
                    "kind": "bullets",
                    # "Overall" / "Nothing" open their bullets — grammar, not
                    # names; "Q3 push" is grounded (case-insensitive).
                    "items": ["Overall the Q3 Push is live", "Nothing else moved recently"],
                }
            ],
        ),
        _seed_index(),
    )
    assert problems == []


async def test_abstain_and_clarify_skip_grounding_but_not_citation_checks():
    index = _seed_index()
    abstain = AnswerPayload.model_validate({"kind": "abstain", "headline": "NO_COMPANY_RECORDS"})
    assert gate_problems(abstain, index) == []
    clarify = AnswerPayload.model_validate(
        {"kind": "clarify", "headline": "Which Chevrolet do you mean?"}
    )
    assert gate_problems(clarify, index) == []  # echoes the user's word, legitimately
    bad_abstain = AnswerPayload.model_validate(
        {
            "kind": "abstain",
            "headline": "NO_COMPANY_RECORDS",
            "citations": ["org_query:nope"],
            "facts": [{"evidence_id": "org_query:nope"}],
        }
    )
    assert any("unknown citation" in p for p in gate_problems(bad_abstain, index))


async def test_table_source_id_cells_do_not_trip_number_grounding():
    index = _seed_index()
    payload = _payload(
        blocks=[
            {
                "kind": "table",
                "columns": ["Campaign", "Status", "Source"],
                "rows": [["Q3 push", "live", "get_campaign:c1:status"]],
            }
        ],
    )
    assert gate_problems(payload, index) == []


# ── the brief states the rules BEFORE the first attempt ──────────────────────


async def test_the_brief_carries_the_grounding_rules_on_attempt_one():
    """Until 2026-09-11 the brief held no rule at all: the formatter met them
    only by failing, which cost an attempt every time. `ungrounded entity` was
    39 of 88 rejections."""
    brief = compose_brief("chevy is fine.", _seed_index(), feedback="")
    assert "RULES (checked in code" in brief
    # BE-3: names and numbers ground in the executor answer AND a tool result
    # (the whole turn's results, not only the rows listed in the brief)
    assert "must appear in the executor answer and in a tool result" in " ".join(brief.split())
    # the specific behaviour that produced the biggest bucket
    assert "Do not coin section headings" in brief


async def test_an_empty_index_tells_the_formatter_to_abstain():
    """With nothing citable, kind="answer" cannot validate — so say which kinds
    ARE available instead of letting the attempt fail at an impossible payload."""
    brief = compose_brief("nothing in our records on that.", {}, feedback="")
    assert "(none — this turn retrieved nothing citable)" in brief
    assert 'kind="abstain"' in brief
    assert "RULES (checked in code" not in brief  # the grounding block is moot


async def test_feedback_tells_an_invented_label_to_go_away_not_to_be_renamed():
    """ "Use the exact name the tool returned" is unfollowable for a coined
    heading — there is no such name — so the model invented a different label
    and failed again. Near-misses keep the old advice."""
    index = _seed_index()  # evidence says "chevy"
    misspelled = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "latest from Chevrolet is strong."}]), index
    )
    assert any("use the exact name the tool returned" in p for p in misspelled)

    invented = gate_problems(
        _payload(blocks=[{"kind": "bullets", "items": ["Net Momentum held up this quarter"]}]),
        index,
    )
    # "Net" is trimmed as the bullet's opener, so the run reported is
    # "Momentum" — partial, but it still names the phrase the model must drop.
    assert any("Momentum" in p and "no tool result" in p for p in invented)
    assert any("start the bullet with the claim" in p for p in invented)
    # the log taxonomy stays "ungrounded entity" for both
    assert all("ungrounded entity" in p for p in misspelled + invented)


# ── citation repair, pure ─────────────────────────────────────────────────────


def _cited(*ids) -> AnswerPayload:
    return AnswerPayload.model_validate(
        {
            "kind": "answer",
            "headline": "the Q3 push is live",
            "blocks": [{"kind": "text", "text": "the Q3 push is live."}],
            "citations": list(ids),
            "facts": [{"evidence_id": i, "value": "live"} for i in ids],
        }
    )


async def test_repair_restores_a_dropped_tool_prefix():
    """Live attempt 1 of the reported turn: the bare row id, `org_query:` gone."""
    index = _seed_index()
    bare = "a1"  # org_query:a1 is in the index
    payload, repaired = repair_citations(_cited(bare), index)
    assert repaired and payload.citations == ["org_query:a1"]
    # facts carry the same ids and the contract checks both directions
    assert [f.evidence_id for f in payload.facts] == ["org_query:a1"]
    assert not any("unknown citation" in p for p in gate_problems(payload, index))


async def test_repair_resolves_a_spliced_id_by_its_unique_head():
    """Live attempt 2: head of one row's id, tail of another's. The leading
    characters identify one row, so the fix is a lookup, not a guess."""
    index = {
        "org_query:4bf32a55-6e79-4f9e-86f8-f6dfd0c776e4": {"value": "see the usa"},
        "org_query:0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da": {"value": "t1-1 hd mcp"},
    }
    spliced = "org_query:4bf32a55-6c13-4fea-8dd6-6a93bb0fb6da"
    payload, repaired = repair_citations(_cited(spliced), index)
    assert payload.citations == ["org_query:4bf32a55-6e79-4f9e-86f8-f6dfd0c776e4"]
    assert repaired


async def test_repair_stays_ambiguous_between_two_different_rows():
    """The entity-row tie-break applies only to a row and its OWN field rows.
    Two unrelated rows sharing the matched text must stay a rejection."""
    index = {
        "org_query:a1": {"value": "one", "field": None},
        "org_query:a1b": {"value": "two", "field": None},
    }
    payload, repaired = repair_citations(_cited("a1"), index)
    assert repaired == []
    assert any("unknown citation" in p for p in gate_problems(payload, index))


async def test_repair_refuses_to_guess_and_leaves_the_rejection_standing():
    index = {
        "org_query:4bf32a55-6e79-4f9e-86f8-f6dfd0c776e4": {"value": "see the usa"},
    }
    for invented in ("org_query:deadbeef-0000-0000-0000-000000000000", "totally-made-up"):
        payload, repaired = repair_citations(_cited(invented), index)
        assert repaired == []
        assert any("unknown citation" in p for p in gate_problems(payload, index))


async def test_repair_is_a_no_op_on_correct_citations():
    index = _seed_index()
    payload, repaired = repair_citations(_cited("org_query:a1"), index)
    assert repaired == [] and payload.citations == ["org_query:a1"]


# ── brief + template, pure ────────────────────────────────────────────────────


async def test_brief_carries_answer_evidence_and_feedback():
    index = _seed_index()
    brief = compose_brief("chevy is fine.", index, feedback="ungrounded number: 999")
    assert "EXECUTOR ANSWER" in brief
    assert "chevy is fine." in brief
    assert "- get_campaign:c1:budget = 1200000" in brief
    assert "FORMAT_FEEDBACK" in brief and "999" in brief
    assert "FORMAT_FEEDBACK" not in compose_brief("x", index, feedback="")


async def test_template_render_shows_executor_prose_not_evidence_rows():
    index = _seed_index()
    payload = template_payload("Here is the long executor answer\nmore detail", index)
    assert payload.kind == "answer"
    assert payload.headline == "Here is the long executor answer"  # verbatim — no filler veto
    # The BODY is the executor's prose. Evidence values are a tool's response
    # row; rendering them is what shipped raw JSON to users (2026-09-11).
    body = payload.blocks[0]
    assert body.kind == "text"
    assert body.text == "more detail"
    # Evidence still travels, just not as body text: citations feed the bot's
    # attribution chips and facts feed the conflict filter (blend 05).
    assert set(payload.citations) == {f.evidence_id for f in payload.facts}
    assert all(cid in index for cid in payload.citations)
    # entity rows preferred over per-field rows
    assert "org_query:a1" in payload.citations


async def test_template_never_renders_a_tool_row_when_there_is_prose():
    """The live regression: good executor prose, evidence values that are
    serialized tool rows. The reader must see the prose."""
    index = {
        "org_query:results0": {
            "value": '{"count": 1, "totalBudget": "1500000"}',
            "entity_id": None,
            "field": None,
        },
        "org_query:c1": {
            "value": '{"id": "c1", "accountId": "a9", "name": "T1-1 HD MCP", "status": "pitch"}',
            "entity_id": "c1",
            "field": None,
        },
    }
    prose = (
        "In the last month, the most significant movement has been centered on Chevy's "
        "Heavy Duty (HD) truck portfolio, with a new pitch opened.\n"
        "The T1-1 HD MCP pitch carries a $1.5M budget."
    )
    payload = template_payload(prose, index)
    rendered = (
        payload.headline
        + " "
        + " ".join(getattr(b, "text", " ".join(getattr(b, "items", []))) for b in payload.blocks)
    )
    assert '{"id"' not in rendered and '"accountId"' not in rendered
    assert "T1-1 HD MCP pitch carries" in rendered
    # and the headline stops at a clause boundary rather than mid-phrase
    assert not payload.headline.rstrip(" …").endswith(("with a", "with", "a"))
    assert count_words(payload.headline) <= HEADLINE_MAX_WORDS


async def test_template_with_no_evidence_falls_back_to_executor_text():
    reset_evidence_index(SimpleNamespace(invocation_id=INV))
    payload = template_payload("nothing was retrieved this turn", {})
    assert payload.blocks[0].kind == "text"
    assert payload.citations == []


async def test_template_always_carries_a_block_for_kind_answer():
    """`kind="answer"` with no block is a contract violation the gate itself
    must not emit — it is built with model_construct, so nothing would catch
    it downstream."""
    for text in ("", "one line only.", "Short.\nmore"):
        assert template_payload(text, {}).blocks, f"no block for {text!r}"
        assert template_payload(text, _seed_index()).blocks, f"no block for {text!r}"


# ── the gate's control flow, on a real InvocationContext ─────────────────────


@dataclass
class Streamed:
    """A formatter reply streamed the way ADK relays an SSE call: one partial
    event per chunk, each carrying only its own slice of the text. A stream
    that is not closed then ends the way LlmAgent ends it: the WHOLE text is
    validated against `output_schema` (ADK's `validate_schema`, which raises
    before anything is yielded) and one final event carries the text and the
    state_delta."""

    text: str
    chunks: list[str] | None = None
    size: int = 16

    def pieces(self) -> list[str]:
        if self.chunks is not None:
            assert "".join(self.chunks) == self.text
            return self.chunks
        return [self.text[i : i + self.size] for i in range(0, len(self.text), self.size)]


class ScriptedFormatter(BaseAgent):
    """Stand-in for the formatter LLM: plays back one scripted outcome per run
    — a payload dict (emitted the way LlmAgent's output_schema does: JSON text
    + state_delta), a `Streamed` reply, or a ValidationError to raise. The last
    item repeats."""

    script: list = []
    runs: list = []  # one entry per run — mutated in place, pydantic-safe
    briefs: list = []  # per run: the brief (the model input) it was given
    closed: list = []  # per run: None if it ran to its end, else the text streamed before its close
    emitted: list = []  # every event it yielded, in order

    async def _run_async_impl(self, ctx):
        item = self.script[min(len(self.runs), len(self.script) - 1)]
        self.runs.append(ctx.invocation_id)
        self.briefs.append(formatter_brief(ctx.invocation_id))
        self.closed.append(None)
        if isinstance(item, Exception):
            raise item
        text = json.dumps(item) if isinstance(item, dict) else item.text
        if isinstance(item, Streamed):
            streamed = ""
            try:
                for piece in item.pieces():
                    streamed += piece
                    event = Event(
                        invocation_id=ctx.invocation_id,
                        author=self.name,
                        partial=True,
                        content=genai_types.Content(
                            role="model", parts=[genai_types.Part(text=piece)]
                        ),
                    )
                    self.emitted.append(event)
                    yield event
            except GeneratorExit:
                self.closed[-1] = streamed
                raise
            item = validate_schema(AnswerPayload, item.text)
        event = Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
            actions=EventActions(state_delta={"answer_payload": item}),
        )
        self.emitted.append(event)
        yield event


def _scripted(script: list) -> ScriptedFormatter:
    return ScriptedFormatter(
        name="formatter", script=script, runs=[], briefs=[], closed=[], emitted=[]
    )


async def _gate_ctx(executor_text: str | None) -> InvocationContext:
    service = InMemorySessionService()
    session = await service.create_session(app_name="gub", user_id="u", state={})
    if executor_text is not None:
        await service.append_event(
            session,
            Event(
                invocation_id=INV,
                author=AGENT_NAME,
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=executor_text)]
                ),
            ),
        )
    return InvocationContext(
        session_service=service,
        invocation_id=INV,
        agent=BaseAgent(name="host"),
        session=session,
    )


def _valid_dict() -> dict:
    return _payload().model_dump(exclude_none=True)


async def test_gate_passes_a_valid_payload_through_untouched():
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 1
    assert [e.author for e in events] == ["formatter"]
    assert events[0].actions.state_delta["answer_payload"]["headline"] == "chevy is in good shape"


async def test_gate_retries_with_feedback_then_accepts():
    _seed_index()
    bad = _payload(
        citations=["org_query:nope"],
        facts=[{"evidence_id": "org_query:nope", "value": "x"}],
    ).model_dump(exclude_none=True)
    formatter = _scripted([bad, _valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 2
    feedback_events = [
        e
        for e in events
        if e.author == "format_gate"
        and e.actions
        and (e.actions.state_delta or {}).get("format_feedback")
    ]
    assert len(feedback_events) == 1
    assert "unknown citation" in feedback_events[0].actions.state_delta["format_feedback"]
    # last payload-bearing event is the accepted formatter one, not a template
    assert events[-1].author == "formatter"


async def test_an_empty_evidence_index_costs_one_attempt_not_three():
    """The 20-rejection cluster of 2026-09-10, all the same message.

    With nothing citable, `kind="answer"` cannot validate — the contract wants
    a citation — so feedback cannot repair it and each retry spends a model
    call on an impossible payload. The formatter still gets ONE call (only it
    can tell an abstention from a clarification); after that the gate settles.
    """
    reset_evidence_index(SimpleNamespace(invocation_id=INV))  # no evidence at all
    uncitable = {
        "kind": "answer",
        "headline": "nothing in our records",
        "blocks": [],
        "citations": [],
    }
    formatter = _scripted([uncitable])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("I have nothing on that in company records.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 1, f"formatter ran {len(formatter.runs)}x — should cost one call"
    assert len(formatter.runs) < MAX_FORMAT_ATTEMPTS
    # and the turn still ends with a payload for the bot
    assert events, "the turn must not end payload-less"


async def test_an_empty_index_still_lets_the_formatter_choose_abstain():
    """The one call it does get is real: a valid abstain passes straight
    through, no template, no second attempt."""
    reset_evidence_index(SimpleNamespace(invocation_id=INV))
    abstain = {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}
    formatter = _scripted([abstain])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("not in our records.")

    [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 1


async def test_gate_emits_template_after_two_failed_retries():
    _seed_index()
    bad = _payload(
        citations=["org_query:nope"],
        facts=[{"evidence_id": "org_query:nope", "value": "x"}],
    ).model_dump(exclude_none=True)
    formatter = _scripted([bad])  # every attempt returns the same bad payload
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 3  # initial + two retries
    last = events[-1]
    assert last.author == "format_gate"
    template = last.actions.state_delta["answer_payload"]
    assert template["kind"] == "answer"
    assert json.loads(last.content.parts[0].text) == template


async def test_pydantic_validation_errors_take_the_same_retry_path():
    _seed_index()
    try:
        AnswerPayload.model_validate({"kind": "answer", "headline": "Sure, here it is"})
    except ValidationError as exc:
        boom = exc
    formatter = _scripted([boom, _valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 2
    feedback = next(
        (e.actions.state_delta or {}).get("format_feedback", "")
        for e in events
        if e.author == "format_gate" and e.actions and e.actions.state_delta
    )
    assert "invalid payload" in feedback


async def test_exact_abstention_becomes_a_payload_with_no_formatter_run():
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("NO_COMPANY_RECORDS")

    events = [e async for e in gate.run_async(ctx)]

    assert formatter.runs == []  # deterministic — no LLM for one marker word
    assert len(events) == 1
    payload = events[0].actions.state_delta["answer_payload"]
    assert payload["kind"] == "abstain"
    assert events[0].author == "format_gate"


async def test_no_executor_text_means_no_events_and_no_formatter_run():
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx(None)

    events = [e async for e in gate.run_async(ctx)]

    assert events == []
    assert formatter.runs == []


async def test_critic_gate_recognises_the_abstain_payload():
    """The pipeline invariant: NO_COMPANY_RECORDS keeps skipping the critic
    LLM whether it arrives as the bare marker or as the typed abstain payload
    the format gate emitted this pass. After a search, the verdict written in
    code sends it back once to be confirmed by a second route (#102)."""
    from gub_agent.agents.critic import CriticGate

    class _RecordingCritic(BaseAgent):
        ran: list = []

        async def _run_async_impl(self, ctx):
            self.ran.append(ctx.invocation_id)
            yield Event(invocation_id=ctx.invocation_id, author=self.name)

    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(
        name="critic_gate", sub_agents=[critic], payload_authors=("format_gate", "formatter")
    )

    service = InMemorySessionService()
    session = await service.create_session(app_name="gub", user_id="u")
    # The executor looked (a tool call this turn) and found nothing: its text
    # is NOT the bare marker — only the payload says abstain.
    await service.append_event(
        session,
        Event(
            invocation_id=INV,
            author=AGENT_NAME,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(name="org_query", args={})
                    )
                ],
            ),
        ),
    )
    await service.append_event(
        session,
        Event(
            invocation_id=INV,
            author=AGENT_NAME,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text="I cannot answer that from GUB.")]
            ),
        ),
    )
    # …and the format gate rendered that as the typed abstention.
    await service.append_event(
        session,
        Event(
            invocation_id=INV,
            author="format_gate",
            actions=EventActions(
                state_delta={
                    "answer_payload": {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}
                }
            ),
        ),
    )
    ctx = InvocationContext(
        session_service=service, invocation_id=INV, agent=BaseAgent(name="host"), session=session
    )

    events = [e async for e in gate.run_async(ctx)]

    assert critic.ran == []
    verdict = events[0].actions.state_delta["critic_verdict"]
    assert verdict["sufficient"] is False
    assert "abstention after a search" in verdict["reason"]


# ── early abort (FORMAT_GATE_EARLY_ABORT=contract) ───────────────────────────
#
# The formatter's reply streams; ADK validates it only at the end. With the
# flag on the gate runs the contract's field checks on `headline` + `blocks`
# the moment `blocks` closes and, on a field error, closes the attempt and
# takes today's retry with today's feedback.

EXEC_TEXT = "chevy is in good shape — the Q3 push is live."
# 26 words: one over BULLET_MAX_WORDS, the q14 (2026-09-28) rejection.
LONG_BULLET = "the Q3 push is live and " + " ".join(["moving"] * 20)
REJECTION = (
    "invalid payload — blocks.0.BulletBlock.items: Value error, "
    "bullet 0 is 26 words, over the 25-word budget"
)


def _bulleted(first: str = "the Q3 push is live", **over) -> dict:
    """A grounded answer whose first bullet is `first`, keys in the order
    Gemini streams them (kind, headline, blocks, citations, facts, …)."""
    payload = {
        "kind": "answer",
        "headline": "chevy is in good shape",
        "blocks": [{"kind": "bullets", "items": [first, "the chevy account is active"]}],
        "citations": ["get_campaign:c1:status"],
        "facts": [
            {
                "evidence_id": "get_campaign:c1:status",
                "entity_id": "c1",
                "field": "status",
                "value": "live",
            }
        ],
        "follow_ups": ["what is the Q3 push budget?"],
    }
    payload.update(over)
    return payload


def _full_feedback(text: str) -> str:
    """Today's feedback for a contract failure: ADK validates the WHOLE text
    and the gate explains the error."""
    with pytest.raises(ValidationError) as caught:
        validate_schema(AnswerPayload, text)
    return _explain_validation(caught.value)


def _feedbacks(events: list) -> list[str]:
    return [
        e.actions.state_delta["format_feedback"]
        for e in events
        if e.author == "format_gate" and e.actions and "format_feedback" in e.actions.state_delta
    ]


def _stored(events: list) -> list[tuple]:
    """What the session stores of a run: its non-partial events, as data."""
    return [
        (
            e.author,
            "".join(p.text or "" for p in e.content.parts) if e.content and e.content.parts else "",
            json.dumps(e.actions.state_delta if e.actions else {}, sort_keys=True),
        )
        for e in events
        if not e.partial
    ]


def _gate_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "gub_agent.agents.format_gate"]


async def _run_gate(monkeypatch, flag: str, script: list) -> tuple[list, ScriptedFormatter]:
    monkeypatch.setattr(format_gate_module, "FORMAT_GATE_EARLY_ABORT", flag)
    _seed_index()
    formatter = _scripted(script)
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx(EXEC_TEXT)
    return [e async for e in gate.run_async(ctx)], formatter


def test_the_fixtures_carry_what_the_tests_are_about():
    assert count_words(LONG_BULLET) == 26
    assert _full_feedback(json.dumps(_bulleted(LONG_BULLET))) == REJECTION
    validate_schema(AnswerPayload, json.dumps(_bulleted()))  # the twin is valid
    assert gate_problems(AnswerPayload.model_validate(_bulleted()), _seed_index()) == []


@pytest.mark.parametrize("flag", ["0", "1", "grounding"])
async def test_early_abort_off_relays_every_formatter_event_as_today(monkeypatch, flag):
    """Flag off — 0, or anything but `contract`: every chunk of the rejected
    attempt is relayed, the attempt runs to its end, and the feedback is the
    full payload's."""
    bad = json.dumps(_bulleted(LONG_BULLET), indent=2)
    good = json.dumps(_bulleted(), indent=2)

    events, formatter = await _run_gate(monkeypatch, flag, [Streamed(bad), Streamed(good)])

    assert formatter.closed == [None, None]
    relayed = [e for e in events if e.author == "formatter"]
    assert len(relayed) == len(formatter.emitted)
    assert all(a is b for a, b in zip(relayed, formatter.emitted, strict=True))
    first_run = len(Streamed(bad).pieces())
    assert [e.author for e in events] == ["formatter"] * first_run + ["format_gate"] + [
        "formatter"
    ] * (len(formatter.emitted) - first_run)
    assert _feedbacks(events) == [_full_feedback(bad)] == [REJECTION]


async def test_early_abort_never_stops_a_valid_payload(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.agents.format_gate")
    good = json.dumps(_bulleted(), indent=2)
    # control: the same shape with one bullet over budget is stopped
    assert _ContractScan().feed(json.dumps(_bulleted(LONG_BULLET), indent=2)) == REJECTION

    events, formatter = await _run_gate(monkeypatch, "contract", [Streamed(good, size=1)])

    assert formatter.closed == [None]
    assert all(a is b for a, b in zip(events, formatter.emitted, strict=True))
    assert events[-1].actions.state_delta["answer_payload"]["headline"] == "chevy is in good shape"
    assert not [line for line in _gate_lines(caplog) if "early abort" in line]


async def test_early_abort_closes_a_26_word_bullet_before_citations_with_todays_feedback(
    monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="gub_agent.agents.format_gate")
    bad = json.dumps(_bulleted(LONG_BULLET), indent=2)
    good = json.dumps(_bulleted(), indent=2)

    events, formatter = await _run_gate(monkeypatch, "contract", [Streamed(bad), Streamed(good)])

    streamed = formatter.closed[0]
    assert streamed is not None, "attempt 1 ran to its end"
    assert bad.startswith(streamed) and '"citations"' not in streamed
    assert formatter.closed[1] is None
    assert _feedbacks(events) == [_full_feedback(bad)] == [REJECTION]
    # attempt 1's chunks up to the close, the feedback, then attempt 2 — every
    # event the formatter yielded, relayed as the same object, in order
    relayed = [e for e in events if e.author == "formatter"]
    assert all(a is b for a, b in zip(relayed, formatter.emitted, strict=True))
    k = [e.author for e in events].index("format_gate")
    assert all(e.partial for e in events[:k])
    assert "".join(e.content.parts[0].text for e in events[:k]) == streamed
    assert events[k].actions.state_delta == {"format_feedback": REJECTION}
    assert events[-1].actions.state_delta["answer_payload"]["headline"] == "chevy is in good shape"
    lines = _gate_lines(caplog)
    assert (
        f"format_gate: early abort attempt=1 at_chars={len(streamed)} class=contract "
        f"(inv={INV}) tenant=anomaly"
    ) in lines
    assert f"format_gate: attempt 1 rejected (inv={INV}) tenant=anomaly — {REJECTION}" in lines


async def test_early_abort_retries_with_todays_request_and_stores_todays_events(
    monkeypatch, caplog
):
    """The neutrality claim itself: the brief attempt 2 is sent (the
    formatter's whole model input, FORMAT_FEEDBACK included), the events the
    session stores and the `attempt … rejected` line are those of flag off;
    only attempt 1 is closed early."""
    caplog.set_level(logging.INFO, logger="gub_agent.agents.format_gate")
    bad = json.dumps(_bulleted(LONG_BULLET), indent=2)
    good = json.dumps(_bulleted(), indent=2)
    seen = {}
    for flag in ("0", "contract"):
        caplog.clear()
        events, formatter = await _run_gate(monkeypatch, flag, [Streamed(bad), Streamed(good)])
        rejected = [line for line in _gate_lines(caplog) if "rejected" in line]
        seen[flag] = (formatter.briefs, _stored(events), rejected, formatter.closed)

    (briefs_off, stored_off, rejected_off, closed_off) = seen["0"]
    (briefs_on, stored_on, rejected_on, closed_on) = seen["contract"]
    assert briefs_on == briefs_off
    assert f"FORMAT_FEEDBACK (your previous payload was rejected): {REJECTION}" in briefs_on[1]
    assert stored_on == stored_off
    assert rejected_on == rejected_off and len(rejected_on) == 1
    assert closed_off[0] is None and closed_on[0] is not None


async def test_early_abort_a_partial_never_trips_the_payload_contract():
    """`_contract` judges a whole payload: any unfinished one fails it ("needs
    at least one citation", "over the 250 budget"). Only field errors count."""
    valid = json.dumps(_bulleted())
    head = valid[: valid.index(', "citations"')]  # through the close of `blocks`
    with pytest.raises(ValidationError) as caught:
        AnswerPayload.model_validate_json(head + "}")  # the trap
    assert [err["loc"] for err in caught.value.errors()] == [()]
    assert "citation" in caught.value.errors()[0]["msg"]

    scan = _ContractScan()
    assert scan.feed(head) is None
    assert scan.done  # decided at the close of `blocks`: nothing to report
    assert scan.feed(valid[len(head) :]) is None

    # Over the 250-word total on headline + blocks alone: every field is within
    # budget, so the scan must not abort — the full payload's check decides.
    blocks = [{"kind": "bullets", "items": [" ".join(["word"] * 25)] * 7}] * 4
    wordy = json.dumps({"kind": "answer", "headline": "chevy is in good shape", "blocks": blocks})
    as_abstain = wordy.replace('"answer"', '"abstain"', 1)
    with pytest.raises(ValidationError) as caught:
        AnswerPayload.model_validate_json(as_abstain)
    [err] = caught.value.errors()
    assert err["loc"] == () and "705 words total, over the 250 budget" in err["msg"]
    scan = _ContractScan()
    assert scan.feed(wordy[:-1]) is None and scan.done


@pytest.mark.parametrize(
    "order",
    [
        ("kind", "blocks", "headline", "citations", "facts", "follow_ups"),
        ("headline", "blocks", "kind", "citations", "facts", "follow_ups"),
        ("kind", "headline", "follow_ups", "blocks", "citations", "facts"),
    ],
    ids=["blocks-before-headline", "kind-after-blocks", "tail-key-before-blocks"],
)
async def test_early_abort_skips_an_attempt_not_streamed_in_schema_order(monkeypatch, order):
    """`blocks` before `headline` (a partial would report `headline` missing),
    `kind` still to come, or a tail field ahead of `blocks`: the scan cannot
    vouch that the full payload has no other error, so the attempt runs to its
    end and is judged as today."""
    payload = _bulleted(LONG_BULLET)
    bad = json.dumps({key: payload[key] for key in order}, indent=2)
    in_order = json.dumps(payload, indent=2)
    assert _ContractScan().feed(in_order) == REJECTION  # control: schema order aborts

    events, formatter = await _run_gate(monkeypatch, "contract", [Streamed(bad), _bulleted()])

    assert formatter.closed[0] is None
    assert _feedbacks(events) == [_full_feedback(bad)] == [REJECTION]


async def test_early_abort_never_stops_a_fenced_payload(monkeypatch):
    """ADK strips a ```json fence before validating; the scan does not follow
    one, so a fenced attempt is judged whole, as today."""
    bad = json.dumps(_bulleted(LONG_BULLET), indent=2)
    fenced = f"```json\n{bad}\n```"
    assert _ContractScan().feed(bad) == REJECTION  # control: unfenced aborts

    events, formatter = await _run_gate(monkeypatch, "contract", [Streamed(fenced), _bulleted()])

    assert formatter.closed[0] is None
    assert _feedbacks(events) == [_full_feedback(fenced)] == [REJECTION]


async def test_early_abort_follows_chunk_boundaries_inside_escapes_and_keys(monkeypatch):
    """Escaped keys, `\\uXXXX`, an escaped quote and backslash, and a chunk
    boundary at every position — the feedback is always the full payload's."""
    tricky = 'the café push is "live" — back\\slash and'
    long_tricky = tricky + " moving" * (26 - count_words(tricky))
    assert count_words(long_tricky) == 26

    def encoded(first: str) -> str:
        text = json.dumps(_bulleted(first), ensure_ascii=True)
        return text.replace('"blocks":', '"blo\\u0063ks":').replace(
            '"headline":', '"head\\u006cine":'
        )

    bad, good = encoded(long_tricky), encoded("the café push is live")
    assert "\\u00e9" in bad and '\\"live\\"' in bad and "\\\\slash" in bad
    full = _full_feedback(bad)
    assert full == REJECTION
    validate_schema(AnswerPayload, good)

    for text, expected in ((bad, full), (good, None)):
        for cut in range(1, len(text)):
            scan = _ContractScan()
            got = scan.feed(text[:cut]) or scan.feed(text[cut:])
            assert got == expected, f"cut at {cut}: {text[cut - 8 : cut]!r}|{text[cut : cut + 8]!r}"
        scan = _ContractScan()
        assert next((fb for ch in text if (fb := scan.feed(ch))), None) == expected

    # …and through the gate, cut inside the escape of "é", inside an escaped
    # key, between a backslash and the quote it escapes, and where `blocks`
    # closes.
    cuts = sorted(
        {
            bad.index("\\u00e9") + 3,
            bad.index("head\\u006c") + 6,
            bad.index('\\"live') + 1,
            bad.index(', "citations"'),
        }
    )
    chunks = [bad[a:b] for a, b in zip([0, *cuts], [*cuts, len(bad)], strict=True)]
    events, formatter = await _run_gate(
        monkeypatch, "contract", [Streamed(bad, chunks=chunks), _bulleted()]
    )
    assert formatter.closed[0] is not None and '"citations"' not in formatter.closed[0]
    assert _feedbacks(events) == [full]


async def test_early_abort_leaves_grounding_rejections_to_the_full_check(monkeypatch):
    """Contract only: a grounding-rejected attempt is stored before the gate
    judges it, so it always runs to its end."""
    ungrounded = json.dumps(_bulleted("latest from Chevrolet is strong"), indent=2)
    validate_schema(AnswerPayload, ungrounded)  # the contract passes
    scan = _ContractScan()
    assert scan.feed(ungrounded) is None and scan.done  # checked at `blocks`, not stopped

    events, formatter = await _run_gate(
        monkeypatch, "contract", [Streamed(ungrounded), _bulleted()]
    )

    assert formatter.closed[0] is None
    [feedback] = _feedbacks(events)
    assert feedback.startswith("ungrounded entity: Chevrolet")


async def test_early_abort_on_every_attempt_ends_in_todays_template(monkeypatch, caplog):
    """Three aborted attempts end exactly as three rejected ones: two feedback
    events, the `attempts spent` line, the template render."""
    caplog.set_level(logging.INFO, logger="gub_agent.agents.format_gate")
    bad = json.dumps(_bulleted(LONG_BULLET), indent=2)
    seen = {}
    for flag in ("0", "contract"):
        caplog.clear()
        events, formatter = await _run_gate(monkeypatch, flag, [Streamed(bad)])
        seen[flag] = (_stored(events), formatter.closed, _gate_lines(caplog))

    assert seen["contract"][0] == seen["0"][0]
    assert seen["0"][0][-1][0] == "format_gate"  # the template
    assert seen["0"][1] == [None] * 3
    assert all(closed is not None for closed in seen["contract"][1])
    early = [line for line in seen["contract"][2] if "early abort" in line]
    assert [line.split()[3] for line in early] == ["attempt=1", "attempt=2", "attempt=3"]
    assert [line for line in seen["contract"][2] if "early abort" not in line] == seen["0"][2]


# ── the same, through ADK's own LlmAgent (the real formatter) ────────────────


def _request_of(llm_request) -> dict:
    """Everything the model is sent, as data: the contents (the brief, with
    any FORMAT_FEEDBACK), the system instruction and the whole generation
    config — the response schema by name, as it is the class itself."""
    config = llm_request.config
    return {
        "model": llm_request.model,
        "contents": [c.model_dump(mode="json", exclude_none=True) for c in llm_request.contents],
        "config": config.model_dump(mode="json", exclude_none=True, exclude={"response_schema"}),
        "response_schema": getattr(
            config.response_schema, "__name__", repr(config.response_schema)
        ),
    }


class _StreamingModel(BaseLlm):
    """A model that streams its scripted reply the way ADK's Gemini client does
    under SSE — one partial response per chunk, then the whole text — and
    remembers, per call, the request, the text its consumer read (`sent`),
    whether it reached the end, and the text sent when it was closed (None:
    not closed). The last reply repeats."""

    replies: list = []
    size: int = 24
    requests: list = []
    sent: list = []
    finished: list = []
    closed: list = []

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        reply = self.replies[min(len(self.requests), len(self.replies) - 1)]
        self.requests.append(_request_of(llm_request))
        call = len(self.requests) - 1
        self.sent.append("")
        self.finished.append(False)
        self.closed.append(None)
        try:
            if stream:
                for i in range(0, len(reply), self.size):
                    piece = reply[i : i + self.size]
                    self.sent[call] += piece
                    yield LlmResponse(
                        content=genai_types.Content(
                            role="model", parts=[genai_types.Part(text=piece)]
                        ),
                        partial=True,
                    )
            yield LlmResponse(
                content=genai_types.Content(role="model", parts=[genai_types.Part(text=reply)])
            )
            self.finished[call] = True
        except GeneratorExit:
            self.closed[call] = self.sent[call]
            raise


class _Executor(BaseAgent):
    """The deep path's executor, reduced to what the gate reads: this turn's
    evidence and its prose."""

    async def _run_async_impl(self, ctx):
        _seed_index(ctx.invocation_id)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=EXEC_TEXT)]),
        )


async def test_early_abort_through_a_real_formatter_closes_the_model_call(monkeypatch, caplog):
    """The real formatter LlmAgent (its callbacks, output_schema, ADK's flow)
    on an InMemoryRunner with SSE, only the model stubbed: with the flag on the
    model's stream is closed before `citations`, its `model_call` line says
    `status=error:closed` exactly as a contract-rejected call does today, and
    the retry request, the stored session and the stream's non-partial events
    are flag off's."""
    caplog.set_level(logging.INFO)
    bad = json.dumps(_bulleted(LONG_BULLET), indent=2)
    good = json.dumps(_bulleted(), indent=2)
    seen = {}
    for flag in ("0", "contract"):
        monkeypatch.setattr(format_gate_module, "FORMAT_GATE_EARLY_ABORT", flag)
        caplog.clear()
        model = _StreamingModel(
            model="stub", replies=[bad, good], requests=[], sent=[], finished=[], closed=[]
        )
        formatter = formatter_agent.clone(
            update={"model": VendorRouter(model=GEMINI_MODEL, gemini=model)}
        )
        root = SequentialAgent(
            name="root",
            sub_agents=[
                _Executor(name=AGENT_NAME),
                FormatGate(name="format_gate", sub_agents=[formatter]),
            ],
        )
        runner = InMemoryRunner(agent=root, app_name="gub")
        session = await runner.session_service.create_session(app_name="gub", user_id="u")
        streamed = [
            event
            async for event in runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=genai_types.Content(
                    # A lookup, not an open read: "how is chevy?" with no routed
                    # intent is a rundown (BE-5), whose wider budgets would let
                    # LONG_BULLET through and leave nothing to abort.
                    role="user",
                    parts=[genai_types.Part(text="is the Q3 push live?")],
                ),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        ]
        stored = (
            await runner.session_service.get_session(
                app_name="gub", user_id="u", session_id=session.id
            )
        ).events
        calls = [
            r.getMessage().split(" status=")[1].split()[0]
            for r in caplog.records
            if r.getMessage().startswith("model_call: agent=formatter")
        ]
        # VendorRouter iterates the vendor stream without aclosing, so the
        # stream beneath a closed call is closed by asyncio's async-generator
        # finalizer a few loop turns later, not inside the gate's close.
        for _ in range(50):
            if all(c is not None or f for c, f in zip(model.closed, model.finished, strict=True)):
                break
            gc.collect()
            await asyncio.sleep(0)
        seen[flag] = SimpleNamespace(
            sent=model.sent,
            finished=model.finished,
            closed=model.closed,
            calls=calls,
            retry=model.requests[1],
            stored=_stored(stored),
            streamed=_stored(streamed),
            partials=sum(1 for e in streamed if e.partial and e.author == "formatter"),
            feedback=_feedbacks(stored),
        )

    off, on = seen["0"], seen["contract"]
    # today: the whole reply is read, then the raise closes the call at its end
    assert off.sent == [bad, good] and off.closed == [bad, None]
    # flag on: read up to the close of `blocks`, closed there, never finished
    assert bad.startswith(on.sent[0]) and '"citations"' not in on.sent[0]
    assert on.closed == [on.sent[0], None] and on.finished == [False, True]
    assert on.sent[1] == good
    assert off.calls == on.calls == ["error:closed", "ok"]
    assert on.retry == off.retry
    assert REJECTION in json.dumps(on.retry["contents"], ensure_ascii=False)
    assert on.stored == off.stored and on.streamed == off.streamed
    assert on.feedback == off.feedback == [REJECTION]
    assert on.partials < off.partials  # the only difference: fewer attempt-1 chunks


def test_early_abort_a_scan_that_trips_gives_up_instead_of_raising():
    """Anything the scan itself trips on (here a `kind` nested past the
    recursion limit) ends the scan: the attempt runs to its end and the full
    validation judges it, as today — a scan never costs the attempt."""
    deep = "[" * 30000 + "]" * 30000
    scan = _ContractScan()
    assert scan.feed('{"kind":' + deep + ',"headline":"h","blocks":[]}') is None
    assert scan.done
    assert scan.feed(',"citations":[]}') is None


# ── REQUERY_SKIPS_FORMATTER: a from-memory first pass abstains in code ────────
#
# A pass whose executor made no tool call has an empty evidence index, so the
# formatter could only abstain, and the critic gate then sends the executor
# back for one re-query on that abstain. With the switch on the gate writes the
# abstain itself. The cases below are every way the pass can fail to be that
# pass; each must still reach the formatter.

MEMORY_TEXT = "Here is what's new: three hires and a refreshed Q3 plan."
SHORTCUT_LINE = re.compile(
    r"^critic_requery: shortcut reason=no_tool_call pass=1 inv=(\S+) tenant=(\S+)$"
)


def _tool_call_event(invocation_id: str) -> Event:
    return Event(
        invocation_id=invocation_id,
        author=AGENT_NAME,
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(function_call=genai_types.FunctionCall(name="org_query", args={}))
            ],
        ),
    )


def _sent_back_event(invocation_id: str) -> Event:
    verdict = {
        "info_sufficient": False,
        "answer_satisfies": False,
        "sufficient": False,
        "reason": "Deterministic: from memory",
        "feedback": "query GUB",
    }
    return Event(
        invocation_id=invocation_id,
        author="critic",
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=json.dumps(verdict))]
        ),
        actions=EventActions(state_delta={"critic_verdict": verdict}),
    )


def _sandbox_critic_off() -> Event:
    return Event(
        invocation_id=INV,
        author="user",
        actions=EventActions(state_delta={"sandbox": {"critic_enabled": False}}),
    )


# What the gate's own abstain event carries (the NO_COMPANY_RECORDS marker's too).
_ABSTAIN_DUMP = AnswerPayload(kind="abstain", headline="NO_COMPANY_RECORDS").model_dump(
    exclude_none=True
)

# case -> (events appended to the session, flag, a fast-path draft, formatter runs expected)
_MEMORY_CASES = {
    "first pass from memory": ([], True, "", 0),
    "an earlier turn's tool call is not this turn's": (
        [_tool_call_event("inv-earlier")],
        True,
        "",
        0,
    ),
    "an earlier turn's send-back is not this turn's": (
        [_sent_back_event("inv-earlier")],
        True,
        "",
        0,
    ),
    "second pass": ([_sent_back_event(INV)], True, "", 1),
    "a tool call with an empty index": ([_tool_call_event(INV)], True, "", 1),
    "a fast-path draft": ([], True, "the account is live (get_account:a1:status)", 1),
    "the flag at 0": ([], False, "", 1),
    "the sandbox critic off": ([_sandbox_critic_off()], True, "", 1),
}


@pytest.mark.parametrize("case", list(_MEMORY_CASES), ids=list(_MEMORY_CASES))
async def test_only_a_from_memory_first_pass_skips_the_formatter(monkeypatch, case):
    extra, flag, draft, expected_runs = _MEMORY_CASES[case]
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", flag)
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    reset_evidence_index(SimpleNamespace(invocation_id=INV))  # no evidence at all
    if draft:
        set_answer_draft(INV, draft)
    formatter = _scripted([{"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx(MEMORY_TEXT)
    for event in extra:
        await ctx.session_service.append_event(ctx.session, event)

    try:
        events = [e async for e in gate.run_async(ctx)]
    finally:
        reset_evidence_index(SimpleNamespace(invocation_id=INV))  # the draft too

    assert len(formatter.runs) == expected_runs, case
    # either way the pass ends with the abstain payload, from one event
    # (the formatter's own, or the gate's) — the critic gate reads this
    payloads = [e.actions.state_delta["answer_payload"] for e in events if e.actions.state_delta]
    if expected_runs:  # the stand-in formatter's, as scripted
        assert [p["kind"] for p in payloads] == ["abstain"]
    else:  # the gate's own
        assert payloads == [_ABSTAIN_DUMP]


async def test_the_shortcut_abstain_is_the_gates_own_event_and_logs_two_lines(monkeypatch, caplog):
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    reset_evidence_index(SimpleNamespace(invocation_id=INV))
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx(MEMORY_TEXT)

    with caplog.at_level(logging.INFO):
        events = [e async for e in gate.run_async(ctx)]

    assert formatter.runs == []  # the stub is never called
    assert [e.author for e in events] == ["format_gate"]
    abstain = events[0].actions.state_delta["answer_payload"]
    assert abstain == _ABSTAIN_DUMP
    assert json.loads(events[0].content.parts[0].text) == abstain

    shortcut = [m for m in caplog.messages if m.startswith("critic_requery:")]
    assert len(shortcut) == 1
    match = SHORTCUT_LINE.match(shortcut[0])
    assert match, shortcut[0]
    assert match.groups() == (INV, "anomaly")
    # not a gate line: rate queries match `textPayload:"format_gate"`
    assert "format_gate" not in shortcut[0]
    assert [r.name for r in caplog.records if r.getMessage() == shortcut[0]] == [
        "gub_agent.agents.critic"
    ]
    # one `answer_shape` line, as the formatter's abstain would have left
    shapes = [m for m in caplog.messages if m.startswith("answer_shape:")]
    assert len(shapes) == 1
    assert shapes[0].startswith("answer_shape: kind=abstain ")
    assert f"exec_chars={len(MEMORY_TEXT)} " in shapes[0]
    assert shapes[0].endswith(f"inv={INV} tenant=anomaly")


async def test_evidence_rows_keep_the_formatter_even_without_a_tool_call_event(monkeypatch):
    """The gate's own guard on top of the predicate: rows in the index mean
    something was retrieved, whatever the session's events say."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")  # no function_call event

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 1
    assert [e.author for e in events] == ["formatter"]


async def test_a_clarify_after_a_tool_call_with_nothing_indexed_passes_through(monkeypatch):
    """The shortcut is for a pass that made no call. One that called a tool and
    got nothing indexed is the formatter's to judge (abstain, clarify), with
    the flag at its deployed value."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    reset_evidence_index(SimpleNamespace(invocation_id=INV))
    clarify = {
        "kind": "clarify",
        "headline": "Which campaign do you mean?",
        "candidates": [
            {"name": "Holiday Event", "entity_type": "campaign"},
            {"name": "Spring Push", "entity_type": "campaign"},
        ],
    }
    formatter = _scripted([clarify])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("Two campaigns match; which one?")
    await ctx.session_service.append_event(ctx.session, _tool_call_event(INV))

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 1
    wire = [json.loads(e.content.parts[0].text) for e in events if e.content]
    assert wire[-1]["kind"] == "clarify" and len(wire[-1]["candidates"]) == 2


async def test_a_pass_whose_executor_wrote_nothing_is_not_a_from_memory_pass(monkeypatch):
    """The gate emits nothing for an empty draft, so the speculative critic must
    not be skipped for it: with no payload the resolver would find nothing
    held and run the critic serially, after the join."""
    from gub_agent.agents.critic import from_memory_pass

    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    assert from_memory_pass(await _gate_ctx(MEMORY_TEXT)) is True
    assert from_memory_pass(await _gate_ctx("")) is False  # an event with no text
    assert from_memory_pass(await _gate_ctx("   \n")) is False
    assert from_memory_pass(await _gate_ctx(None)) is False  # no executor event at all
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", False)
    assert from_memory_pass(await _gate_ctx(MEMORY_TEXT)) is False  # the flag is the first term


async def test_the_shortcuts_abstain_carries_the_gates_branch(monkeypatch):
    """ADK hides a branch-stamped event from a sibling branch (the speculative
    critic) and shows an unstamped one to every agent. The formatter's own
    abstain carries the gate's branch, so the gate's must too."""
    monkeypatch.setattr(config, "REQUERY_SKIPS_FORMATTER", True)
    reset_evidence_index(SimpleNamespace(invocation_id=INV))
    gate = FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])])
    ctx = await _gate_ctx(MEMORY_TEXT)
    ctx = ctx.model_copy(update={"branch": "format_and_critic.format_gate"})

    events = [e async for e in gate.run_async(ctx)]

    assert [e.author for e in events] == ["format_gate"]
    assert events[0].branch == "format_and_critic.format_gate"

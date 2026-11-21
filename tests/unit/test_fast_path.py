"""
agents.fast_path — the deterministic lookup (blend 04).

The pure halves (`resolve_hits`, `org_query_args`, `period_range`,
`compose_draft`) are tested directly. The agent itself runs on a real ADK
InvocationContext with `gub_get` / `gub_post` mocked at the tool modules'
import sites — the same functions the deep path calls, which is the point of
reusing them — and with the format gate replaced by a scripted stand-in
wherever the test is not about the gate itself.
"""

from __future__ import annotations

import itertools
import json
import logging
from importlib import import_module

import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from gub_agent.agents import dispatcher as dp
from gub_agent.agents import fast_path as fp
from gub_agent.agents.answers import not_found_payload
from gub_agent.agents.evidence_index import answer_draft, evidence_index
from gub_agent.agents.format_gate import FormatGate, gate_problems
from gub_agent.agents.formatter import ANSWER_STATE_KEY
from gub_agent.schemas import RouterDecision
from tests.helpers import invocation_ctx

from .test_router import CAMPAIGN_ID, PIECE_ID, _guess, _selected

INV = "inv-fast"


# ── doubles ───────────────────────────────────────────────────────────────────


class FakeGub:
    """(method, path) → payload, plus a call log. Registered paths match by
    prefix so `/org/campaigns/<uuid>` needs no uuid in the test."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, str, dict]] = []
        self.calls: list[tuple[str, str]] = []

    def on(self, method: str, path: str, payload: dict) -> FakeGub:
        self.routes.append((method, path, payload))
        return self

    def _answer(self, method: str, path: str) -> dict:
        self.calls.append((method, path))
        for m, prefix, payload in self.routes:
            if m == method and path.startswith(prefix):
                return payload
        return {"error": True, "status": 404, "message": "not found"}

    async def get(self, path: str, tool_context=None, **params) -> dict:
        return self._answer("GET", path)

    async def post(self, path: str, body: dict, tool_context=None) -> dict:
        self.calls.append(("BODY", json.dumps(body, sort_keys=True, ensure_ascii=False)))
        return self._answer("POST", path)


@pytest.fixture
def gub(monkeypatch) -> FakeGub:
    """Patch the tool modules' own `gub_get` / `gub_post` bindings — they
    imported the names, so patching `_client` would not be seen.

    Addressed through `import_module`, not a dotted string:
    `gub_agent.tools.org_query` as an ATTRIBUTE is the function re-exported by
    `tools/__init__.py`, which shadows the module of the same name.
    """
    fake = FakeGub()
    for name in ("accounts", "discovery", "staff"):
        monkeypatch.setattr(import_module(f"gub_agent.tools.{name}"), "gub_get", fake.get)
    monkeypatch.setattr(import_module("gub_agent.tools.org_query"), "gub_post", fake.post)
    return fake


class ScriptedGate(BaseAgent):
    """Stand-in for the format gate: records that it ran and emits one payload
    event, the way the real gate's deterministic branch does."""

    runs: list = []

    async def _run_async_impl(self, ctx):
        self.runs.append(ctx.invocation_id)
        payload = {"kind": "answer", "headline": "scripted", "citations": [], "facts": []}
        yield Event(
            invocation_id=ctx.invocation_id,
            author="format_gate",
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(payload))]
            ),
            actions=EventActions(state_delta={ANSWER_STATE_KEY: payload}),
        )


@pytest.fixture
def gate(monkeypatch) -> ScriptedGate:
    scripted = ScriptedGate(name="format_gate", runs=[])
    monkeypatch.setattr(fp, "format_gate", scripted)
    return scripted


def _decision(**over) -> RouterDecision:
    base = {
        "intent": "campaign_status",
        "confidence": 0.93,
        "entity_surface": "Silverado 2026 Q3",
        "language": "ru",
    }
    base.update(over)
    return RouterDecision.model_validate(base)


async def _ctx(decision: RouterDecision | None = None, question: str = "статус Silverado 2026 Q3"):
    """A context carrying the router's decision in state and a JWT in the
    session, so no token exchange is attempted.

    `question` matters for `count_or_rank`: the builder reads the user's own
    words to tell a count from a list or a sum (`_asked_shape`).
    """
    state = {"gub_jwt": "test-jwt"}
    if decision is not None:
        state["router_decision"] = decision.model_dump()
    return await invocation_ctx(state=state, invocation_id=INV, user_text=question)


async def _bot_ctx(
    kind: str,
    entity_id: str,
    question: str = "статус Silverado 2026 Q3",
    *,
    guess: bool = False,
    name: str = "Silverado 2026 Q3",
    **over,
):
    """A turn the bot resolved: its prefix in front of `question` — the bot's
    own wording, so `entity_type` comes out of the same parse production runs —
    and a router decision that copied the id out of it."""
    prefix = (_guess if guess else _selected)(kind, entity_id, name, question)
    return await _ctx(_decision(entity_id=entity_id, **over), prefix)


def _hit(**over) -> dict:
    base = {"type": "campaign", "id": "c1", "name": "Silverado 2026 Q3", "similarity": 0.9}
    base.update(over)
    return base


CAMPAIGN = {
    "id": "c1",
    "name": "Silverado 2026 Q3",
    "status": "live",
    "budget": 1200000,
    "accountName": "chevy",
}


# ── resolve_hits: blend 02's rules ────────────────────────────────────────────


def test_a_single_typed_hit_resolves():
    r = fp.resolve_hits([_hit()], "Silverado 2026 Q3", ("campaign",))
    assert (r.kind, r.entity_id) == ("one", "c1")


def test_an_exact_name_wins_over_a_narrow_margin():
    hits = [
        _hit(id="a", name="Silverado", similarity=0.9),
        _hit(id="b", name="Silverado 2026 Q3", similarity=0.89),
    ]
    r = fp.resolve_hits(hits, "Silverado 2026 Q3", ("campaign",))
    assert (r.kind, r.entity_id) == ("one", "b")


def test_two_hits_with_the_same_exact_name_are_ambiguous():
    hits = [_hit(id="a", name="Silverado", similarity=1.0), _hit(id="b", name="Silverado")]
    assert fp.resolve_hits(hits, "silverado", ("campaign",)).kind == "ambiguous"


def test_hits_within_the_tie_band_are_ambiguous():
    hits = [
        _hit(id="a", name="EV One", similarity=0.62),
        _hit(id="b", name="EV Two", similarity=0.60),
    ]
    assert fp.resolve_hits(hits, "EV", ("campaign",)).kind == "ambiguous"


def test_hits_outside_the_tie_band_resolve_to_the_top_hit():
    hits = [
        _hit(id="a", name="EV Everywhere", similarity=0.7),
        _hit(id="b", name="Evergreen", similarity=0.3),
    ]
    r = fp.resolve_hits(hits, "EV", ("campaign",))
    assert (r.kind, r.entity_id) == ("one", "a")


def test_candidates_all_scoring_1_0_are_ambiguous_not_confident():
    """The regression clarify-01 was written for. `/org/search` scores by WORD
    match, so every campaign whose name contains "Silverado" comes back at 1.0.
    The old rule read that as a near-exact top hit and committed to one of
    them; a tie band has to call it what it is."""
    hits = [
        _hit(id="a", name="Chevrolet | Fall 2024 Silverado + Equinox", similarity=1.0),
        _hit(id="b", name="Chevrolet | Silverado Post-Produced Retail", similarity=1.0),
        _hit(id="c", name="Silverado EV // Oregon Triple", similarity=1.0),
    ]
    assert fp.resolve_hits(hits, "silverado", ("campaign",)).kind == "ambiguous"


def test_hits_of_the_wrong_type_do_not_count():
    hits = [_hit(type="idea", id="i1", similarity=0.99), _hit(type="staff", id="s1")]
    assert fp.resolve_hits(hits, "Silverado", ("campaign",)).kind == "none"


async def test_resolve_reads_the_bare_array_gub_actually_returns(monkeypatch):
    """`GET /org/search` answers with a bare array, not `{"hits": [...]}`. Every
    other test here stubs the dict shape, which is why `_resolve` could sit in
    production returning "none" for every surface without one of them failing."""
    calls: list[str] = []

    async def fake_find(query, tool_context=None):
        calls.append(query)
        return [_hit(id="only", name="Daytona", similarity=1.0)]

    monkeypatch.setattr(fp, "find", fake_find)
    r = await fp._resolve("Daytona", ("campaign",), None)
    assert calls == ["Daytona"]
    assert (r.kind, r.entity_id) == ("one", "only")


async def test_resolve_still_reads_the_error_envelope(monkeypatch):
    async def fake_find(query, tool_context=None):
        return {"error": True, "status": 403, "message": "nope"}

    monkeypatch.setattr(fp, "find", fake_find)
    assert (await fp._resolve("Daytona", ("campaign",), None)).kind == "none"


# ── the org_query builder ─────────────────────────────────────────────────────


def test_period_range_takes_explicit_years_and_quarters_only():
    assert fp.period_range("2026") == ["2026-01-01", "2026-12-31"]
    assert fp.period_range("2026 Q3") == ["2026-07-01", "2026-09-30"]
    assert fp.period_range("Q1 2025") == ["2025-01-01", "2025-03-31"]
    assert fp.period_range("this year") is None  # a judgement, not a regex


async def test_a_count_becomes_an_aggregate_query(gub):
    args = await fp.org_query_args(
        _decision(
            intent="count_or_rank",
            slots={"entity": "campaigns", "status": "live", "complete": True},
        ),
        fp._ToolShim(await _ctx()),
        "how many campaigns are live?",
    )
    assert args == {
        "entity": "campaigns",
        "filter": {"status": {"eq": "live"}},
        "aggregate": {"count": {"op": "count"}},
    }


async def test_a_metric_becomes_a_ranking_with_a_default_limit(gub):
    args = await fp.org_query_args(
        _decision(
            intent="count_or_rank",
            slots={"entity": "campaigns", "metric": "budget", "complete": True},
        ),
        fp._ToolShim(await _ctx()),
        "which campaign has the largest budget?",
    )
    assert args["sort"] == [{"field": "budget", "direction": "desc"}]
    assert args["limit"] == 5
    assert "aggregate" not in args
    # NULL budgets sort FIRST, so a ranking that does not exclude them returns
    # a row with no budget as the "largest" — live regression `fact-15`.
    assert args["filter"]["budget"] == {"is_null": False}


async def test_a_named_account_is_resolved_to_an_id_never_similar_to(gub):
    gub.on("GET", "/org/search", {"hits": [_hit(type="account", id="a1", name="chevy")]})
    args = await fp.org_query_args(
        _decision(
            intent="count_or_rank",
            slots={
                "entity": "campaigns",
                "status": "live",
                "account": "chevy",
                "complete": True,
            },
        ),
        fp._ToolShim(await _ctx()),
        "how many live campaigns does chevy have?",
    )
    # id filter, not `similar_to` — which v1 requires to be a call's SOLE
    # filter and could therefore never combine with the status above.
    assert args["filter"] == {"status": {"eq": "live"}, "accountId": {"eq": "a1"}}
    assert "similar_to" not in json.dumps(args)


@pytest.mark.parametrize(
    "slots",
    [
        # The router itself says the fields do not cover the sentence.
        {"entity": "campaigns", "status": "live", "complete": False},
        # An office needs an id this path cannot resolve deterministically.
        {"entity": "staff", "office": "London", "complete": True},
        {"entity": "campaigns", "status": "active", "complete": True},  # another entity's status
        {"entity": "campaigns", "period": "недавно", "complete": True},  # a relative period
        {"entity": "accounts", "period": "2026", "complete": True},  # no date field for accounts
        {"status": "live", "complete": True},  # no entity named
        {"entity": "campaigns", "metric": "engagement", "complete": True},  # no such field
        {"entity": "campaigns", "limit": 500, "complete": True},  # out of range
        {"entity": "campaigns", "group_by": "region", "complete": True},  # not groupable
        {"entity": "staff", "industry": "auto", "complete": True},  # industry is accounts-only
    ],
)
async def test_slots_that_do_not_assemble_deterministically_go_deep(gub, slots):
    args = await fp.org_query_args(
        _decision(intent="count_or_rank", slots=slots),
        fp._ToolShim(await _ctx()),
        "how many campaigns are live?",
    )
    assert args is None


def test_asked_shape_separates_the_eight_cells_the_eval_measured():
    """The blend-06 eval (2026-09-10) found the fast path served eight FACT
    cells and got four wrong — all `count_or_rank`, and the split is exactly
    by what the sentence asked for. `Slots` cannot see that difference, so the
    builder reads the user's words. This table IS the regression.
    """
    served_correctly = [
        "How many client accounts do we have?",  # fact-02
        "How many campaigns are currently live?",  # fact-04
        "How many active staff members do we have?",  # fact-08
        "How many campaigns are in pitch status?",  # fact-20
    ]
    answered_wrongly = [
        "Which client accounts do we have?",  # fact-01 — got a count
        "Which campaigns are in the awarded status?",  # fact-05 — got a count
        "What is the combined budget of all our campaigns?",  # fact-16 — got a ranking
    ]
    for question in served_correctly:
        assert fp._asked_shape(question) == "count", question
    for question in answered_wrongly:
        # "other" means the deep path, which answered 11 of 11 FACT cells right
        assert fp._asked_shape(question) == "other", question
    # fact-15 stays on the fast path — it IS a ranking — and is correct now
    # only because the builder excludes NULLs (tested above).
    assert fp._asked_shape("Which campaign has the largest budget?") == "rank"


def test_asked_shape_prefers_the_disqualifying_reading():
    """An aggregate or an enumeration wins over counting words in the same
    sentence — the builder can serve neither."""
    assert fp._asked_shape("list how many we have per office") == "other"
    assert fp._asked_shape("what is the total number of campaigns") == "other"
    assert fp._asked_shape("сколько кампаний live?") == "count"
    assert fp._asked_shape("какие кампании live?") == "other"
    assert fp._asked_shape("суммарный бюджет кампаний") == "other"
    assert fp._asked_shape("самая дорогая кампания") == "rank"
    # Nothing recognisable is a deep-path question, never a guessed count.
    assert fp._asked_shape("chevy campaigns") == "other"


# ── the agent ─────────────────────────────────────────────────────────────────


def _answers(events: list) -> list:
    """Everything the turn actually SAYS, i.e. the progress pings removed.

    The fast path emits a function_call/function_response pair around its
    lookup so the bot's bubble can advance its step line — its tool runs as a
    plain coroutine, so nothing else would announce it. Those events are
    `partial`, which is what keeps them out of the session history, and it is
    also what makes them easy to drop here: an answer is never partial.
    """
    return [e for e in events if not e.partial]


def _progress(events: list) -> list[tuple[str, bool]]:
    """(tool name, is_result) for each progress ping, in order."""
    out = []
    for e in events:
        if not e.partial or not e.content:
            continue
        for part in e.content.parts or []:
            if part.function_call:
                out.append((part.function_call.name, False))
            if part.function_response:
                out.append((part.function_response.name, True))
    return out


@pytest.mark.parametrize("guess", [False, True], ids=["selected", "best-guess"])
async def test_a_bot_supplied_campaign_id_skips_the_search_entirely(gub, gate, guess):
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _bot_ctx("campaign", "c1", guess=guess)

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert [m for m, _ in gub.calls] == ["GET"]  # no /org/search round trip
    assert gub.calls[0][1] == "/org/campaigns/c1"
    assert fp.outcome(INV) == "answered"
    assert gate.runs == [INV]
    assert len(_answers(events)) == 1


async def test_the_evidence_ids_are_indistinguishable_from_the_deep_paths(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _ctx(_decision())

    [e async for e in fp.fast_path.run_async(ctx)]

    index = evidence_index(INV)
    assert "get_campaign:c1" in index  # <tool>:<row id>, exactly as after a tool call
    assert index["get_campaign:c1:status"]["value"] == "live"
    # The resolution step is internal — the search result is not offered as
    # citable evidence.
    assert not any(key.startswith("find:") for key in index)


async def test_the_draft_carries_the_question_and_verbatim_values(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _ctx(_decision())

    [e async for e in fp.fast_path.run_async(ctx)]

    draft = answer_draft(INV)
    assert "статус Silverado 2026 Q3" in draft
    assert "status: live" in draft
    assert "budget: 1200000" in draft  # never reformatted — the gate grounds it


async def test_an_ambiguous_surface_goes_deep_with_no_card_and_no_detail_call(gub, gate):
    gub.on(
        "GET",
        "/org/search",
        {
            "hits": [
                _hit(id="a", name="EV One", similarity=0.62),
                _hit(id="b", name="EV Two", similarity=0.60),
            ]
        },
    )
    ctx = await _ctx(_decision(entity_surface="EV"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert _answers(events) == []
    assert fp.outcome(INV) == "deep"
    assert gate.runs == []
    assert [path for _, path in gub.calls] == ["/org/search"]


async def test_an_empty_detail_result_falls_through_to_the_deep_path(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", {})
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert _answers(events) == []
    assert fp.outcome(INV) == "deep"


async def test_a_403_is_answered_immediately_without_the_gate(gub, gate):
    # The id came from this path's own search, so a refusal about it IS the
    # answer to what was asked. (A bot-supplied id is another case, below.)
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on(
        "GET", "/org/campaigns/", {"error": True, "status": 403, "message": "denied"}
    )
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert fp.outcome(INV) == "answered"
    assert gate.runs == []  # no formatter LLM to hear the same 403
    assert len(_answers(events)) == 1
    payload = _answers(events)[0].actions.state_delta[ANSWER_STATE_KEY]
    assert payload["kind"] == "answer"
    assert "Нет доступа" in payload["headline"]
    assert _answers(events)[0].author == "format_gate"  # the bot's answer channel


async def test_a_404_names_what_was_looked_for_in_the_assumption_line(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit(id="c9")]}).on(
        "GET", "/org/campaigns/", {"error": True, "status": 404, "message": "nope"}
    )
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    payload = _answers(events)[0].actions.state_delta[ANSWER_STATE_KEY]
    assert payload["assumptions"] == ["Искал: Silverado 2026 Q3"]
    # The surface rides in `assumptions` precisely so the gate's grounding
    # checks pass against an EMPTY index — the headline names no entity.
    assert gate_problems(not_found_payload("ru", "Silverado 2026 Q3"), {}) == []


# ── a bot-supplied id (BE-6) ──────────────────────────────────────────────────
#
# 2026-10-08 18:32: the reader picked a PIECE from a card; the router copied its
# id into `entity_id` and called the intent campaign_facts; the fast path asked
# `/org/campaigns/<piece id>`, got a 404 for a record that exists, and answered
# "not found" — a shortcut taken on a question nobody had asked. The id is now
# used only when the bot's own prefix says it is a campaign, and a refusal about
# an id the bot handed over goes to the deep path (once) instead of being the
# reader's answer.

PIECE_PREFIXED = "статус ролика Hero Spot 30s"  # the reader's own words, Russian


async def _piece_turn(**over):
    """The 18:32 turn: a piece picked from a card, routed as a campaign intent."""
    return await _bot_ctx(
        "piece",
        PIECE_ID,
        PIECE_PREFIXED,
        name="Hero Spot 30s",
        intent="campaign_facts",
        entity_surface="Hero Spot 30s",
        **over,
    )


async def test_a_piece_id_is_never_sent_to_the_campaigns_endpoint(gub, gate):
    """The 10-08 18:32 case. With GUB answering exactly as it did that evening
    (404 on `/org/campaigns/<id>`), the old path ANSWERED "not found". Now the
    fast path makes no call at all — not even the surface search, which could
    settle on a different campaign than the piece the reader chose — and the
    turn is the deep path's."""
    gub.on("GET", "/org/campaigns/", {"error": True, "status": 404, "message": "not found"})
    gub.on("GET", "/org/search", {"hits": [_hit(name="Hero Spot 30s")]})  # a tempting campaign
    ctx = await _piece_turn()

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert gub.calls == []  # neither /org/campaigns/<piece id> nor /org/search
    assert _answers(events) == []  # no "not found" for the reader
    assert fp.outcome(INV) == "deep" and gate.runs == []


async def test_the_piece_turn_is_answered_by_the_deep_path_through_the_dispatcher(
    gub, gate, monkeypatch
):
    """…and the turn does get answered: through the dispatcher, the deep agent
    runs exactly once and its payload is the only one the reader receives."""
    deep_runs = []

    class Deep(BaseAgent):
        async def _run_async_impl(self, ctx):
            deep_runs.append(ctx.invocation_id)
            payload = {"kind": "answer", "headline": "deep", "citations": [], "facts": []}
            yield Event(
                invocation_id=ctx.invocation_id,
                author="format_gate",
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=json.dumps(payload))]
                ),
                actions=EventActions(state_delta={ANSWER_STATE_KEY: payload}),
            )

    gub.on("GET", "/org/campaigns/", {"error": True, "status": 404, "message": "not found"})
    fast = fp.FastPath(name="fast_path")  # the singleton already has its parent
    monkeypatch.setattr(dp, "fast_path", fast)
    agent = dp.Dispatcher(name="dispatcher", sub_agents=[fast, Deep(name="gub_pipeline")])
    ctx = await _piece_turn(confidence=0.95)

    events = [e async for e in agent.run_async(ctx)]

    assert deep_runs == [INV]
    assert gub.calls == []
    payloads = [e.actions.state_delta[ANSWER_STATE_KEY] for e in _answers(events)]
    assert [p["headline"] for p in payloads] == ["deep"]


@pytest.mark.parametrize("kind", ["account", "piece", "staff"])
@pytest.mark.parametrize("intent", ["campaign_status", "campaign_facts"])
async def test_only_a_campaign_id_is_used_as_one(gub, gate, caplog, kind, intent):
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _bot_ctx(kind, "x-1", intent=intent)

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        events = [e async for e in fp.fast_path.run_async(ctx)]

    assert gub.calls == []
    assert _answers(events) == [] and fp.outcome(INV) == "deep"
    [line] = [r.getMessage() for r in caplog.records if "is not a campaign" in r.getMessage()]
    assert line == (
        f"fast_path: bot-supplied id is not a campaign entity_type={kind} intent={intent}"
        f" — deep path (inv={INV}) tenant=anomaly"
    )


async def test_an_id_the_prefix_does_not_vouch_for_is_not_used(gub, gate):
    """The router "copied" an id that is not the prefix's (or there is no prefix
    to copy it from). Whatever it is, it is not shown to be a campaign."""
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    other = _decision(entity_id="c1")  # prefix names a different id
    mangled = await _ctx(other, _selected("campaign", "c2", "Silverado", "статус?"))
    invented = await _ctx(_decision(entity_id="c1"), "статус Silverado 2026 Q3")

    for ctx in (mangled, invented):
        events = [e async for e in fp.fast_path.run_async(ctx)]
        assert _answers(events) == [] and fp.outcome(INV) == "deep"
    assert gub.calls == []


@pytest.mark.parametrize("status", [403, 404])
@pytest.mark.parametrize("guess", [False, True], ids=["selected", "best-guess"])
async def test_a_refusal_about_a_bot_supplied_id_falls_through_to_the_deep_path_once(
    gub, gate, caplog, status, guess
):
    gub.on("GET", "/org/campaigns/", {"error": True, "status": status, "message": "refused"})
    ctx = await _bot_ctx("campaign", "c1", guess=guess)

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        events = [e async for e in fp.fast_path.run_async(ctx)]

    assert gub.calls == [("GET", "/org/campaigns/c1")]  # asked once: no retry, no search
    assert _answers(events) == []  # no "not found" / "no access" on the reader's behalf
    assert fp.outcome(INV) == "deep" and gate.runs == []
    messages = [r.getMessage() for r in caplog.records]
    assert not any(m.startswith("fast_path: shortcut") for m in messages)  # nothing was answered
    assert f"fast_path: bot-supplied id refused status={status} entity_type=campaign" in (
        next(m for m in messages if "bot-supplied id refused" in m)
    )


async def test_the_deep_path_runs_exactly_once_after_a_refused_bot_id(gub, gate, monkeypatch):
    deep_runs = []

    class Deep(BaseAgent):
        async def _run_async_impl(self, ctx):
            deep_runs.append(ctx.invocation_id)
            return
            yield  # pragma: no cover — an async generator that says nothing

    gub.on("GET", "/org/campaigns/", {"error": True, "status": 404, "message": "gone"})
    fast = fp.FastPath(name="fast_path")
    monkeypatch.setattr(dp, "fast_path", fast)
    agent = dp.Dispatcher(name="dispatcher", sub_agents=[fast, Deep(name="gub_pipeline")])
    ctx = await _bot_ctx("campaign", "c1", confidence=0.95)

    [e async for e in agent.run_async(ctx)]

    assert deep_runs == [INV]
    assert gub.calls == [("GET", "/org/campaigns/c1")]


async def test_a_refusal_about_an_id_this_path_found_is_still_answered_on_the_spot(
    gub, gate, caplog
):
    """The fall-through is for ids the bot HANDED OVER. An id the fast path
    resolved itself, from the reader's own words, keeps the shortcut — and its
    line, byte for byte."""
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on(
        "GET", "/org/campaigns/", {"error": True, "status": 404, "message": "nope"}
    )
    ctx = await _ctx(_decision())

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        events = [e async for e in fp.fast_path.run_async(ctx)]

    assert len(_answers(events)) == 1 and fp.outcome(INV) == "answered"
    assert [r.getMessage() for r in caplog.records if "shortcut" in r.getMessage()] == [
        f"fast_path: shortcut status=404 intent=campaign_status inv={INV} tenant=anomaly"
    ]
    assert not any("bot-supplied" in r.getMessage() for r in caplog.records)


async def test_a_prefix_less_turn_resolves_exactly_as_before(gub, gate):
    """No prefix, no id: the surface goes through `/org/search` and the hit's id
    is fetched — the path every turn without a bot-resolved entity takes."""
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _ctx(_decision(), "статус Silverado 2026 Q3")

    [e async for e in fp.fast_path.run_async(ctx)]

    assert gub.calls == [("GET", "/org/search"), ("GET", "/org/campaigns/c1")]
    assert fp.outcome(INV) == "answered"


async def test_the_bots_literal_prefix_reaches_the_lookup_end_to_end(gub, gate):
    """The wire, start to finish, with the bot's own Russian original question:
    its prefix → the parsed type → the one `/org/campaigns/<uuid>` call."""
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    text = _guess("campaign", CAMPAIGN_ID, "Silverado", "какой статус у кампании?", "Chevy")
    ctx = await _ctx(_decision(entity_id=CAMPAIGN_ID), text)

    [e async for e in fp.fast_path.run_async(ctx)]

    assert gub.calls == [("GET", f"/org/campaigns/{CAMPAIGN_ID}")]
    assert fp.outcome(INV) == "answered"


async def test_ids_of_other_types_do_not_touch_the_account_and_staff_lookups(gub, gate):
    """BE-6 changes the campaign lookup only: the account and staff lookups
    never read `entity_id`, with or without a type."""
    gub.on("GET", "/org/search", {"hits": [_hit(type="account", id="a1", name="Chevy")]})
    gub.on("GET", "/org/accounts/", {"id": "a1", "name": "Chevy", "status": "active"})
    ctx = await _bot_ctx(
        "account",
        "a1",
        "как дела у Chevy?",
        name="Chevy",
        intent="account_facts",
        entity_surface="Chevy",
    )

    [e async for e in fp.fast_path.run_async(ctx)]

    assert [path for _, path in gub.calls][0] == "/org/search"  # resolved by name, as before


async def test_a_500_falls_through_to_the_deep_path(gub, gate):
    gub.on("GET", "/org/campaigns/", {"error": True, "status": 500, "message": "boom"})
    ctx = await _bot_ctx("campaign", "c1")

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert _answers(events) == []
    assert fp.outcome(INV) == "deep"


async def test_an_unexpected_exception_costs_the_deep_path_not_the_turn(monkeypatch, gate):
    async def boom(*a, **kw):
        raise RuntimeError("httpx exploded")

    monkeypatch.setattr(import_module("gub_agent.tools.accounts"), "gub_get", boom)
    ctx = await _bot_ctx("campaign", "c1")

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert _answers(events) == []
    assert fp.outcome(INV) == "deep"


async def test_staff_falls_back_to_search_staff_when_the_cross_search_misses(gub, gate):
    gub.on("GET", "/org/search", {"hits": []}).on(
        "GET", "/org/staff", {"staff": [{"id": "s1", "name": "Alex Kim", "title": "strategist"}]}
    )
    ctx = await _ctx(_decision(intent="staff_lookup", entity_surface="Alex"))

    [e async for e in fp.fast_path.run_async(ctx)]

    assert fp.outcome(INV) == "answered"
    assert "search_staff:s1" in evidence_index(INV)


async def test_a_count_answers_from_total_not_from_row_arithmetic(gub, gate):
    gub.on("POST", "/org/query", {"results": [{"count": 7}], "total": 1})
    ctx = await _ctx(
        _decision(
            intent="count_or_rank",
            entity_surface=None,
            slots={"entity": "campaigns", "status": "live", "complete": True},
        ),
        question="сколько кампаний сейчас live?",
    )

    [e async for e in fp.fast_path.run_async(ctx)]

    assert fp.outcome(INV) == "answered"
    assert "org_query:results0:count" in evidence_index(INV)


# ── the invariant: nothing ships without the gate ─────────────────────────────


class ScriptedFormatter(BaseAgent):
    """The formatter LLM, scripted — emits whatever payload the test gives it
    the way `output_schema` does (JSON text + state_delta)."""

    payload: dict = {}
    runs: list = []

    async def _run_async_impl(self, ctx):
        self.runs.append(ctx.invocation_id)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(self.payload))]
            ),
            actions=EventActions(state_delta={ANSWER_STATE_KEY: self.payload}),
        )


async def test_the_real_gate_renders_the_fast_path_draft_and_grounds_it(gub, monkeypatch):
    """The fast path's payload goes through the SAME gate as the deep path's:
    the gate reads the draft (there is no executor text to read) and its
    grounding checks run against the fast path's evidence index."""
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    formatter = ScriptedFormatter(
        name="formatter",
        payload={
            "kind": "answer",
            "headline": "Silverado 2026 Q3 — live",
            "blocks": [{"kind": "text", "text": "budget 1200000."}],
            "citations": ["get_campaign:c1:status"],
            "facts": [
                {
                    "evidence_id": "get_campaign:c1:status",
                    "entity_id": "c1",
                    "field": "status",
                    "value": "live",
                }
            ],
        },
        runs=[],
    )
    monkeypatch.setattr(fp, "format_gate", FormatGate(name="format_gate", sub_agents=[formatter]))
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert formatter.runs == [INV]  # one formatter pass, no retry: it grounded
    assert [e.author for e in _answers(events)] == ["formatter"]
    assert fp.outcome(INV) == "answered"


# ── the progress pings ────────────────────────────────────────────────────────
#
# The bot draws its "Querying org records…" line from tool activity in the
# event stream. The deep path gets that for free — its calls go through a
# model, which emits function_call parts. The fast path calls its tool as a
# plain coroutine, so it emitted nothing at all and the bubble sat on one
# generic phrase for the whole turn, which reads as a hang (reported
# 2026-09-16: "streaming output does not work + no status updates arrive", on
# questions the dispatcher sends down the fast path — every "top N …" is
# intent=count_or_rank).


async def test_the_fast_path_announces_its_tool_before_calling_it(gub, gate):
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _bot_ctx("campaign", "c1")

    events = [e async for e in fp.fast_path.run_async(ctx)]

    # named, and opened BEFORE the call — a step that arrives with the answer
    # is not progress
    assert _progress(events) == [("get_campaign", False), ("get_campaign", True)]
    assert events[0].partial is True


async def test_the_pings_are_partial_so_they_never_enter_the_history(gub, gate):
    """Load-bearing, not cosmetic. ADK appends an event to the session only
    when it is not partial (`runners.py:881`). A non-partial function_call
    here would be written into the conversation history, and the deep path —
    which runs whenever the fast path declines — would then read a tool call
    that never happened, answered by a response carrying no data."""
    gub.on("GET", "/org/search", {"hits": []})
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    pings = [e for e in events if e.partial]
    assert pings
    assert all(e.partial is True for e in pings)


async def test_a_ping_carries_no_text_so_it_cannot_reach_the_answer(gub, gate):
    """The bot builds the reply from `extractText(evt)`. A progress ping that
    carried text would be appended to the answer the reader sees."""
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _bot_ctx("campaign", "c1")

    events = [e async for e in fp.fast_path.run_async(ctx)]

    for e in (e for e in events if e.partial):
        assert all(p.text is None for p in (e.content.parts or []))


async def test_the_pair_is_closed_even_when_the_fast_path_declines(gub, gate):
    """Declining is the common case — the deep path takes over. An unclosed
    call would hold the bubble on "Reading campaign details…" while the deep
    path is already working underneath it."""
    gub.on("GET", "/org/search", {"hits": []})
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert _answers(events) == []  # nothing said; the deep path will answer
    assert [done for _, done in _progress(events)] == [False, True]


async def test_an_intent_with_no_lookup_announces_nothing(gub, gate):
    """No tool will run, so there is nothing to report. A ping here would
    promise work that never starts."""
    ctx = await _ctx(_decision(intent="smalltalk"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert _progress(events) == []
    assert _answers(events) == []


# ── declined_before_lookup: a decline decided before any call ─────────────────
#
# The speculative root (SPECULATIVE_FAST_PRECHECK) keeps the deep run already
# started beside the router when this says the lookup would decline. The one
# thing it may never do is name a reason for a question the lookup would have
# answered: that would change which branch answers the turn.


class AllSucceed:
    """Every GUB read the fast path can make, answered with data — the most
    accepting GUB there is. A lookup that declines against it declines on any
    answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def get(self, path: str, tool_context=None, **params):
        self.calls.append(("GET", path))
        if path == "/org/search":  # one exact-name hit of every type the lookups ask for
            return [
                {"type": t, "id": f"{t}-1", "name": params["q"], "similarity": 1.0}
                for t in ("account", "campaign", "staff")
            ]
        if path == "/org/staff":
            return {"staff": [{"id": "staff-1", "name": params.get("q")}], "total": 1}
        if path.endswith("/campaigns"):
            return {"campaigns": []}
        if path.endswith("/metadata"):
            return {"metadata": []}
        return {"id": path.rsplit("/", 1)[-1], "name": "chevy", "status": "live"}

    async def post(self, path: str, body: dict, tool_context=None) -> dict:
        self.calls.append(("POST", path))
        return {"results": [{"count": 3}], "total": 3, "truncated": False}


@pytest.fixture
def all_succeed(monkeypatch) -> AllSucceed:
    fake = AllSucceed()
    for name in ("accounts", "discovery", "staff"):
        monkeypatch.setattr(import_module(f"gub_agent.tools.{name}"), "gub_get", fake.get)
    monkeypatch.setattr(import_module("gub_agent.tools.org_query"), "gub_post", fake.post)
    return fake


ACCOUNT_ID = "fdb3f9ff-8094-4851-8ad4-a62f82761ad9"
# The bot's prefix when it has a best guess (blend 02) — `user_text` carries it.
GUESS = (
    f'Best guess: account {ACCOUNT_ID} "Chevy" — the user did not choose this. Ignore the '
    "guess if the question is not about it. Original question: "
)

PURE_DECLINES = [
    # Production shapes (anomaly, 2026-09-18..28): each declined, and each
    # lost the speculation's lead to a restart.
    pytest.param(
        {"intent": "account_facts", "entity_surface": None, "entity_id": ACCOUNT_ID},
        GUESS + "Give me an overview of the Chevy account",
        "no_surface",
        id="q4-account-by-id-only",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "entity_surface": "Chevy",
            "entity_id": ACCOUNT_ID,
            "slots": {"entity": "campaigns", "account": "Chevy", "complete": False},
        },
        GUESS + "Which Chevy campaigns end in the next two months?",
        "slots_incomplete",
        id="q9-slots-incomplete",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "entity_surface": "chevy",
            "slots": {"entity": "campaigns", "account": "chevy", "limit": 20, "complete": True},
        },
        "give me top 20 companies of chevy",
        "rank_no_metric",
        id="top-20-without-a-metric",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "entity_surface": "Chevy",
            "entity_id": ACCOUNT_ID,
            "slots": {
                "entity": "campaigns",
                "status": "live",
                "account": "Chevy",
                "complete": True,
            },
        },
        GUESS + "What campaigns are currently active for Chevy?",
        "shape_other",
        id="which-campaigns",
    ),
    # Every other rule.
    pytest.param(
        {"intent": "account_facts", "entity_surface": "   "},
        "chevy?",
        "no_surface",
        id="account-blank",
    ),
    pytest.param(
        {"intent": "staff_lookup", "entity_surface": None, "entity_id": "s1"},
        "who is she?",
        "no_surface",
        id="staff-by-id-only",
    ),
    pytest.param(
        {"intent": "staff_lookup", "entity_surface": ""}, "who?", "no_surface", id="staff-empty"
    ),
    pytest.param(
        {"intent": "campaign_facts", "entity_surface": "  ", "entity_id": None},
        "tell me about it",
        "no_surface",
        id="campaign-blank-no-id",
    ),
    # BE-6: the id the bot handed over is a piece (the 2026-10-08 18:32 turn) or
    # an account, so `_campaign` hands the turn over before any call — whatever
    # the surface says, and with the prefix the real run reads its type from.
    pytest.param(
        {
            "intent": "campaign_facts",
            "entity_surface": "Hero Spot 30s",
            "entity_id": PIECE_ID,
            "entity_type": "piece",
        },
        _selected("piece", PIECE_ID, "Hero Spot 30s", "статус ролика Hero Spot 30s"),
        "entity_type",
        id="campaign-intent-on-a-piece-id",
    ),
    pytest.param(
        {
            "intent": "campaign_status",
            "entity_surface": None,
            "entity_id": ACCOUNT_ID,
            "entity_type": "account",
        },
        GUESS + "What is the status?",
        "entity_type",
        id="campaign-intent-on-an-account-id-by-id-only",
    ),
    # An id with no type the prefix vouches for: the router invented it, or
    # mangled the copy. The lookup treats it the same way.
    pytest.param(
        {"intent": "campaign_status", "entity_surface": "Silverado", "entity_id": "c1"},
        "статус Silverado",
        "entity_type_unknown",
        id="campaign-intent-on-an-untyped-id",
    ),
    # The chat client rewords the prefix head: the id the router copied is no
    # longer vouched for, and that is a reason of its own (not "entity_type"),
    # so the drift stays countable when the precheck is the only line written.
    pytest.param(
        {
            "intent": "campaign_facts",
            "entity_surface": "Hero Spot 30s",
            "entity_id": PIECE_ID,
        },
        f'User picked piece {PIECE_ID} "Hero Spot 30s". Original question: статус ролика',
        "entity_type_unknown",
        id="campaign-intent-on-an-id-from-a-reworded-prefix",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "staff", "office": "London", "complete": True},
        },
        "how many people are in London?",
        "office",
        id="office",
    ),
    pytest.param(
        {"intent": "count_or_rank", "slots": {"status": "live", "complete": True}},
        "how many are live?",
        "no_entity",
        id="no-entity",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "status": "active", "complete": True},
        },
        "how many campaigns are active?",
        "status_vocab",
        id="another-entitys-status",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "staff", "industry": "auto", "complete": True},
        },
        "how many staff work in auto?",
        "industry_entity",
        id="industry-not-on-accounts",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "period": "recently", "complete": True},
        },
        "how many campaigns did we win recently?",
        "period",
        id="relative-period",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "staff", "account": "chevy", "complete": True},
        },
        "how many people work on chevy?",
        "account_entity",
        id="account-on-staff",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "group_by": "region", "complete": True},
        },
        "how many campaigns per region?",
        "group_by",
        id="unmapped-group-by",
    ),
    pytest.param(
        {"intent": "count_or_rank", "slots": {"entity": "campaigns", "limit": 0, "complete": True}},
        "how many campaigns?",
        "limit",
        id="limit-out-of-range",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "metric": "reach", "complete": True},
        },
        "which campaign has the highest reach?",
        "rank_metric",
        id="unmapped-metric",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "metric": "budget", "complete": True},
        },
        "how much budget do the campaigns have?",
        "count_with_metric",
        id="count-with-a-metric",
    ),
]


@pytest.mark.parametrize("over,question,reason", PURE_DECLINES)
async def test_a_decline_decided_in_code_is_named_before_any_call(
    all_succeed, gate, over, question, reason
):
    decision = _decision(language="en", **over)
    ctx = await _ctx(decision, question)

    assert await fp.declined_before_lookup(decision, question, fp._ToolShim(ctx)) == reason
    assert all_succeed.calls == []  # not one request, the account name included

    # …and the fast path itself declines it with every GUB read succeeding.
    events = [e async for e in fp.fast_path.run_async(ctx)]
    assert _answers(events) == [] and fp.outcome(INV) == "deep"
    assert gate.runs == []


async def test_a_ranking_without_a_metric_is_declined_without_resolving_the_account(all_succeed):
    """ "give me top 20 companies of chevy" (2026-09-18): the lookup resolves
    the account through `/org/search` before it reaches the check that
    declines the question, so it pays a call to learn what the words already
    said. The precheck asks the same builder with the name resolved offline."""
    [param] = [p for p in PURE_DECLINES if p.id == "top-20-without-a-metric"]
    over, question, _ = param.values
    decision = _decision(language="en", **over)
    shim = fp._ToolShim(await _ctx(decision, question))

    assert await fp._LOOKUPS["count_or_rank"](decision, shim, question) is None
    assert all_succeed.calls == [("GET", "/org/search")]  # the lookup: one call, then None
    all_succeed.calls.clear()
    assert await fp.declined_before_lookup(decision, question, shim) == "rank_no_metric"
    assert all_succeed.calls == []  # the precheck: none


NEEDS_A_CALL = [
    pytest.param({"intent": "campaign_status"}, "статус Silverado 2026 Q3", id="campaign-by-name"),
    pytest.param(
        {
            "intent": "campaign_facts",
            "entity_surface": None,
            "entity_id": "c1",
            "entity_type": "campaign",
        },
        "tell me about it",
        id="campaign-by-id",
    ),
    # An id the bot vouched for as a campaign is fetched even with a blank
    # surface; only the call can say whether GUB has it.
    pytest.param(
        {
            "intent": "campaign_status",
            "entity_surface": "Silverado",
            "entity_id": "c1",
            "entity_type": "campaign",
        },
        "статус Silverado",
        id="campaign-by-id-and-name",
    ),
    pytest.param(
        {"intent": "account_facts", "entity_surface": "Chevy"}, "chevy?", id="account-by-name"
    ),
    # A whitespace-only name still reaches `search_staff`, which may match it.
    pytest.param({"intent": "staff_lookup", "entity_surface": "  "}, "who?", id="staff-blank"),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "status": "live", "complete": True},
        },
        "how many campaigns are live?",
        id="complete-count",
    ),
    # Only `/org/search` can say whether "chevy" is one account.
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {
                "entity": "campaigns",
                "status": "live",
                "account": "chevy",
                "complete": True,
            },
        },
        "how many live campaigns does chevy have?",
        id="count-needing-the-account",
    ),
    pytest.param(
        {
            "intent": "count_or_rank",
            "slots": {"entity": "campaigns", "metric": "budget", "complete": True},
        },
        "which campaign has the largest budget?",
        id="ranking-with-a-metric",
    ),
    pytest.param({"intent": "smalltalk"}, "hi", id="not-a-fast-intent"),
]


@pytest.mark.parametrize("over,question", NEEDS_A_CALL)
async def test_a_decline_only_a_call_can_decide_is_left_to_the_lookup(all_succeed, over, question):
    decision = _decision(language="en", **over)
    shim = fp._ToolShim(await _ctx(decision, question))
    assert await fp.declined_before_lookup(decision, question, shim) is None
    assert all_succeed.calls == []


async def test_the_precheck_speaks_only_for_the_lookups_it_restates(monkeypatch, all_succeed):
    """Keyed on the function `_LOOKUPS` holds: a lookup it does not restate —
    here a stand-in that answers — gets no reason, however the decision
    reads."""

    async def answers(decision, shim, question):
        return fp.Lookup(payload=not_found_payload("en", None), tool="get_account_overview")

    monkeypatch.setitem(fp._LOOKUPS, "account_facts", answers)
    decision = _decision(language="en", intent="account_facts", entity_surface=None)
    shim = fp._ToolShim(await _ctx(decision, "chevy?"))
    assert await fp.declined_before_lookup(decision, "chevy?", shim) is None


def _soundness_table() -> list[tuple[dict, str]]:
    """Decisions × questions: every surface / id pairing for the entity
    intents, and the count_or_rank slot space across the question shapes."""
    cases: list[tuple[dict, str]] = []
    for intent in ("campaign_status", "campaign_facts", "account_facts", "staff_lookup"):
        for surface in (None, "", "   ", "Chevy"):
            for entity_id in (None, "", ACCOUNT_ID):
                # What the bot's prefix said the id is (None: nothing vouched).
                for entity_type in (None, "campaign", "account"):
                    over = {
                        "intent": intent,
                        "entity_surface": surface,
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                    }
                    cases.append((over, GUESS + "how is chevy doing?"))
    questions = (
        "how many campaigns are live?",  # count
        "which campaign has the largest budget?",  # rank
        "which campaigns are live?",  # enumerate
        "what is the combined budget?",  # aggregate
        "chevy campaigns",  # nothing recognisable
    )
    space = itertools.product(
        (None, "campaigns", "accounts", "staff", "pieces"),  # entity
        (None, "live", "active"),  # status
        (None, "auto"),  # industry
        (None, "2026 Q3", "recently"),  # period
        (None, "chevy"),  # account
        (None, "status", "region"),  # group_by
        (None, 0, 20),  # limit
        (None, "budget", "reach"),  # metric
    )
    for n, (entity, status, industry, period, account, group_by, limit, metric) in enumerate(space):
        slots = {
            "entity": entity,
            "status": status,
            "industry": industry,
            "period": period,
            "account": account,
            "group_by": group_by,
            "limit": limit,
            "metric": metric,
            "complete": n % 7 != 0,  # the router's own "incomplete" on a slice
            "office": "London" if n % 11 == 0 else None,
        }
        for question in questions:
            cases.append(({"intent": "count_or_rank", "slots": slots}, question))
    for param in PURE_DECLINES + NEEDS_A_CALL:
        over, question = param.values[:2]
        cases.append((over, question))
    return cases


async def test_a_reason_means_the_lookup_declines_whatever_gub_answers(all_succeed):
    """SOUNDNESS. Whenever the precheck names a reason, the real lookup
    returns None with every tool succeeding — so no question the fast path
    would answer is ever sent down the deep path. And the converse on this
    table: with every tool succeeding, each None the lookup returns is one the
    precheck named, so the check is the lookup's code-decided part, not a
    guess at it."""
    ctx = await _ctx(_decision(), "x")
    shim = fp._ToolShim(ctx)
    named = answered = 0
    for over, question in _soundness_table():
        decision = _decision(language="en", **over)
        reason = await fp.declined_before_lookup(decision, question, shim)
        if decision.intent not in fp._LOOKUPS:  # no lookup: the fast path never runs
            assert reason is None
            continue
        before = len(all_succeed.calls)
        lookup = await fp._LOOKUPS[decision.intent](decision, shim, question)
        if reason is not None:
            named += 1
            assert lookup is None, (over, question, reason)
        else:
            answered += 1
            assert lookup is not None, (over, question)
            assert len(all_succeed.calls) > before  # a real lookup was made
    assert named > 1000 and answered > 100  # the table exercises both sides


async def test_the_reason_lines_carry_inv_and_tenant(caplog):
    """The builder's two reason lines and the non-403/404 line used to carry
    neither inv nor tenant, so a decline could not be joined to its turn or
    counted per bot. Appended, never interleaved."""
    ctx = await invocation_ctx(
        state={"gub_jwt": "test-jwt", "tenant": "chevy"}, invocation_id=INV, user_text="x"
    )
    shim = fp._ToolShim(ctx)
    incomplete = _decision(intent="count_or_rank", slots={"entity": "campaigns"})
    office = _decision(
        intent="count_or_rank", slots={"entity": "staff", "office": "London", "complete": True}
    )
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        assert await fp.org_query_args(incomplete, shim, "how many?") is None
        assert await fp.org_query_args(office, shim, "how many?") is None
        assert fp._shortcut({"error": True, "status": 500}, _decision(), shim) is None
        fp.log_declined_before_lookup(shim, "slots_incomplete")
    assert [r.getMessage() for r in caplog.records] == [
        "fast_path: router says the slots do not cover the question — deep path"
        f" (inv={INV}) tenant=chevy",
        f"fast_path: office constraint needs an id lookup — deep path (inv={INV}) tenant=chevy",
        f"fast_path: 500 from the lookup — deep path (inv={INV}) tenant=chevy",
        f"fast_path: declined before lookup reason=slots_incomplete (inv={INV}) tenant=chevy",
    ]


# ── staffing questions on the /ask engine (Runn eval, run 1) ───────────────────

STAFFING = [
    ("staff_lookup", "Marcus Bell", "Is Marcus Bell free next week?"),
    ("staff_lookup", "maya.singh@anomaly.com", "What is maya.singh@anomaly.com working on?"),
    ("staff_lookup", "Kai Nakamura", "What is Kai Nakamura booked on?"),
    ("staff_lookup", "Jack Turner", "When is Jack Turner back from leave?"),
    ("staff_lookup", "Isaac Moore", "Is Isaac Moore overbooked anywhere before Christmas?"),
    ("count_or_rank", None, "How many people are on Vela Motors EV Launch right now?"),
]


@pytest.mark.parametrize(("intent", "surface", "question"), STAFFING)
async def test_on_ask_a_staffing_question_goes_to_the_planner_not_a_profile(
    monkeypatch, all_succeed, intent, surface, question
):
    """A person's bookings, availability or leave live in Runn, which the one
    UB lookup never reads: on the /ask engine the lookup declines before any
    call, and the precheck says so (so the speculative root keeps its run)."""
    monkeypatch.setattr(fp.config, "ASK_CONTRACT", True)
    over = {"intent": intent, "entity_surface": surface, "language": "en"}
    if intent == "count_or_rank":
        over["slots"] = {"entity": "staff", "complete": True}
    decision = _decision(**over)
    shim = fp._ToolShim(await _ctx(decision, question))
    assert await fp._LOOKUPS[intent](decision, shim, question) is None
    assert await fp.declined_before_lookup(decision, question, shim) == "staffing"
    assert all_succeed.calls == []


@pytest.mark.parametrize(
    "question",
    ["Who is Marcus Bell?", "What office is Jane Park in?", "What team is Harry on?"],
)
async def test_a_question_about_who_someone_is_stays_on_the_fast_path(monkeypatch, question):
    monkeypatch.setattr(fp.config, "ASK_CONTRACT", True)
    assert not fp.is_staffing_question(question)


async def test_off_the_ask_engine_staffing_questions_are_not_declined(monkeypatch):
    """The chat bot's engine has no planner tools: its fast path is unchanged."""
    monkeypatch.setattr(fp.config, "ASK_CONTRACT", False)
    decision = _decision(intent="staff_lookup", entity_surface="Marcus Bell", language="en")
    question = "Is Marcus Bell free next week?"
    shim = fp._ToolShim(await _ctx(decision, question))
    assert await fp.declined_before_lookup(decision, question, shim) is None


async def test_a_short_follow_up_to_a_staffing_question_is_one_too(monkeypatch, all_succeed):
    """Run 1's "What's Jane on next week?" → "Jane Park, in LA.": the second
    message names a person and nothing else, so it reads as staff_lookup."""
    monkeypatch.setattr(fp.config, "ASK_CONTRACT", True)
    decision = _decision(intent="staff_lookup", entity_surface="Jane Park", language="en")
    shim = fp._ToolShim(await _ctx(decision, "Jane Park, in LA."))
    shim.prior_question = "What's Jane on next week?"
    assert await fp.declined_before_lookup(decision, "Jane Park, in LA.", shim) == "staffing"
    shim.prior_question = "Who is Jane Park?"
    assert await fp.declined_before_lookup(decision, "Jane Park, in LA.", shim) is None


def test_the_prior_question_is_the_users_last_message_before_this_turn():
    from types import SimpleNamespace

    def said(author, inv, text):
        return SimpleNamespace(
            author=author,
            invocation_id=inv,
            content=genai_types.Content(role="user", parts=[genai_types.Part(text=text)]),
        )

    ctx = SimpleNamespace(
        invocation_id="now",
        session=SimpleNamespace(
            events=[
                said("user", "before", "What's Jane on next week?"),
                said("gub_agent", "before", "Which Jane?"),
                said("user", "now", "Jane Park, in LA."),
            ]
        ),
    )
    assert fp._prior_user_text(ctx) == "What's Jane on next week?"

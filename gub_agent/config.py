"""
config.py — Central configuration for the GUB ADK agent.

All values are read from environment variables (via .env for local dev).
"""

import os

from dotenv import load_dotenv
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.planners import BuiltInPlanner
from google.genai import types as genai_types

load_dotenv()

# gemini-3.5-flash is served ONLY from the Vertex *global* endpoint, and the
# model location is a property of the model — not of the environment. Pin it in
# code so it survives Agent Engine's runtime default (which forces the engine's
# own region, us-central1, and 404s the model — a `.env` value does not stick).
# The engine RESOURCE stays regional (set by `adk deploy --region`); this only
# steers the genai client used for model inference.
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

# ── GUB backend ───────────────────────────────────────────────────────────────
# URL of the GUB backend this agent calls for data.
GUB_BASE_URL: str = os.environ.get(
    "GUB_BASE_URL", "https://gcp-universal-backend-dev-843516467880.us-central1.run.app"
)

# Service JWT for local dev and CI — bypasses the Gemini Enterprise OAuth flow.
# In production (Gemini Enterprise), the JWT is obtained dynamically via
# token exchange and injected per-session.
GUB_SERVICE_JWT: str = os.environ.get("GUB_SERVICE_JWT", "")

# ── Gemini Enterprise OAuth ───────────────────────────────────────────────────
# The Authorization ID registered in Gemini Enterprise under
# Agents → Authorization. Gemini Enterprise stores the user's OAuth token in
# ToolContext at: state["auth_tokens"][GUB_AUTHORIZATION_ID]["token"]["access_token"]
# This value must match EXACTLY — any mismatch causes silent token injection failure.
GUB_AUTHORIZATION_ID: str = os.environ.get("GUB_AUTHORIZATION_ID", "gub-oauth-3")

# ── Agent ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
AGENT_NAME: str = os.environ.get("AGENT_NAME", "gub_agent")

# ── Debugging ─────────────────────────────────────────────────────────────────
# When true, the executor + critic RETURN their thinking-token summaries in the
# event stream (consumed by the debug client). Thinking happens either way — this
# flag only controls whether the summaries are emitted. Keep OFF in production so
# internal reasoning never reaches end users.
EMIT_THINKING: bool = os.environ.get("EMIT_THINKING", "false").lower() in ("1", "true", "yes")

# ── Sandbox (per-call overrides) ───────────────────────────────────────────────
# Master switch for the experiment sandbox (sandbox.py): when false, a `sandbox`
# key in session state is ignored completely — no warn, no error, nothing logged.
# Set it ONLY on the dev/sandbox engine; the prod engine must stay inert so an
# override aimed at a sandbox run can never reach the engine serving the Chat bot.
SANDBOX_ENABLED: bool = os.environ.get("SANDBOX_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Models a sandbox run may select. An allowlist rather than a free-text field:
# `llm_request.model` is swapped inside the SAME genai client (Vertex, global
# endpoint — pinned above), so only models that client can serve are valid; a
# cross-provider model needs a different BaseLlm and is out of scope.
# The default holds only ids verified to answer from this project's global
# endpoint (2026-09-07): `gemini-3.5-pro` 404s there AND in us-central1, and a
# 404 inside the engine surfaces to the caller as an EMPTY 200 stream — so an
# unservable id in the allowlist is a silent failure, not a loud one.
SANDBOX_MODEL_ALLOWLIST: tuple[str, ...] = tuple(
    name.strip()
    for name in os.environ.get("SANDBOX_MODEL_ALLOWLIST", "gemini-3.5-flash,gemini-2.5-pro").split(
        ","
    )
    if name.strip()
)

# Models that accept a NAMED thinking level (`thinking_level=LOW|MEDIUM|HIGH`, the
# 3-series knob). Any other allowlisted model rejects it with 400 INVALID_ARGUMENT
# — verified live 2026-09-07 with gemini-2.5-pro — and inside the engine that 400
# reaches the caller as an empty 200 stream. The baseline planners pin MEDIUM
# (executor) and LOW (critic), and thinking off (router, formatter), so a sandbox
# run that swaps to a model outside this set must ALSO set every role to DYNAMIC
# (`thinking_budget=-1`); sandbox.py refuses the run up front otherwise.
SANDBOX_THINKING_LEVEL_MODELS: tuple[str, ...] = tuple(
    name.strip()
    for name in os.environ.get("SANDBOX_THINKING_LEVEL_MODELS", "gemini-3.5-flash").split(",")
    if name.strip()
)


def build_thinking_planner(
    thinking_level: str | None = None, *, thinking_budget: int | None = None
) -> BuiltInPlanner:
    """Native thinking planner for one agent — a FRESH ThinkingConfig per call.

    Default (neither given): dynamic budget — the model thinks as much as it
    wants. Pass a level ('MINIMAL'/'LOW'/'MEDIUM'/'HIGH', the 3-series knob)
    to cap it — the critic runs at LOW because it's a checklist judge whose
    unbounded thinking measured 13-16s/turn (~29% of a whole turn). Or pass a
    budget in tokens: 0 switches thinking off (the router and the formatter,
    ROUTER_THINKING_OFF / FORMATTER_THINKING_OFF below), -1 is the dynamic
    default. Not both — they are two answers to one question.

    Fresh, never shared: the planner assigns its own object into every request
    (ADK planners/built_in_planner.py; see sandbox.py:_thinking_config), so two
    agents on one config would be one mutation away from sharing a knob.
    Thought summaries are emitted only when EMIT_THINKING is set.
    """
    if thinking_level is not None and thinking_budget is not None:
        raise ValueError("build_thinking_planner: pass thinking_level or thinking_budget, not both")
    if thinking_level is not None:
        return BuiltInPlanner(
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=EMIT_THINKING,
            ),
        )
    budget = -1 if thinking_budget is None else thinking_budget
    return BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=budget,
            # Nothing to summarise at budget 0, and include_thoughts=true with
            # thinking off is a pairing production has never sent — the sandbox
            # engine runs EMIT_THINKING=1, where a rejection would reach the
            # caller as an empty 200 stream.
            include_thoughts=EMIT_THINKING and budget != 0,
        ),
    )


# ── Thinking off for the router and the formatter ────────────────────────────
# Both run with thinking_budget=0 instead of thinking_level=LOW. Neither needs
# to deliberate: the router makes one classification and the formatter renders
# given text into a given schema. Measured 2026-09-25 by replaying 20 real
# production router requests under four arms: at LOW every router call spent
# ~99 thought tokens, and TTFT p90 was 3.2 s against 1.1 s at budget 0, with
# the same intent on 19 of 20. Under LOW the formatter thinks on ~43% of its
# calls, p90 3.1k thought tokens, worth ~3.2-3.8 s of the mean formatter call.
# The formatter's output at budget 0 was NOT replayed. Watch `format_gate:
# attempt … rejected … tenant=…` against formatter calls after a deploy.
#
# Off (0) is the rollback of each: thinking_level=LOW, as before. Read at import
# (sandbox.py derives the baseline levels its provenance reports from these), so
# a change needs a redeploy. A sandbox run still overrides either per call
# (router_thinking_level / formatter_thinking_level). Set explicitly in both
# deploy env files, like the other rollbacks.
ROUTER_THINKING_OFF: bool = os.environ.get("ROUTER_THINKING_OFF", "1").lower() in (
    "1",
    "true",
    "yes",
)
FORMATTER_THINKING_OFF: bool = os.environ.get("FORMATTER_THINKING_OFF", "1").lower() in (
    "1",
    "true",
    "yes",
)

# ── A stalled router call is sent twice ──────────────────────────────────────
# When a ROUTER call has had no first chunk for this many milliseconds, its
# request is sent a second time and whichever answers first is relayed; the
# other is cancelled (models.py: VendorRouter._hedged). The router's first
# token sometimes waits 10-67 s inside Vertex, in day-dependent episodes, and a
# paid replay of 20 production router requests — stalled ones included — saw
# no stall: the condition is the server's at that moment, not the request's.
# Healthy router TTFT is p90 1.1 s with thinking off, so 4000 fires only on a
# call already ~4x late. The deep path hides the router behind its speculative
# run (SPECULATIVE_DEEP); this is for the turns that wait for it — smalltalk,
# abstain, clarify, the fast path.
#
# Same model, same request, same config (a sandbox run's per-call overrides
# included): the answer is unchanged, only when it arrives. The price is one
# extra router call on the calls it fires on: the router prompt is uncached,
# 2.6k tokens in a short session and ~20k in a long one (model_call prompt=,
# cached=0), so ~$0.004-0.03 per fire at the Flash input rate. A call genai is
# already retrying (a 429/5xx answered it) is not hedged; once the hedge has
# fired, each request may still retry on its own budget.
#
# 0 is the rollback: nothing is ever sent twice, and every call takes the
# unhedged path, byte for byte. Read at import (build_model below hands it to
# every VendorRouter; only the router's calls use it): a change needs a
# redeploy. Set explicitly in both deploy env files, like the other rollbacks.
ROUTER_HEDGE_AFTER_MS: int = int(os.environ.get("ROUTER_HEDGE_AFTER_MS", "4000"))

# ── A stalled executor or formatter call is sent twice ───────────────────────
# The router hedge above, for the two agents a reader waits through on every
# deep turn (models.py, "The executor and formatter hedges"). The first token
# of an executor ReAct round has waited 102.7 s on a production turn with no
# retry and nothing wrong with the request (2026-10-15, round 3 of a "whats
# new?" turn: prompt 57.6k, ttft_ms=102724, retries=0); a formatter's 62 s on
# another. The condition is the server's, as for the router, so the same copy
# of the same request sent again answers on its own clock.
#
# Two deadlines, not one: the executor thinks before its first chunk (healthy
# rounds take seconds to start), the formatter runs with thinking off. Each is
# the milliseconds a call of THAT agent may go without a first chunk before
# its request is sent a second time; the stream that answers first is relayed
# whole and the other is cancelled. The answer is unchanged — same model, same
# request, same config — and so are the tools: ADK runs them after the reply,
# from the relayed stream only.
#
# The price is one extra call's input on the calls it fires on, never
# otherwise: an executor round is 20-65k prompt tokens, often uncached, so
# ~$0.03-0.10 per fire at the Flash input rate; a formatter brief up to ~75k.
#
# 0 is the code default and the ROLLBACK: nothing is ever sent twice and every
# call takes today's path byte for byte. Read at import: a redeploy either
# way. Stated in both deploy env files.
EXECUTOR_HEDGE_AFTER_MS: int = int(os.environ.get("EXECUTOR_HEDGE_AFTER_MS", "0"))
FORMATTER_HEDGE_AFTER_MS: int = int(os.environ.get("FORMATTER_HEDGE_AFTER_MS", "0"))

#: The formatter's agent name (agents/formatter.py FORMATTER_NAME), the key its
#: hedge deadline is looked up by. Restated here because that module imports
#: this one; a test pins the two together.
FORMATTER_AGENT_NAME = "formatter"

# ── A stalled critic call is sent twice ──────────────────────────────────────
# The executor and formatter hedges above, for the critic (models.py, "The
# critic hedge"). Its streamed first token stalls the same way, and a pass
# waits for the verdict. Measured on 33 instrumented tenant turns over 7 days
# (a $0 log analysis): first-token p50 5.0 s, 35 of 49 calls answer within
# 9.6 s and 12 took more than 15 s (the other 2 fall in between), each of those
# 12 with TTFT about equal to its duration and at most 297 thought tokens, so
# the wait is not thinking. 15000 fires on those 12 alone.
#
# The price, only on the calls it fires on: the critic's prompt sent once more,
# ~$0.045 at the median and ~$0.13 at most — about $0.59 per 100 turns. One
# invocation fires at most ONE critic hedge (models.py, "The hedge cap").
#
# 0 is the code default and the ROLLBACK, as for the other hedges. Read at
# import: a redeploy either way. Stated in both deploy env files.
CRITIC_HEDGE_AFTER_MS: int = int(os.environ.get("CRITIC_HEDGE_AFTER_MS", "0"))

#: The critic's agent name (agents/critic.py), the key its hedge deadline and
#: its hedge budget are looked up by. Restated here because that module imports
#: this one; a test pins the two together.
CRITIC_AGENT_NAME = "critic"

# ── The executor's answer round is hedged on a later deadline ────────────────
# The executor hedge above counts from the request's start to its first chunk.
# For a round that calls tools that is a fair clock: it thinks 0.2-0.5k tokens
# first and the chunk takes seconds. The round past the ReAct budget
# (agents/round_limiter.py), the one that writes the answer, is not like that:
# its first chunk comes after ALL of its thinking, 0.7-4.3k thought tokens
# (median 2.1k, 6.5k seen), so a healthy call can wait well past 20 s. In a
# local run (2026-10-16) two hedges fired at 20 s on such rounds and both lost:
# their first requests answered at 26.5 s and 28.9 s, with thinking of 4.3k and
# 6.5k tokens and no stall. Each lost hedge is a second prompt's input
# (~$0.03-0.10) and one of the invocation's two executor-or-formatter hedges
# (models.py, "The hedge cap"), which a formatter stall then finds spent.
#
# This is the deadline, in milliseconds, of THAT round only (models.py, "The
# answer round's deadline"); the tool rounds keep EXECUTOR_HEDGE_AFTER_MS.
# 35000 is above both healthy calls seen and still fires on a stall of the 100 s
# kind (an executor round waited 102.7 s on 2026-10-15).
#
# 0 is the code default and the ROLLBACK: the round takes the executor's
# deadline like every other. An EXECUTOR_HEDGE_AFTER_MS of 0 turns every
# executor hedge off, this one included. Read at import: a redeploy either way.
# Stated in both deploy env files.
EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS: int = int(
    os.environ.get("EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS", "0")
)

# ── Thinking on the executor's answer round ──────────────────────────────────
# The round past the ReAct budget (agents/round_limiter.py) writes the answer
# from data already gathered, and it thinks longest of any executor round:
# 0.7-4.3k thought tokens (median 2.1k, 6.5k seen) against 0.2-0.5k on a tool
# round. Thinking is not streamed, so the first chunk waits for all of it: about
# 2.5 s + 4.4 s per thousand thought tokens in a local run, most of that round's
# 13 s median in production (16 of 29 passes reach it). LOW on that one round is
# expected to take ~5 s off a turn that reaches it — arithmetic, not a
# measurement — at the risk of a thinner answer.
#
# AUTO is the code default and the ROLLBACK: the round thinks as every other
# executor round does (MEDIUM). LOW lowers that round only; nothing else takes
# it, not the tool rounds and not the other agents. Anything but LOW reads as
# AUTO, so a typo in an env file can neither stop the engine from starting nor
# send a level the model may refuse (only levels verified on this model are
# accepted). Read as a module attribute at call time; the sandbox override
# `synthesis_round_thinking_level` wins over it per call. Stays AUTO in both
# deploy env files until a paid replay of the round on both levels shows the
# answer holds.
SYNTHESIS_ROUND_THINKING_LEVELS = ("AUTO", "LOW")
SYNTHESIS_ROUND_THINKING_LEVEL: str = (
    "LOW"
    if os.environ.get("SYNTHESIS_ROUND_THINKING_LEVEL", "AUTO").strip().upper() == "LOW"
    else "AUTO"
)


# Where Anthropic models are served for a sandbox run that selects a `claude-*`
# id (gub_agent/models.py). Claude on Vertex serves from the global endpoint;
# the genai client's own GOOGLE_CLOUD_LOCATION pin above is a separate thing.
# The fast path's disambiguation tie band — the agent-side twin of the bot's
# CLARIFY_TIE_BAND. Two `/org/search` hits whose scores differ by at most this
# are TIED, and a tie means ask rather than pick. Replaces the module constants
# FAST_MARGIN_FLOOR / FAST_EXACT_FLOOR, which were a hardcoded copy of the
# bot's old pair and carried the same inversion: the "top score is high, so we
# are confident" clause vetoed the question precisely when every candidate
# matched equally well. See gub-gchat-bot entity/resolve.ts:initialDecision and
# the Anomaly workspace's task-specs/clarify-01-always-ask-when-unsure.md.
FAST_TIE_BAND: float = float(os.environ.get("FAST_TIE_BAND", "0.10"))

# Below this the router is guessing between intents — ask instead of guessing
# with it (agents/dispatcher.py:choose).
#
# NOT RATIFIED. Unlike the tie band above, this one cannot be measured without
# spending model calls: it gates a branch that only exists once the router has
# run. It is config rather than a constant so it can be moved without a code
# change, and the dispatcher already logs `intent=… confidence=… branch=…` on
# every turn, so the reliability diagram this needs can be built from
# production logs for free:
#   resource.type="aiplatform.googleapis.com/ReasoningEngine"
#     AND textPayload:"dispatcher: intent"
# See the Anomaly workspace's task-specs/clarify-01-always-ask-when-unsure.md.
ROUTER_CONFIDENCE_FLOOR: float = float(os.environ.get("ROUTER_CONFIDENCE_FLOOR", "0.70"))

# ── Conversation window ──────────────────────────────────────────────────────
# How many COMPLETE turns of transcript the model is shown, counted back from
# and INCLUDING the current one (agents/context_pruning.py:trim_to_recent_turns).
#
# Counted in TURNS, never in messages: one Chat turn is 4-10 genai Content
# objects (question -> function_call -> function_response -> ... -> answer), so
# a message-counted window cuts a turn in half.
#
# This is what replaces the bot's 5-minute idle reset as the bound on context
# growth (gub-gchat-bot threads.ts:IDLE_RESET_MS). 0 or negative disables it,
# which is also the rollback: no code redeploy, just the variable — and it
# wins over any per-session value, or it would roll back nothing.
#
# PER SESSION since thread-topics. The bot writes `context_turn_window` into
# session state — 10 for the main DM stream, 200 for a thread — and the
# executor, router and critic all read it per request
# (context_pruning.resolve_turn_window). This value is the DEFAULT, for every
# session that carries no key: the Chevy tenant bot (a pre-memory-00 image on
# this same engine), gub-sandbox-ui, the /blend harness, and any bot session
# created before the key existed until its next turn pushes it. So the default
# is a finite window and must never become "unlimited": those callers would
# inherit it with no way to opt out. 10, matching the main stream, so an
# unlabelled caller behaves like a top-level conversation.
#
# Per-run A/B follows from the same thing: a sandbox run varies the window by
# creating its session with `state={"context_turn_window": N}` — no
# SandboxOverrides field, and it works on the production engine too, where
# `state["sandbox"]` is ignored.
CONTEXT_TURN_WINDOW: int = int(os.environ.get("CONTEXT_TURN_WINDOW", "10"))

# ── File search (search-01) ──────────────────────────────────────────────────
# Master switch for the Drive file-NAME search tool. OFF is exactly today's
# behaviour: `agents/tool_gate.py` withholds `find_files` from every turn, so
# the model is never offered it and can never call it.
#
# It exists because the three services deploy independently and GUB's own
# `FILE_SEARCH` flag defaults to off — and while it is off `/org/files/search`
# answers `200 []`, which is indistinguishable from a genuine miss. An engine
# deployed ahead of its backend would therefore report a file GUB DOES hold as
# "not found by name": a confident false statement produced by an env var, with
# nothing in the response to warn the model. A switch on this side makes the
# deploy order safe in either direction — turn this on only once the backend
# this engine calls has FILE_SEARCH on.
#
# Set explicitly in both deploy env files (the same reasoning as
# SANDBOX_ENABLED): it is the rollback, and an operator under pressure should
# be editing a line that is already in front of them.
FILE_SEARCH_ENABLED: bool = os.environ.get("FILE_SEARCH_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)

# ── The critic's place in the deep path ──────────────────────────────────────
# Run the critic LLM CONCURRENTLY with the format gate instead of after it
# (agent.py:build_deep_agent, agents/critic.py:SpeculativeCritic). The critic
# judges the executor's tool coverage and never reads the formatter's output,
# so waiting for the formatter bought nothing but time: over 155 router-era
# deep turns, overlapping the two saves p50 2.8 s / p90 8.9 s / mean 4.5 s per
# turn. The price is a critic call on the turns the gate would have settled in
# code after the formatter — an abstain payload (21 of 155) — whose verdict is
# then discarded unseen.
#
# Off is the rollback of the TREE: the serial one this replaced, [executor,
# format_gate, critic_gate, loop_escalator], built from the same objects. It is
# not a rollback to b11e400's behaviour. The last-pass skip below has its own
# switch, and the serial gate reads this pass's payload off its events like the
# resolver does — a fix both wirings keep (agents/critic.py:_this_pass_payload).
# Read at import — a change needs a redeploy. Set explicitly in both deploy env
# files for the reason CONTEXT_TURN_WINDOW is: an operator under pressure should
# be editing a line that is already in front of them.
CRITIC_PARALLEL: bool = os.environ.get("CRITIC_PARALLEL", "1").lower() in ("1", "true", "yes")

# No critic LLM on the deep loop's LAST iteration (agents/critic.py:
# _is_final_pass): its verdict there cannot buy another pass, so the gate writes
# a sufficient one in code — 2.8-20.7 s off every retried production turn. The
# price is the critic's opinion of the retry: gubCriticSufficient on a retried
# turn is always true. Independent of CRITIC_PARALLEL, in both wirings.
#
# Off is its rollback: the critic LLM judges the last pass again, as it did
# before. Read at import, set explicitly in both deploy env files, like the flag
# above.
CRITIC_SKIP_FINAL_PASS: bool = os.environ.get("CRITIC_SKIP_FINAL_PASS", "1").lower() in (
    "1",
    "true",
    "yes",
)

# A first pass that answered from memory is sent back without a formatter call
# (agents/critic.py:from_memory_pass; the shortcut in agents/format_gate.py).
# Such a pass made no tool call, so the evidence index is empty and the gate
# could only abstain; the critic gate then re-queries on that abstain in code
# (`CriticGate._abstain_verdict`). The formatter call that produces the abstain
# takes 2-4.6 s on a healthy turn and 15-30 s when its first token stalls, and
# the speculative critic started beside it is cancelled unread. With the switch
# on, the gate writes the same abstain payload itself and no critic call is
# started for the pass. The verdict, its reason and its feedback are today's.
#
# Two things differ. A from-memory pass whose formatter would have chosen
# `clarify` over abstain is now re-queried as well. And the pass-1 payload
# event, which a stream reader sees before the retry's answer replaces it, is
# the gate's own generic abstain instead of the formatter's wording.
#
# 0 is the code default and the ROLLBACK: the formatter is called as before.
# Read as a module attribute at call time. Stated in both deploy env files.
REQUERY_SKIPS_FORMATTER: bool = os.environ.get("REQUERY_SKIPS_FORMATTER", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Stop a formatter attempt the answer contract has already rejected
# (agents/format_gate.py, "Early abort"). The contract's field checks — filler,
# word budgets, block shape — on the headline and the blocks are decidable the
# moment the streamed `blocks` array closes, at a median 24% of the payload
# text; the rest of a rejected attempt (citations, facts, follow-ups) is
# generated only to be thrown away. `contract` checks those two fields as they
# stream and, on a field error, closes the model call and takes the retry with
# the feedback today's full-payload check writes. That class only: a
# grounding-rejected attempt is yielded and stored before the gate judges it,
# so aborting one would change the session history.
#
# 0 is the code default and the rollback: every formatter event is relayed as
# before and nothing is scanned. Any value other than `contract` reads as 0.
# Read at import: a redeploy either way. Stated in both deploy env files —
# deliberately NOT at the same value (see the comments there).
FORMAT_GATE_EARLY_ABORT: str = os.environ.get("FORMAT_GATE_EARLY_ABORT", "0").strip().lower()

# Import at package load what ADK would otherwise import on the first model
# request (gub_agent/eager_imports.py): the anthropic SDK behind ADK's content
# builder, the genai client's async stack, auth and flow modules — 2,115
# modules a fresh process imports on its first turn and never again, ~2 s
# locally, and the bulk of the 6.2-16.4 s the first request of a production
# process waited before its first model send (2026-09-17..25). Imports only:
# nothing is constructed or called, and a module missing in another ADK
# version is skipped. The package is loaded by the first request of any kind,
# so this moves the cost off the first user turn onto whichever request comes
# first after a restart — the warm-up ping's job. 0 is the rollback: exactly
# today's lazy imports. Read at import: a redeploy either way.
EAGER_ADK_IMPORTS: bool = os.environ.get("EAGER_ADK_IMPORTS", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Stream the evidence index to the caller (latency-04 A4). The index the
# formatter's brief is composed from is final the moment the format gate reads
# it — 6.9-24.8 s before the answer payload (n=10) — but it lives in this
# process only: its ids, its compaction and its 400-entry cap exist nowhere on
# the wire. With this on, the gate yields ONE partial event per run, authored
# `format_gate`, with no content and `custom_metadata={"evidence_rows": [...]}`
# — {evidence_id, entity_id, field, value} per entry, in index order — which
# the bot's claim filter uses to start on the facts the formatter will cite
# (its B3 part ii). A partial event is streamed but never appended to the
# session (ADK runners.py), so session events, history, state and billing are
# unchanged, and a bot that predates it skips it for content (format_gate
# partials are ignored) — it only counts it: every bot's `gubRawEvents` and the
# sandbox batch record's `events` go up by one per gate run. Off is the
# rollback: no event. Read at import; set explicitly in both deploy env files.
EVIDENCE_ROWS_EVENT: bool = os.environ.get("EVIDENCE_ROWS_EVENT", "0").lower() in (
    "1",
    "true",
    "yes",
)

# The `/ask` contract (gub-agent#88, UB docs/ASK.md). On, the engine's root is
# `AskTurn` around the usual root: it streams progress lines and ends each turn
# with one AskResponse, both as partial events authored `ask` (agents/ask.py).
# Meant for the engine UB's /ask calls. Off, the default, the root is today's,
# event for event, so an engine serving the chat bot is unchanged. Read at
# import.
ASK_CONTRACT: bool = os.environ.get("ASK_CONTRACT", "0").lower() in ("1", "true", "yes")

# Reader text carries no record ids and no raw ISO timestamps (latency-05 W1-5).
# An account overview came back as "The account (id: <uuid>) was created on
# 2026-06-04T18:33:52.846Z ..." because the fast path's draft copies every
# scalar field of the response, the id and the timestamps included, and the
# formatter wrote what it was given. On: the draft leaves out the record-id
# fields (they stay in the evidence rows and `facts`, which is where a machine
# reads them), and every payload the format gate emits has the record ids and
# ISO timestamps rewritten out of its reader-visible fields (headline, text
# blocks, bullets, table cells) by `agents/answers.py:scrub_for_emit` — a UUID
# out, an ISO timestamp as "Jun 4, 2026"; `facts`, `citations`, source ids and
# `[n]` / `[evidence id]` markers are never touched. 0 is the ROLLBACK:
# today's draft and today's payloads, byte for byte. Read at import: a
# redeploy either way.
#
# Watch `reader_text: scrubbed ids=… dates=… inv=… tenant=…`, logged only when
# a scrub changed something (from answers.py, so `textPayload:"format_gate"`
# does not see it).
READER_TEXT_SCRUB: bool = os.environ.get("READER_TEXT_SCRUB", "0").lower() in (
    "1",
    "true",
    "yes",
)

CLAUDE_VERTEX_LOCATION: str = os.environ.get("CLAUDE_VERTEX_LOCATION", "global")


def build_model() -> BaseLlm:
    """The agents' model object: a VendorRouter (gub_agent/models.py) over a
    Gemini client, so a sandbox run that writes a `claude-*` id into
    `llm_request.model` reaches Anthropic on Vertex while every other request —
    and every prod request, where the sandbox never writes the field — goes to
    the same Gemini client as before.

    Gemini model wired with truncated exponential backoff + jitter on 429/5xx.

    gemini-3.5-flash on Vertex is served via Dynamic Shared Quota: a transient
    429 RESOURCE_EXHAUSTED reflects shared-pool congestion, NOT a project quota
    that can be raised. Google's required mitigation is client-side retry with
    backoff. ADK/genai default to NO retries (one attempt), so a single
    transient 429 kills the whole turn — the executor makes several model calls
    per question, so the odds of one hitting congestion are real. HttpRetryOptions
    adds tenacity-backed exponential backoff + jitter, retrying 429/408/5xx.

    Bounds are chat-latency-aware. Retry is PER CALL and a turn makes several
    model calls with NO shared retry budget, so under sustained congestion the
    per-call waits compound. attempts=3 (up to ~5s of backoff per call worst
    case — ~2s after the first failure, ~3s after the second) keeps that
    compounding well under the bot's 120s stream ceiling: a transient
    blip still clears (most do on the first retry), but a genuinely overloaded
    turn fails fast instead of dragging toward the timeout. Retriable codes are
    the genai defaults (408/429/5xx) — genuine client errors (400/403/404) are
    NOT retried. Retries are logged by genai at INFO (before_sleep), and each
    call's `model_call:` line counts its own (`retries=`, models.py).

    The router hedge (ROUTER_HEDGE_AFTER_MS) sits above this, not inside it:
    one second send per router call at most, never one per attempt, never
    while the first send is retrying, and each send counts its own retries.
    The executor, formatter and critic hedges (EXECUTOR_/FORMATTER_/
    CRITIC_HEDGE_AFTER_MS) work the same way, on their own deadlines.
    """
    from .models import VendorRouter  # noqa: PLC0415 — models.py imports config

    gemini = Gemini(
        model=GEMINI_MODEL,
        retry_options=genai_types.HttpRetryOptions(
            attempts=3,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=1.0,
        ),
    )
    return VendorRouter(
        model=GEMINI_MODEL,
        gemini=gemini,
        claude_location=CLAUDE_VERTEX_LOCATION,
        router_hedge_after_ms=ROUTER_HEDGE_AFTER_MS,
        agent_hedge_after_ms={
            name: after_ms
            for name, after_ms in (
                (AGENT_NAME, EXECUTOR_HEDGE_AFTER_MS),
                (FORMATTER_AGENT_NAME, FORMATTER_HEDGE_AFTER_MS),
                (CRITIC_AGENT_NAME, CRITIC_HEDGE_AFTER_MS),
            )
            if after_ms > 0
        },
        synthesis_hedge_after_ms=EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS,
    )

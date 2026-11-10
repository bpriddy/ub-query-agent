"""
agent.py — GUB AI Agent pipeline.

The question is ROUTED first and only then answered (blend 04):

  SequentialAgent("gub_root")
    ├─ sandbox_echo — records the resolved sandbox config on experiment runs
    │                  (silent on ordinary ones); see sandbox.py
    ├─ router      — one LLM call, no tools, output_schema=RouterDecision
    │                  (agents/router.py); classifies and stops
    └─ dispatcher  — picks a branch in plain code (agents/dispatcher.py):
                     abstain / smalltalk / clarify with no data at all,
                     fast_path for a FACT intent with a known entity,
                     deep_agent for everything else

  deep_agent = LoopAgent("gub_pipeline", max_iterations=2)
    ├─ executor_agent — runs the existing tool-using LLM
    ├─ format_and_critic — ParallelAgent, both at once (CRITIC_PARALLEL=1):
    │   ├─ format_gate — renders the answer into the typed AnswerPayload
    │   │                (agents/formatter.py) and enforces the contract in
    │   │                code (agents/format_gate.py, blend 03)
    │   └─ critic_speculation — the critic LLM, its verdict held
    ├─ critic_gate — evaluates information sufficiency; emits structured
    │                  verdict {sufficient, reason, feedback} into state
    │                  (with CRITIC_PARALLEL=1: resolves the held verdict
    │                  against THIS pass's payload). On the last iteration it
    │                  passes in code: no verdict could buy another pass
    └─ loop_escalator — exits the loop early when critic verdict is sufficient

  With CRITIC_PARALLEL=0 the middle is the serial pair it replaced —
  format_gate, then critic_gate running the critic LLM (`build_deep_agent`).

The deep path is the pipeline as it was, renamed: the critic-before-commit
pattern from the Agentic RAG architecture, with the per-pass budget resets
still attached to the executor, so a question that reaches it behaves exactly
as it does today. What blend 04 adds is the option NOT to reach it: latency is
model turns (thinking tokens ↔ elapsed, r=0.86), so the only way to answer
"what is the status of Silverado 2026 Q3" in seconds is to not take 1-3
executor rounds plus a critic pass for one HTTP call's worth of information.

root_agent is what ADK looks for at deploy time and what callers
(gub-gchat-bot, Agentspace) invoke via stream_query. Same engine ID, same
`stream_query` shape, same author-routed answer channel — the routing layer is
invisible at the boundary, the way the LoopAgent's introduction was in June.

sandbox_echo stays FIRST: the provenance event must precede any work.

With SPECULATIVE_DEEP=1 (the default) the router and the dispatcher run inside
`speculative_dispatch`, which starts the deep path at the same moment as the
router and keeps it only when the dispatcher picks it (agents/speculation.py);
0 is the tree above, exactly (`build_root`).

The executor's instruction text lives in `prompts/executor.py` (edit it
there); the critic's lives in `prompts/critic.py`, the router's in
`prompts/router.py`.

Prompt, model, thinking level and temperature are overridable per call from
`state["sandbox"]` (`sandbox.py`) so experiments need no redeploy — the router
included (`router_instruction`, `router_variant`, `router_thinking_level`).
With no such key every override point is a no-op.
"""

from google.adk.agents import Agent, BaseAgent, LoopAgent, ParallelAgent, SequentialAgent

from .agents.ask import AskTurn
from .agents.circuit_breaker import circuit_breaker, reset_tool_budget
from .agents.context_pruning import (
    strip_prior_turn_tool_parts,
    strip_source_metadata,
    trim_to_recent_turns,
)
from .agents.critic import (
    CriticGate,
    CriticResolver,
    SpeculativeCritic,
    critic_agent,
    escalator_agent,
    settle_speculation,
)
from .agents.dispatcher import Dispatcher
from .agents.evidence_index import record_evidence, reset_evidence_index
from .agents.fast_path import fast_path
from .agents.format_gate import format_gate
from .agents.round_limiter import reset_rounds, round_limit, synthesis_round_thinking
from .agents.router import router_agent
from .agents.speculation import SPECULATIVE_DEEP, SpeculativeDispatch
from .agents.tool_gate import tool_gate
from .config import (
    AGENT_NAME,
    ASK_CONTRACT,
    CRITIC_PARALLEL,
    EAGER_ADK_IMPORTS,
    build_model,
    build_thinking_planner,
)
from .instruction_utils import with_current_date
from .models import bind_model_call
from .prompts import EXECUTOR_INSTRUCTION, RUNN_GUIDANCE, TENANT_EXECUTOR_INSTRUCTION
from .sandbox import (
    EXECUTOR_THINKING_LEVEL,
    sandbox_before_model,
    sandbox_echo,
    sandbox_instruction,
)
from .tenant import tenant_instruction
from .tools import ALL_TOOLS
from .tools.runn import RUNN_TOOLS


def _before_agent(callback_context):
    """before_agent_callback — fires once per executor pass (LoopAgent runs the
    executor's run_async each iteration). Reset the per-pass round + tool budgets
    so a critic-requested retry starts fresh instead of inheriting the previous
    pass's counts (which would open the retry already at the cap, tools stripped,
    unable to make the call the critic asked for). The evidence index resets on
    the same cadence: the formatter grounds against what THIS pass retrieved
    (the executor's own doctrine is to re-query, never answer from prior data)."""
    reset_rounds(callback_context)
    reset_tool_budget(callback_context)
    reset_evidence_index(callback_context)
    return None


def _before_model(callback_context, llm_request):
    """Chain the model-level guards: apply any sandbox overrides (model,
    thinking, temperature), window the transcript to the session's last N
    turns (N from state["context_turn_window"], else CONTEXT_TURN_WINDOW), prune
    prior-turn tool payloads, mask `_sources` citation plumbing from
    current-turn results, withhold the file-search tool on non-file turns, then
    cap ReAct rounds (strip tools past the budget) and, when
    SYNTHESIS_ROUND_THINKING_LEVEL says so, think less on the round that writes
    the answer.

    Sandbox first, deliberately: it only writes `llm_request.model` and fields
    of `config`, while the round limiter may clear `config.tools` — neither can
    undo the other. With no `sandbox` key in state it is a no-op.

    Two orderings after it are load-bearing (memory-00 §3.1):

    * `trim_to_recent_turns` BEFORE `round_limit`, for CORRECTNESS. The round
      limiter appends a role="user" text content ("Tool-call limit reached…")
      that satisfies _has_user_text. A window running after it would read that
      injection as the newest turn boundary and could cut away the user's
      actual question on a synthesis round.
    * `trim_to_recent_turns` BEFORE `strip_prior_turn_tool_parts`, for COST —
      the pruner then rebuilds fewer Content objects. NOT for orphan safety:
      the reverse order orphans nothing (§3.4), and justifying it that way
      would send the next reader looking for a hazard that is not there.

    And one more since search-01:

    * `tool_gate` BEFORE `round_limit`, the last step that touches the tool
      list (`synthesis_round_thinking` follows it, below). The limiter's whole
      job past the budget is to empty `config.tools` so the model cannot call
      anything; a gate running after it would be rewriting a list the limiter
      had deliberately cleared. Today that is merely pointless — filtering an
      empty list removes nothing — but the invariant worth keeping is that
      NOTHING writes `config.tools` after the limiter has cleared it, because
      the next gate written here may well rebuild the list rather than filter
      it, and would hand back a declaration the limiter had just taken away.
      Withholding one tool and withdrawing all of them compose in one
      direction only.

    And one since latency-05:

    * `synthesis_round_thinking` AFTER `round_limit`, because it reads the round
      counter the limiter just ticked (it never ticks) and so must run once per
      model call, after the tick. It writes `config.thinking_config` only —
      never `config.tools` — so `round_limit` stays the last writer of the tool
      list, and it comes after the sandbox's thinking override on purpose: an
      explicit `synthesis_round_thinking_level` is the more specific of the two.
    """
    bind_model_call(callback_context)  # who is calling, for the model_call line
    sandbox_before_model(callback_context, llm_request, role="executor")
    trim_to_recent_turns(callback_context, llm_request, role="executor")
    strip_prior_turn_tool_parts(callback_context, llm_request)
    strip_source_metadata(callback_context, llm_request)
    tool_gate(callback_context, llm_request)
    round_limit(callback_context, llm_request)
    return synthesis_round_thinking(callback_context, llm_request)


executor_agent = Agent(
    # Model carries retry-with-backoff on 429/5xx (Dynamic Shared Quota); see
    # config.build_model. A string here would disable retries (ADK default).
    model=build_model(),
    name=AGENT_NAME,
    # InstructionProvider — appends today's date deterministically per request.
    # Wrapped for the sandbox: a run carrying state["sandbox"].executor_instruction
    # (or .executor_variant) swaps the prompt for that call only, with no redeploy;
    # without one, the base provider's text is returned unchanged.
    #
    # Then wrapped for tenancy, OUTSIDE the sandbox wrapper: one engine serves
    # more than one branded Chat app, and which surface a turn arrived through is
    # true of a sandbox run too. A turn with no state["tenant"] gets the inner
    # text unchanged (gub_agent/tenant.py).
    instruction=tenant_instruction(
        sandbox_instruction(
            with_current_date(EXECUTOR_INSTRUCTION + (RUNN_GUIDANCE if ASK_CONTRACT else "")),
            role="executor",
        ),
        # A tenant turn swaps the agency-wide persona and capability text for
        # the tenant persona (prompts/executor.py); a sandbox override wins.
        tenant_base=sandbox_instruction(
            with_current_date(TENANT_EXECUTOR_INSTRUCTION), role="executor"
        ),
    ),
    # Native thinking capped at MEDIUM. Unbounded (dynamic) thinking was the top
    # latency driver (thinking_tokens ↔ elapsed r=0.86): hard questions ran away
    # to 9-13k thought tokens / one 40s pause per step (pitch 70s @ 2 calls,
    # chevy 77s). MEDIUM keeps room to reason while cutting the runaway tail.
    # (EXECUTOR_THINKING_LEVEL, so the sandbox provenance can't drift from what
    # actually runs; a sandbox run overrides it per call in _before_model.)
    planner=build_thinking_planner(EXECUTOR_THINKING_LEVEL),
    # Runn is a source only on the /ask engine (config.ASK_CONTRACT); every
    # other engine is offered exactly today's tools.
    tools=ALL_TOOLS + (RUNN_TOOLS if ASK_CONTRACT else []),
    # Reset the per-pass round + tool budgets at the start of each executor pass
    # (so critic-requested retries aren't born over budget). See _before_agent.
    before_agent_callback=_before_agent,
    # Prune prior-turn tool payloads, withhold `find_files` off file-shaped
    # turns, cap ReAct rounds (context_pruning.py, tool_gate.py,
    # round_limiter.py) — the round cap forces synthesis instead of endless fan-out.
    before_model_callback=_before_model,
    # Dedupe repeated calls + per-turn tool budget (circuit_breaker.py) —
    # reliability guard against true loops / runaway fan-out.
    before_tool_callback=circuit_breaker,
    # Build the turn's citable evidence index from every tool result
    # (evidence_index.py) — what the formatter may cite and the format gate
    # grounds numbers/entities against.
    after_tool_callback=record_evidence,
)


# The DEEP path: wrap [executor → format-gate → critic-gate → escalator] in a
# LoopAgent (`build_deep_agent` below: the format gate and the critic now run
# side by side unless CRITIC_PARALLEL=0).
# The format gate (blend 03) turns the executor's prose into the typed
# AnswerPayload and enforces the contract in code — validation retries happen
# INSIDE the gate, never through this loop. The critic gate then judges
# information sufficiency only (its old shape axis moved into the format
# gate); it skips its LLM deterministically for abstentions — the bare
# NO_COMPANY_RECORDS marker, or an abstain payload after a tool call (one
# with no tool call is a draft from memory, and the gate itself sends it
# back once for a re-query). On clean answers the
# critic emits sufficient=true, escalator triggers loop exit after one
# iteration. On flagged failures, the executor runs again seeing the critic's
# feedback in session state. Capped at 2 iterations — and the second is not
# judged by the critic LLM, whose verdict there could not buy a third
# (agents/critic.py:_is_final_pass).
#
# sandbox_echo moved up to the root (below) with blend 04 — it must still
# precede any work, and the router is now the first thing that runs.
def build_deep_agent(
    executor: BaseAgent,
    gate: BaseAgent,
    critic: BaseAgent,
    escalator: BaseAgent,
    *,
    parallel: bool,
) -> LoopAgent:
    """The deep-path loop, serial or with the critic beside the format gate.

    Serial (`parallel=False`) is the tree as it was, piece for piece:
    [executor, format_gate, critic_gate(critic), loop_escalator].

    Parallel runs the critic LLM WHILE the formatter renders. The critic
    judges the executor's tool coverage and never reads the formatter's
    payload, so the only thing the serial order bought was the wait: over 155
    router-era deep turns, p50 2.8 s / p90 8.9 s per turn. What the critic
    gate decided from the payload — the abstain pass and the from-memory
    re-query — moves to a resolver after the join (agents/critic.py:
    CriticResolver), which reads THIS pass's payload and holds the critic's
    events until it has decided, so the stream and the state see what they
    saw before: payload, verdict, escalator. The join would also wait out a
    critic call whose verdict the resolver then drops (an abstain payload is
    settled in code), so the gate carries an after-callback that cancels that
    call (critic.py:settle_speculation).

    A function over the agents, not over the module singletons, so both
    shapes can be built in one process for the tests; the module builds one.
    """
    # Both gates read the payload off THIS pass's events by these authors, never
    # off state, which can still hold an earlier turn's (critic.py:_this_pass_payload).
    payload_authors = (gate.name, *(agent.name for agent in gate.sub_agents))
    if parallel:
        speculation = SpeculativeCritic(
            name="critic_speculation", sub_agents=[critic], payload_authors=payload_authors
        )
        # On the gate's own callback, not on the fork: it fires the moment the
        # gate has yielded its last payload, which is when the abandon is
        # decided. Idempotent — the builder can run again over the same gate —
        # and chained behind any callback the gate already has, never in place
        # of it. First in the chain, so an earlier callback's content cannot
        # skip it (a chain stops at the first truthy result). The gate is
        # shared with the fast path, where the callback finds no speculation
        # and does nothing; the serial wiring attaches none.
        existing = gate.after_agent_callback
        callbacks = (
            [] if existing is None else list(existing) if isinstance(existing, list) else [existing]
        )
        if settle_speculation not in callbacks:
            gate.after_agent_callback = [settle_speculation, *callbacks]
        middle: list[BaseAgent] = [
            ParallelAgent(name="format_and_critic", sub_agents=[gate, speculation]),
            CriticResolver(
                name="critic_gate",
                critic_name=critic.name,
                payload_authors=payload_authors,
            ),
        ]
    else:
        middle = [
            gate,
            CriticGate(name="critic_gate", sub_agents=[critic], payload_authors=payload_authors),
        ]
    return LoopAgent(
        name="gub_pipeline",
        sub_agents=[executor, *middle, escalator],
        max_iterations=2,
    )


deep_agent = build_deep_agent(
    executor_agent, format_gate, critic_agent, escalator_agent, parallel=CRITIC_PARALLEL
)

# The branch decision. `sub_agents` puts both destinations in the agent tree;
# ADK allows an agent one parent only (`base_agent.py:135-145`), which is why
# the fast path invokes the shared format_gate through `run_async` instead of
# adopting it (`agents/fast_path.py`).
dispatcher = Dispatcher(name="dispatcher", sub_agents=[fast_path, deep_agent])


# The root: provenance, then routing, then exactly one branch.
#
# sandbox_echo leads — on an experiment run it records what the run resolved
# to (model, thinking, temperature, prompt hashes) as a state_delta the
# caller's event stream carries. On an ordinary run it emits nothing at all,
# so the trace shape of a deep-path turn is today's.
#
# The router's JSON is authored `router`, which the bot's answer channel
# ignores by author (blend 03 step 0) — an engine deployed with this root in
# front of a bot WITHOUT that routing would put routing JSON in the user's
# bubble.
def build_root(
    echo: BaseAgent,
    router: BaseAgent,
    dispatch: Dispatcher,
    *,
    speculative: bool,
) -> SequentialAgent:
    """The root, with the deep path started beside the router or after it.

    Serial (`speculative=False`) is the tree as it was, piece for piece:
    [sandbox_echo, router, dispatcher]. Speculative wraps the last two in
    `SpeculativeDispatch`, which runs the router live and the dispatcher's
    deep agent at the same time on a fork of the session, then makes the
    dispatcher's decision and keeps the deep run only if that is the branch
    it picks (agents/speculation.py). Either way every agent below is the
    same object, and every event reaches the caller under its own author.

    A function over the agents, like `build_deep_agent`, so both shapes can be
    built in one process for the tests; the module builds one.
    """
    if speculative:
        middle: list[BaseAgent] = [
            SpeculativeDispatch(name="speculative_dispatch", sub_agents=[router, dispatch])
        ]
    else:
        middle = [router, dispatch]
    return SequentialAgent(name="gub_root", sub_agents=[echo, *middle])


root_agent: BaseAgent = build_root(
    sandbox_echo, router_agent, dispatcher, speculative=SPECULATIVE_DEEP
)
# The engine UB's /ask calls wraps the same tree in the /ask turn
# (config.ASK_CONTRACT, agents/ask.py); every other engine serves it as is.
if ASK_CONTRACT:
    root_agent = AskTurn(name="ask", sub_agents=[root_agent])

# Last, once the tree exists: the modules ADK would import on the first model
# request, imported now (config.EAGER_ADK_IMPORTS, gub_agent/eager_imports.py).
if EAGER_ADK_IMPORTS:
    from . import eager_imports

    eager_imports.import_now()

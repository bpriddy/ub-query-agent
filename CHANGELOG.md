# Changelog

All notable changes to this repository are tracked here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Security entries
describe what controls were added; operational detail that could narrow
an attacker's search space (which values lived where, timing of rotations,
remaining exposures) lives in internal notes instead of this public log.

### Changed (2026-09-08)

- **`debug_client/` moved** to `Anomaly-Technology/gub-sandbox-ui` (history
  preserved) and became the QA sandbox UI on Cloud Run behind IAP
  (`task-specs/sandbox-07`). A pointer README stays; `src/lib/sandbox.ts`
  there still mirrors `gub_agent/sandbox.py`.

## [Unreleased]

### Added (2026-11-21) — staffing capacity from Runn and Float

- **`runn_capacity(start, end, office?, role?, skill?, team?, person?)`**
  and **`float_capacity(...)`** (same interface, same answer): one call
  reads the planner in bulk through UB's relay (paged, at most 4 requests
  at a time) and works out each person's capacity (contract hours on their
  working days, less leave, rostered days off and public holidays),
  confirmed, tentative and free hours, with totals and whole name lists
  (on leave, over capacity, nothing booked, part-time). Office comes from
  UB staff joined on email, then on normalized name (flagged). UB staff
  with no planner record are listed as "no record", never as free. The
  shared arithmetic is in `gub_agent/tools/capacity.py`.
- **`runn_allocations`**: confirmed vs tentative, placeholders, role names,
  the person's working days, their leave, and a project summary (people
  and hours a day). A date range alone is enough. A UB email that Runn
  holds differently is matched by the name it spells, with a note.
- **Routing**: on the `/ask` engine, the fast path declines staffing
  questions about a named person (free, booked, on, leave, capacity), and
  short follow-ups to one, so they reach the deep path and the planner.
- **Prompt**: ask which person or project when ambiguous, state the window's
  working days and holidays, keep tentative work apart, never call someone
  with no planner record free, and use the capacity tools for totals.
- **Eval** (`tests/evals/runn_resourcing/`): the fake UB serves the new
  Runn routes and a Float relay; `--source float` runs 16 questions against
  the same agency in Float. Live run: 31.5/37 on Runn (9.5/37 before) and
  15/16 on Float.

### Fixed (2026-11-21)

- **An accepted rundown no longer crashes the `/ask` turn.** The format gate
  can pass a payload under the rundown's wider word budgets. `AskTurn` then
  re-validated it under the standard ones, and a 30-word bullet raised.

### Changed (2026-11-21): connect page through UB, UB's own OAuth client

- **The connect page is served through UB** (#93). This service stays
  private; UB forwards exactly `/connect`, `/connect/start`,
  `/connect/callback` and `POST /connect/disconnect`. `PUBLIC_BASE_URL` is now
  UB's public origin, with no path (the service refuses one at startup), and
  the redirect URI is `<it>/connect/callback`. A forwarded
  `X-UB-Connect-Origin` that differs from it answers
  `503 CONNECT_ORIGIN_MISMATCH`. URLs never come from Host or X-Forwarded-*.
- **UB's own Google OAuth client per environment**, not the chat bot's.
  `QUERY_SERVICE_CONNECT_CLIENT` (`ub`, `bot` as a prod-only fallback, or
  unset for none, the default) picks the secret names
  (`deployment/connect-client.sh`): `ub` mounts `ub-connect-google-client-id`
  and `ub-connect-google-client-secret`. Unset, nothing changes. The console
  steps are in `query_service/README.md`, "The OAuth client".

### Added (2026-11-21)

- **Similar-ideas tools** (UB D40): `similar_ideas(facets, …)` posts weighted
  facets to `POST /org/ideas/similar`; `similar_to_idea(idea_id, …)` reads
  `GET /org/ideas/{id}/similar` (`gub_agent/tools/similarity.py`, in
  `ALL_TOOLS`). Only the options the model sets are sent. Each result gains
  `matchedOn`, the text of its top evidence entry, so the match is a citable
  field row. Where UB answers 404 (`IDEA_SIMILARITY` off) or 503
  `EMBEDDINGS_NOT_CONFIGURED`, the result keeps its status and code, adds
  `similarity_unavailable: true` and tells the model to fall back to
  `list_ideas` and match by meaning; 409 `IDEA_NOT_EMBEDDED` says how to query
  the idea's facets instead. Other errors pass through unchanged. `/ask` maps
  both tools to `idea` rows with their own progress lines.
- **Prompts**: the executor (and tenant capabilities) send concept questions
  to `similar_ideas` first with weighted facets (low for the categorical, high
  for the specific), cite what each result matched on, and keep `list_ideas`
  for structured listing and as the fallback. The critic accepts
  `similar_ideas` as the retrieval for a concept question. Executor and critic
  pins in `test_tenant.py` re-pinned. The variants compose the baseline, so
  they pick this up.

### Changed (2026-11-18)

- **Abstention is confirmed before it stands** (#102). A pass that searched
  and abstains is sent back once, in code, to confirm "none" by a different
  route (`CriticGate._abstain_verdict`, `CONFIRM_ABSTENTION`, default on, 0 is
  the rollback). A turn whose only searches were `find_files` keeps today's
  pass (Drive file names are all that is indexed), as does the loop's last
  pass. The critic prompt holds a "not found" answer to the same rule. Critic
  pin in `test_tenant.py` re-pinned.

### Added (2026-09-04)

- **Per-call overrides — prompt, model, thinking level, temperature**
  (`gub_agent/sandbox.py`, new). One `sandbox` key in session state now steers a
  single run: `executor_instruction`/`critic_instruction` (or a variant name),
  `model` (allowlisted), `thinking_level`/`critic_thinking_level`,
  `temperature`, `critic_enabled`, `label`. Experiments no longer need
  `adk deploy agent_engine` (10-15 min per iteration into the engine that serves
  the Chat bot). Applied at three points: the agents' instruction providers, the
  `before_model_callback` chain (overwriting `llm_request.model` and
  `llm_request.config`), and `CriticGate` for `critic_enabled`. A new first
  sub-agent, `sandbox_echo`, emits the resolved config — model, thinking,
  temperature, prompt source, `sha256[:12]`, label — as a `sandbox_resolved`
  state delta in the caller's event stream.
  - Gated by `SANDBOX_ENABLED` (default **off**): with the flag off, or with no
    `sandbox` key, every override point is a no-op — nothing touched, nothing
    logged, no extra event in the trace. Pinned by tests, since prod runs this
    code path on every turn.
  - No silent fallbacks: an unknown key warns and is ignored, while an invalid
    value of a known key (model off `SANDBOX_MODEL_ALLOWLIST`, temperature
    outside 0.0-2.0, prompt over 64 KB, unknown variant name) fails the run —
    an A/B that quietly ran the baseline in both arms is worse than one that
    didn't run.
  - Documented that ADK **skips** `{var}` state injection for callable
    instructions (`bypass_state_injection=True`, ADK 2.6.1), so a prompt
    arriving from session state may contain literal braces. The README's and
    `prompts/__init__.py`'s brace-trap notes were overstated and now name the
    condition: the trap applies to plain-string instructions only.

### Performance (2026-08-12)

- **Agent latency cut ~30%** (mean 44s → 31s, worst case 79s → 46s; prompt
  tokens 3.05M → 1.16M) via three measured levers, benchmarked on a prod-DB copy:
  - **Round-level cutoff** (`gub_agent/agents/round_limiter.py`, new) — caps
    ReAct at `MAX_ROUNDS` gather rounds, then strips the tool declarations so the
    model must synthesize instead of fanning out endlessly.
  - **Thinking budget MEDIUM** (`agent.py`) — the executor's unbounded dynamic
    thinking was the top latency driver (thinking_tokens ↔ elapsed r=0.86);
    MEDIUM cuts the runaway tail. Critic already runs LOW.
  - **`_sources` observation masking** (`gub_agent/agents/context_pruning.py`) —
    strips Drive citation plumbing (128k tokens on one account overview, re-sent
    every round) that the prompts already tell the model to ignore. Copy-on-write,
    so the session-shared payload is never mutated in place.
- **Circuit breaker** (`gub_agent/agents/circuit_breaker.py`, new) — per-executor-
  pass tool-call budget + identical-call dedup; a reliability guard against
  runaway fan-out / loops. Both the round and tool budgets reset at the start of
  each executor pass (`before_agent_callback`), so a critic-requested retry
  starts with a fresh budget rather than inheriting the previous pass's count.

### Added (2026-06-15)

- **Multi-agent pipeline** (`gub_agent/agents/critic.py`, `agent.py`). The
  agent is now `LoopAgent([executor, critic, escalator], max_iterations=2)` —
  a minimal critic-before-commit borrowing from the Agentic-RAG pattern. The
  critic emits a two-axis `CriticVerdict` (`info_sufficient` = did we retrieve
  enough; `answer_satisfies` = right kind of closure, grounded, recent),
  gating a retry. Same engine id; callers unaffected.
- **`org_query` tool** (`gub_agent/tools/org_query.py`) — structured
  query primitive (filter/sort/group/aggregate/`similar_to`) preferred for
  all filter/count/sort/aggregate questions.
- **Deterministic current-date injection** (`gub_agent/instruction_utils.py`).
  Executor + critic use an ADK `InstructionProvider` (callable instruction)
  that appends today's UTC date per request — no LLM-mediated tool, never
  stale — so recency reasoning ("this week") is reliable.
- **Synthesis principles** in the executor prompt: author to the question's
  intended *closure* (fact → value, assessment → verdict, exploratory →
  shortlist), ground every named entity, weight recency. General, not tuned
  to specific phrasings.
- **`debug_client/`** — local-only Next.js tool for inspecting the agent's
  decomposition trace (tool calls, two-axis critic verdict, sources).

### Added (2026-04-23)

- `README.md` — project overview, local setup, deploy pointers. The
  repo had no README before.

### Security (2026-04-23)

- Added a `gitleaks` pre-commit hook (`.githooks/pre-commit`) that scans
  staged changes before every commit and rejects the commit on any
  finding. One-time setup per clone is documented in the README
  ("Local setup").
- Tightened `.gitignore` coverage to block additional secret-file shapes
  from slipping through (additional `.env.*` variants, typical key/cert
  file extensions, common service-account JSON name patterns).

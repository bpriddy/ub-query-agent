# gub-agent

Python ADK agent that exposes GUB (gcp-universal-backend) data to Gemini
Enterprise / Agentspace. Users talk to Gemini Enterprise; Gemini invokes this
agent's tools; the agent calls GUB's HTTP API on the user's behalf.

Sibling repos: [gcp-universal-backend](https://github.com/bpriddy/gcp-universal-backend),
[gub-admin](https://github.com/bpriddy/gub-admin),
[gub-review](https://github.com/bpriddy/gub-review).

## Status (2026-07-17)

Strategic position: the **gchat bot (gub-gchat-bot) is the user entry point**,
blending GUB answers with federated Workspace results. This agent is the
bot's GUB answerer — the scope/abstention design (personal-Workspace
questions answer exactly `NO_COMPANY_RECORDS` so the bot suppresses the GUB
section) exists for that architecture. The Gemini-Enterprise-ambient
direction (GUB as an MCP connector under native Gemini) was explored,
technically validated, and set aside after team reception; the `/mcp`
surface in gcp-universal-backend remains deployed but is not the priority.

Two open threads, both needing one live Gemini Enterprise session to close:

1. **OAuth token injection** (`gub_agent/tools/_client.py`): three candidate
   state-key patterns are probed in order (direct `to_dict()[auth_id]`,
   nested `auth_tokens`, `temp:` prefix) because the ADK/Gemini Enterprise
   state-key format was still being pinned down. The winning pattern hasn't
   been confirmed live — once it is, keep it and delete the other two.
2. **Authorization ID**: the registered Gemini Enterprise authorization is
   `gub-oauth-3` (re-registered during the same debugging; earlier ids are
   dead). A mismatch between `GUB_AUTHORIZATION_ID` and the registration
   fails token injection silently.

## Stack

- Python 3.11+ with [Google ADK](https://github.com/google/adk-python)
- Deployed to Vertex AI Agent Engine
- Registered with a Gemini Enterprise app for end-user chat

## Architecture

`root_agent` is a small multi-agent pipeline, not a single LLM. The question is
ROUTED first (blend 04) and only then answered:

```
SequentialAgent("gub_root")
  ├─ sandbox_echo — resolved experiment config, silent on ordinary runs
  │                 (gub_agent/sandbox.py)
  ├─ router       — one LLM call, no tools, typed RouterDecision
  │                 (gub_agent/agents/router.py)
  └─ dispatcher   — picks ONE branch, in code (gub_agent/agents/dispatcher.py):
                    · workspace_personal → abstain  (0 model, 0 tool calls)
                    · smalltalk          → template (0 model, 0 tool calls)
                    · low confidence     → ask which question was meant
                    · a FACT intent with a known entity → fast_path
                    · everything else    → deep_agent

fast_path (gub_agent/agents/fast_path.py)
  one deterministic tool call → the evidence index → format_gate. No executor,
  no critic; an ambiguous entity or an empty result falls through to the deep
  path once, a 403/404 on a record it looked up itself is answered immediately.
  An id the chat client supplied is fetched only when its prefix says the id is
  a campaign, and a refusal about it goes to the deep path as well.

deep_agent = LoopAgent("gub_pipeline", max_iterations=2)
  ├─ executor    — the tool-using LLM (gub_agent/agent.py)
  ├─ format_and_critic — ParallelAgent: the two below run at once
  │   ├─ format_gate — renders the typed AnswerPayload and enforces the
  │   │                answer contract in code (gub_agent/agents/format_gate.py)
  │   └─ critic      — evaluates information sufficiency; its verdict is held
  │                    (gub_agent/agents/critic.py)
  ├─ critic_gate — decides after the join: deterministic passes and the
  │                from-memory re-query on THIS pass's payload, else the
  │                critic's verdict
  └─ escalator   — exits the loop early when the critic is satisfied
```

The critic LLM never runs on the loop's LAST iteration: its verdict there
cannot buy another pass, so `critic_gate` writes a sufficient one in code,
authored `critic` like the LLM's (`_is_final_pass` in
`gub_agent/agents/critic.py`). On a retried turn that is 2.8-20.7 s saved.
`CRITIC_SKIP_FINAL_PASS=0` is its rollback: the critic judges the retry again.

`CRITIC_PARALLEL=0` restores the serial order — format_gate, then critic_gate
running the critic — with no code change (`build_deep_agent` in
`gub_agent/agent.py`). It restores the tree only: the last-pass skip keeps its
own switch, and both gates read the abstain payload off THIS pass's events,
never a stale one left in state. The critic never reads the formatter's
payload, so waiting for it only cost time: p50 2.8 s / p90 8.9 s per deep turn.

One difference beyond timing: running on its own ADK branch, the critic LLM
does not see this pass's formatter JSON — nor the formatter/format_gate
payloads of any earlier deep turn run in the parallel wiring, because the
branch name repeats every turn (the executor's prose for those turns is still
there). Its request is not the serial one minus one payload, and whether its
verdicts match the serial critic's is unmeasured
(`SpeculativeCritic` in `gub_agent/agents/critic.py`).

The deep path is the Agentic-RAG "critic-before-commit" pattern: on a clean
answer the loop exits after one pass; on a flagged answer the executor runs
again, sees the critic's feedback in session state, and fixes it. We keep just
this one specialist (critic), not a planner/rewriter/fanout fleet — at our
scale the critic is the single piece that materially improves dependability.

What routing adds is the option to skip it. Latency is model turns (thinking
tokens ↔ elapsed, r=0.86), so a fact question with a known entity — one HTTP
call's worth of information — is answered with two model calls (router,
formatter) instead of 1-3 executor rounds plus a critic pass.

Those two run with thinking OFF (`thinking_budget=0`): one classifies, the
other renders given text into a given schema. At `thinking_level=LOW` the
router's TTFT p90 was 3.2 s against 1.1 s off, for the same intent on 19 of 20
replayed production requests. `ROUTER_THINKING_OFF=0` / `FORMATTER_THINKING_OFF=0`
put either back on LOW (`gub_agent/config.py`).

The engine id, the `stream_query` shape and the author-routed answer channel
are unchanged: callers see no difference at the boundary.

### Tools

- `org_query` — structured query primitive (filter / sort / group / aggregate
  / `similar_to` trigram) against GUB's `POST /org/query`. THE preferred tool
  for any filter/count/sort/aggregate question. Lightweight: returns only
  catalog fields, never large text blobs. (`gub_agent/tools/org_query.py`)
- `find_staff_for_resourcing`, `search_staff`, `get_staff_profile` — staff.
- `list_accounts`, `get_account_overview`, `get_campaign` — detail tools.
  These return the rich per-entity fields (e.g. `statusMarkdown`); `org_query`
  is for *finding* entities, the detail tools for *reading* one.
- `find`, `get_piece`, `list_ideas`, `get_idea` — name discovery, pieces and
  the ideas memory. `list_ideas` is for structured listing (an account's, a
  campaign's or a period's ideas). (`gub_agent/tools/discovery.py`)
- `similar_ideas`, `similar_to_idea` — ideas by meaning (UB D40,
  `POST /org/ideas/similar`, `GET /org/ideas/{id}/similar`). The model breaks a
  concept question into weighted facets: low weight for the categorical ("an
  Instagram campaign", 0.2), high for the specific ("a huge grape rolls down the
  street", 1.0). Names are off unless the user names an idea; `scoring` is
  UB's default (`blend`) unless the question is about one element
  (`strongest`) or overall likeness (`cumulative`). Each result carries UB's
  per-facet `evidence` and a `matchedOn` (the match that counted most), which
  the answer cites. Where UB has the feature off (404) or no embedding driver
  (503 `EMBEDDINGS_NOT_CONFIGURED`) the tool result tells the model to fall back
  to `list_ideas` and match by meaning. (`gub_agent/tools/similarity.py`)

### Synthesis principles (how the agent decides what to say)

These are general — phrased to scale across question types, never tuned to a
specific phrasing. See `gub_agent/agent.py` (`SYSTEM_INSTRUCTION`) and
`gub_agent/agents/critic.py`.

- **Author to the question's intended CLOSURE, not its literal words.** A
  question seeking a FACT wants the value ("47"); an ASSESSMENT ("how is X?",
  "should we worry about X?") wants a VERDICT up front plus the few drivers
  that earn it — never a catalog; an EXPLORATORY question wants a shaped
  shortlist. "Technically answered" is a near-zero bar ("Active." technically
  answers "how is the chevy account?"); the real bar is serving the asker's
  intent. Length follows closure, not data volume.
- **Verdicts must be earned** — grounded in retrieved signals, like any claim.
- **Ground every entity.** Never name an account/campaign/person that didn't
  appear in a tool result this turn — no inventing, no "helpful completion".
- **Time is part of relevance.** Weight recent and currently-active items;
  don't surface something long-finished as if it were current.

### The critic — two-axis evaluation

`CriticVerdict` reports two independent judgments, gating `sufficient`:

1. `info_sufficient` — did the tool calls gather enough to answer this
   question? (right tools used, every entity queried, multi-part questions
   decomposed, not answered with zero tool calls).
2. `answer_satisfies` — does the answer deliver the right KIND of closure,
   grounded, recency-respected? (a catalog answer to "how is X" fails here).

`sufficient = info_sufficient AND answer_satisfies`. False triggers a retry.

### Deterministic current-date injection

Gemini doesn't reliably know "now", and a `get_current_date` tool would be
LLM-mediated (the model has to choose to call it). Instead, both the executor
and critic use an ADK **`InstructionProvider`** (a *callable* instruction,
`gub_agent/instruction_utils.py:with_current_date`) that appends
"Today is YYYY-MM-DD (UTC)" on **every** request — computed server-side,
deterministic, never stale, no tool, no caller coupling. This is what makes
"recent" / "this week" reliable.

> **ADK brace trap — and why a callable instruction escapes it:** ADK runs
> `{var}` session-state injection over instruction strings, and any literal
> `{anything}` that isn't a real state key crashes the run with `KeyError`
> before the LLM is called — visible only in the Reasoning Engine logs.
> That pass is **skipped for callable instructions**: `canonical_instruction`
> returns `bypass_state_injection=True` for an InstructionProvider
> (`llm_agent.py:692-698`), so `inject_session_state` never runs over its output
> (`flows/llm_flows/instructions.py:51-59`, ADK 2.6.1 — verified). Both agents
> here use callables, so braces in `prompts/` are in fact safe today, and a
> prompt arriving from session state (see the sandbox below) may contain them.
> The trap is real the moment an `instruction=` becomes a plain string — keep
> them callables. (f-string interpolations like `f"{base}"` were never at risk:
> they don't emit literal braces.)

### Sandbox — per-call prompt / model / thinking / temperature

Both agents are built at module import, so changing a prompt, a model or a
thinking level used to mean a code edit plus `adk deploy agent_engine`: 10-15
minutes per iteration, into the engine that serves the Chat bot.
`gub_agent/sandbox.py` moves those knobs into one session-state key, read fresh
on every call:

```python
create_session(state={"sandbox": {
    "model": "gemini-2.5-pro",          # allowlist: SANDBOX_MODEL_ALLOWLIST
    "thinking_level": "DYNAMIC",        # MINIMAL|LOW|MEDIUM|HIGH|DYNAMIC|OFF — named
    "critic_thinking_level": "DYNAMIC", #   levels are 3-series only; a 2.5 model
                                        #   needs DYNAMIC on every role (enforced);
                                        #   OFF = thinking_budget=0
    "temperature": 0.2,
    "executor_instruction": "<full prompt text>",   # or executor_variant
    "critic_instruction": "<full prompt text>",     # or critic_variant
    "critic_enabled": False,            # measure the executor alone
    "label": "concise-v2",              # free-form, provenance only
}})
```

Three application points: the instruction providers (prompt), the
`before_model_callback` chain (model / thinking / temperature, by overwriting
`llm_request.model` and `llm_request.config`) and `CriticGate` (`critic_enabled`).
`sandbox_echo` — the pipeline's first sub-agent — emits the resolved config
(model, thinking, temperature, prompt source, `sha256[:12]`, label) as a
`sandbox_resolved` state delta the caller's event stream carries.

Two rules the design keeps:

- **`SANDBOX_ENABLED=false` (the default, and prod) makes the whole thing
  inert** — a `sandbox` key is ignored with no warn and no error, and no extra
  event enters the trace. Set the flag only on the sandbox engine.
- **No silent fallbacks.** An unknown key warns and is ignored; an invalid value
  of a known key (model off the allowlist, temperature outside 0.0-2.0, prompt
  over 64 KB, unknown variant name) fails the run. An A/B that quietly ran the
  baseline in both arms is worse than one that didn't run.

#### Prompt variants

`executor_variant` / `critic_variant` name an entry in
`gub_agent/prompts/variants/` — Python modules, like the baseline prompts, so
`adk deploy` cannot drop them from the bundle. `resolve_variant(name, role)`
returns the text or raises listing what exists; `baseline` resolves to the
**live** `EXECUTOR_INSTRUCTION` / `CRITIC_INSTRUCTION` object, not a copy, so
the baseline arm of an A/B can never drift from production.

| role | name | what it tests |
|---|---|---|
| executor | `baseline` | the deployed prompt, byte-identical |
| executor | `v2_concise` | hard word budgets per closure type (answers run long) |
| executor | `v3_grounding` | grounding as a procedure: pre-write check, field-based attribution, audit pass (the agent invents facts) |
| critic | `baseline` | the deployed critic prompt |

The two starter variants are `EXECUTOR_INSTRUCTION` **plus one appended block**
rather than standalone rewrites — one variable under test, and the tool
documentation cannot drift from the baseline. A variant that wins is promoted
into `prompts/executor.py` by hand, in an ordinary PR; nothing does that
automatically.

## Local setup

```bash
git clone git@github.com:bpriddy/gub-agent.git
cd gub-agent

# Python env
python -m venv .venv && source .venv/bin/activate
pip install -r gub_agent/requirements.txt

# Local config
cp .env.example .env
# then edit .env and fill in GUB_BASE_URL, GUB_SERVICE_JWT, etc.

# Secret-scan pre-commit hook (required — refuses commit on detected
# API keys, tokens, JSON keys, etc.)
brew install gitleaks        # or see https://github.com/gitleaks/gitleaks#installation
git config core.hooksPath .githooks
```

## Run

```bash
# Start the agent locally against a running GUB backend (localhost:3000)
python -m gub_agent
```

## Deploy

CI deploys only by a manual run of **Actions → Deploy** on `main`. The run
targets GitHub Environment `prod` and waits for bpriddy's approval (D37).

The agent deploys to **Vertex AI Agent Engine via `adk deploy agent_engine`**
— NOT Cloud Build. It will never appear in `gcloud builds`. Update in place
(same engine id, so callers — the Chat bot, Agentspace — see no change):

```bash
adk deploy agent_engine \
  --project=os-test-491819 \
  --region=us-central1 \
  --agent_engine_id=9136379226620952576 \
  --display_name=gub-agent \
  --description="<what changed>" \
  gub_agent
```

The positional `gub_agent` (the package exporting `root_agent`) is required.
Agent names must be valid Python identifiers — no dashes (e.g. `gub_pipeline`,
not `gub-pipeline`), or ADK rejects the deploy. A transient `code 13
INTERNAL` from Agent Engine usually succeeds on a retry.

**Model endpoint vs engine region.** The model (`gemini-3.5-flash`) is served
only from the Vertex **global** endpoint, so `.env` sets
`GOOGLE_CLOUD_LOCATION=global` (the regional endpoint 404s the model). That is
independent of `--region=us-central1` above, which controls only where the
Agent Engine *resource* lives — `--region` overrides `GOOGLE_CLOUD_LOCATION`
for the deploy target, while the baked runtime env keeps model calls on
`global`. Do **not** pass `--region=global` (Agent Engine isn't deployable
there). Callers (gub-sandbox-ui, gchat bot) keep `GCP_REGION=us-central1` —
they address the regional engine resource, not the model.

Then `register_agent.py` (see `deployment/`) registers the deployed engine
with the Gemini Enterprise app — a separate, occasional step.

### Per-environment config (D37, #105)

No UB URL, project or engine id is a code default any more.

- **Engines.** `GUB_BASE_URL` is required by `gub_agent/config.py` and set in
  each `deploy-*.env`. An engine without it refuses to start.
- **Engine deploy (`deploy.yml`).** `GCP_PROJECT_ID`, `GCP_REGION`,
  `AGENT_ENGINE_ID` and `AGENT_STAGING_BUCKET` are GitHub Environment `prod`
  variables (repo-level copies stay until every workflow reads the
  Environment).
- **Sandbox scripts.** `deploy-sandbox.sh` and `teardown-sandbox.sh` read the
  production engine id from Environment `prod` (`AGENT_ENGINE_ID`) through
  `gh` (`deployment/github-env.sh`), or from `PROD_AGENT_ENGINE_ID`. They stop
  if neither resolves.
- **Query service.** `.github/workflows/deploy-query-service.yml` (manual,
  `environment` input) reads `QUERY_SERVICE_NAME`, `QUERY_SERVICE_SA`,
  `QUERY_SERVICE_ENGINE_ID`, `QUERY_SERVICE_DB_NAME`,
  `QUERY_SERVICE_GRANT_KMS_KEY`, `UB_BASE_URL`, `CLOUD_SQL_INSTANCE` and the
  `GCP_*` variables from that Environment, builds once and deploys by digest.
  It has not been run yet (see its header for the IAM it needs).
  `deployment/deploy-service.sh <env>` reads the same variables and stays as
  the fallback until the workflow's first good run.
- **Still manual:** `deployment/oauth-relay/` is deployed by hand.

### Debug client → gub-sandbox-ui

The debug client — trace inspection, the sandbox config panel and A/B
compare (#32), the batch runner (#33) — moved to its own repository,
**Anomaly-Technology/gub-sandbox-ui**, where it runs as the QA sandbox UI on
Cloud Run behind Cloud IAP (`task-specs/sandbox-07`). `debug_client/README.md`
here is a pointer. One seam crosses the repos: its `src/lib/sandbox.ts`
mirrors `gub_agent/sandbox.py` — change the contract in both.

## Sandbox engine

Until the sandbox epic there was exactly one Agent Engine — the
`9136379226620952576` above — and the Chat bot, the debug client and the
Gemini Enterprise registration all point at it. Every prompt or model
experiment therefore deployed into production. The sandbox engine is a
**second engine from the same package**; the only difference is the env file
baked in at deploy time, and that difference is what isolates experiments from
live users:

| env file | baked into | `SANDBOX_ENABLED` | `EMIT_THINKING` |
|---|---|---|---|
| `deploy-prod.env` | the **production** engine (`deploy.yml` passes it as an absolute `--env_file`) — renamed from `deploy-dev.env`, whose name predated the split | `0`, explicit | `1` |
| `deploy-sandbox.env` | the sandbox engine (`deployment/deploy-sandbox.sh`) | `1` | `1` |

With the flag off, `state["sandbox"]` is ignored outright — no warn, no error,
no extra event — so an override aimed at an experiment can never take effect on
the engine serving the Chat bot (see "Sandbox — per-call …" above). CI pins
this: `tests/unit/test_deploy_env_isolation.py` reads the file `deploy.yml`
deploys and fails the build if it enables the sandbox.

### Deploy

```bash
pip install -e ".[deploy]"                  # google-cloud-aiplatform; not in the base deps
deployment/deploy-sandbox.sh "what changed"  # description is optional
```

The first run has no `SANDBOX_AGENT_ENGINE_ID` and **creates** a new engine;
the script prints the ID and the line to append to `deploy-sandbox.env`. Record
it — every later run then updates that engine in place (`--agent_engine_id`)
instead of creating, and billing, another one. The script refuses to run if
`deploy-sandbox.env` does not enable the sandbox, and refuses to address the
production ID however it got there. It never calls `register_agent.py`: the
sandbox engine must not be registered in Gemini Enterprise, or real users would
be routed to whatever prompt an experiment left behind.

Deploy takes ~10 minutes. `adk deploy` works in a `gub_agent_tmp<timestamp>/`
folder under the repo root (gitignored) and removes it afterwards.

### Verify you are addressing the sandbox, not prod

The two engines run the same code, so a wrong `AGENT_ENGINE_ID` in a caller is
invisible from the answers. Check the ID, not the behaviour:

```bash
# 1. The engine your caller targets must NOT be 9136379226620952576.
grep SANDBOX_AGENT_ENGINE_ID deploy-sandbox.env

# 2. Both engines, with the env each one was deployed with.
TOKEN=$(gcloud auth print-access-token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/os-test-491819/locations/us-central1/reasoningEngines" \
  | python3 -c 'import json,sys
for e in json.load(sys.stdin)["reasoningEngines"]:
    env = {v["name"]: v.get("value") for v in e["spec"].get("deploymentSpec", {}).get("env", [])}
    print(e["name"].rsplit("/", 1)[1], e["displayName"], env)'

# 3. Not registered: the Gemini Enterprise list shows only gub-agent.
python deployment/register_agent.py --list
```

A sandbox run also proves itself from the inside: with `SANDBOX_ENABLED` on and
a non-empty `state["sandbox"]`, the first event of the run carries a
`sandbox_resolved` state delta. The prod engine never emits one.

Live smoke (2026-09-07, four verdicts, all held): sandbox + `state.sandbox` →
`sandbox_resolved` with the requested model/thinking/label, then the full
executor → critic → escalator run; sandbox without the key → no echo event;
**prod + the same `state.sandbox` → no echo, ordinary answer** (inert by deploy
flag); unknown variant name → the run fails.

**How a failed sandbox run looks from the outside.** Agent Engine does not turn
an exception inside the run into an error event: the caller gets **HTTP 200 and
an empty (or truncated) stream**. Both an invalid override (unknown variant,
model off the allowlist) and a model the project cannot serve look identical
from the client — `sandbox_echo` may have fired, then nothing. Treat "no
executor event" as a failure and read the reason from the engine logs:

```bash
# reason for the last failed sandbox run (ValueError / genai ClientError)
gcloud logging read 'resource.type="aiplatform.googleapis.com/ReasoningEngine"
  AND resource.labels.reasoning_engine_id="<SANDBOX_AGENT_ENGINE_ID>"
  AND severity>=ERROR' --limit=5 --freshness=30m --format='value(textPayload)'
```

Two things the logs will show that are not bugs in this repo: every completed
stream on an ADK 2.6.1 engine is followed by `RuntimeError: coroutine raised
StopIteration` from `google/adk/cli/fast_api.py` (end-of-stream artifact of the
api_server template; the client already has the full stream; the older prod
build does not log it), and `gemini-3.5-pro` (like every `gemini-3-pro*` id and
the non-preview `gemini-3-flash`) returns 404 from this project — they are not
in the allowlist for that reason. Probed 2026-09-09 from `global`:
`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3-flash-preview`,
`gemini-2.5-pro`, `gemini-2.5-flash` and `gemini-2.5-flash-lite` answer; that is
the Gemini half of the sandbox allowlist (`deploy-sandbox.env`).

**Claude arms (Anthropic on Vertex).** *Temporarily out of the sandbox allowlist since 2026-09-09 (`deploy-sandbox.env`, user decision: a model that can only 404 confuses testers) until the models are enabled for the project in Model Garden; the router and its tests stay.* A sandbox run may also select a
`claude-*` id (`claude-sonnet-5`, `claude-haiku-4-5`). `gub_agent/models.py`
wraps the Gemini client in a `VendorRouter`: the id the sandbox wrote into
`llm_request.model` decides whether the request goes to Gemini or to ADK's
`Claude` client (Anthropic on Vertex, location `CLAUDE_VERTEX_LOCATION`,
default `global`). Three things differ for a Claude arm, all handled in the
router: Anthropic has no named thinking levels, so the validator's DYNAMIC rule
applies (DYNAMIC = adaptive thinking); the critic's `output_schema` is restated
as an instruction because the Anthropic path ignores Gemini's native JSON mode;
and `temperature` is inert while thinking is on (Anthropic rejects sampling
parameters there) — logged, not silently applied. Prerequisites: the Anthropic
models must be **enabled for the project in Model Garden** (a console step —
until then Vertex answers 404 "does not have access" and the run is an empty
stream), and `anthropic[vertex]` is in `gub_agent/requirements.txt`. Prod is
untouched: with `SANDBOX_ENABLED=0` the sandbox never writes the model field.

One more live-verified trap, now caught up front: a **named `thinking_level`
is a 3-series knob**. `gemini-2.5-pro` rejects it with 400, and the baseline
planners pin MEDIUM / LOW, and thinking off (`OFF`, `thinking_budget=0`) for
the router and the formatter, which 2.5-pro cannot do either — so
`{"model": "gemini-2.5-pro"}` alone would die silently. `read_overrides`
refuses such a run unless `thinking_level`, `formatter_thinking_level`,
`router_thinking_level` (and `critic_thinking_level`, while the critic is on)
are `DYNAMIC`; the set of models that do accept named levels is
`SANDBOX_THINKING_LEVEL_MODELS`. The provenance reports `OFF` for an
untouched router or formatter on an engine with the flags on.

### Billing

Agent Engine bills per deployed engine for as long as it exists, whether or not
it serves traffic. Tear the sandbox down when an experiment series is over and
recreate it from the current commit when the next one starts — the deploy is
reproducible, the engine holds no state worth keeping.

### Tear down

```bash
deployment/teardown-sandbox.sh           # ID from deploy-sandbox.env; asks for confirmation
deployment/teardown-sandbox.sh <id>      # explicit
```

Shows the engine's display name before deleting, refuses the production ID, and
deletes with `force=true` (sessions the engine owns go with it). There is no
`gcloud` surface for reasoning engines, so it calls the Vertex REST API directly.
Afterwards blank `SANDBOX_AGENT_ENGINE_ID` in `deploy-sandbox.env`, or the next
deploy fails on a 404 instead of creating a fresh engine.

### Known deviations from the deploy section above

- **A relative `--env_file` path is silently ignored.** `adk deploy`
  `chdir()`s into its `gub_agent_tmp…/` folder *before* it reads the env file
  (ADK 2.6.1 `cli_deploy.py:1000` vs `:1094`), so a relative
  `--env_file=deploy-prod.env` is looked up in the wrong directory, found
  missing, and skipped with no message — the engine comes up with **no env at
  all**. This is why the production engine's `deploymentSpec` was empty for a
  month: the `EMIT_THINKING=1` the Deploy workflow passed from 2026-08-10
  never reached it, and prod ran entirely on `config.py` defaults.

  **Fixed:** both deploys now pass an absolute path (`deploy.yml` via
  `$GITHUB_WORKSPACE`, `deploy-sandbox.sh` via `$REPO_ROOT`) and both read the
  env back off the deployed resource afterwards
  (`deployment/verify-engine-env.sh`), failing the deploy if a key never
  landed. CI pins the absolute path
  (`tests/unit/test_deploy_env_isolation.py`).

  Because the fix means the file's values now actually reach production,
  `EMIT_THINKING` was pinned to `0` in `deploy-prod.env` — production keeps the
  behavior it has had all along. Flipping it to `1` is a deliberate decision:
  the Chat bot filters thought parts (`src/agent/client.ts:336`) so chat users
  would see nothing new, but the summaries would reach every other consumer of
  the engine, Gemini Enterprise included, and enlarge every event payload.
- **`--env_file` is deprecated** in ADK 2.6.1 (it warns and still works — it
  populates the deploy's `env_vars`). Its successor is an `env_vars` block in
  an `.agent_engine_config.json` passed via `--agent_engine_config_file`.
  Whichever flag, keep the env in a **per-deploy file outside `gub_agent/`**:
  a config file committed inside the package is picked up by default by
  *every* deploy, including production.

### End-to-end: every allowed model, live

`tests/e2e` runs the deployed sandbox engine for real — a session with the
shared sandbox subject's GUB JWT and `state.sandbox`, one reference question,
every model on the engine's **own** allowlist (read from its env, so the
parametrisation follows the deploy). Per model it asserts the two things a QA
run must be able to trust: `sandbox_resolved.model` equals the model that was
asked for, and the executor produced visible text (an exception inside the
engine is an empty 200 stream, so "no text" is how a broken id looks from
outside). Two guard rails: a baseline run emits no provenance, and a named
thinking level on a DYNAMIC-only model is refused before any model call.
Claude cases skip until Anthropic is enabled for the project; models known
to be servable but too weak to finish the question are non-strict xfails
(`SANDBOX_E2E_XFAIL_MODELS`, default `gemini-2.5-flash-lite`).

Opt-in — it costs tokens (~$0.05–0.15 per model) and minutes; CI and the hooks
skip it:

```bash
SANDBOX_E2E=1 .venv/bin/pytest tests/e2e -q -rsxX
```

Needs ADC with `roles/aiplatform.user` and `roles/iam.serviceAccountTokenCreator`
on `sa-gub-sandbox` (terraform `sandbox_operators`). Run it after a sandbox
redeploy or an allowlist change.

## Security

Credentials live in `.env` locally and in GCP Secret Manager in deployed
environments. Never commit `.env`, service account JSON, or any file under
`secrets/` / `keys/` — the gitleaks pre-commit hook above is the second line
of defense after `.gitignore`.

## Secrets & rotation

Documents secrets/credentials this service uses and how to rotate them.
For company-wide incident response (escalation, post-mortem, comms), see
IT's canonical incident-response doc. This section covers system-specific
actions only.

This service is unusual in two ways:

- It runs on **Vertex AI Agent Engine**, not Cloud Run. Deploys go through
  `adk deploy agent_engine` (see `deployment/register_agent.py`), and
  Cloud Run's `--set-secrets` plumbing does not apply — Agent Engine
  injects environment via its own mechanism.
- It depends on a **separate OAuth relay** (`deployment/oauth-relay/`,
  deployed as a Cloud Function) that bridges Gemini Enterprise's OAuth
  flow with GUB's identity API. The relay has its own configuration
  surface, also documented below.

### Inventory

| Credential | Where it lives | Issued by | Used for |
|---|---|---|---|
| `GUB_SERVICE_JWT` | Agent Engine env (set during `adk deploy`); locally `.env` | Self-issued by GUB (re-use a frontend-issued access token, or generate via `scripts/test-broker-flow.mjs`) | **Local dev / CI fallback only.** Authenticated GUB API calls when the OAuth flow isn't active (e.g., direct Python REPL testing). Production traffic uses Gemini Enterprise's OAuth flow with per-user tokens — this JWT is not on the runtime hot path. |
| `GUB_AUTHORIZATION_ID` | Agent Engine env; locally `.env` | Gemini Enterprise registration | OAuth flow identity matching. **Configuration, not a secret** — it's just a string that must match exactly between the Gemini Enterprise console registration and the agent's env. |
| `GEMINI_APP_ID` | Agent Engine env; locally `.env` | Gemini Enterprise app registration | Targets the correct Gemini Enterprise app during `register_agent.py`. **Configuration, not a secret.** |
| `AGENT_ENGINE_ID` | `.env` after deploy | Output of `adk deploy agent_engine` | Pins the running agent revision so `register_agent.py` can register it. **Configuration**, populated post-deploy. |
| OAuth relay deployment config | Cloud Function env (in `deployment/oauth-relay/`) | GCP Cloud Function deploy | Redirect URL whitelist + GUB endpoint. |
| GCP service account key | None — uses Application Default Credentials (`gcloud auth application-default login` locally; the Agent Engine runtime SA in deploys) | GCP | All GCP API calls |

### Rotation procedures

#### `GUB_SERVICE_JWT`

This is a JWT issued by GUB. Rotation = re-issue a new one. The old JWT
remains technically valid until its `exp` (default 15 min for access
tokens), but is hardly used in production paths so the urgency is low.

**Preconditions.** None.

**Steps.**
1. Issue a fresh JWT. Two options:
   - Sign in via the gub-admin frontend with a service account user, then
     copy the access token from the browser's network tab.
   - Run `node scripts/test-broker-flow.mjs` from the gcp-universal-backend
     repo (the broker test flow returns a fresh JWT).
2. Update Agent Engine's env. Because Agent Engine doesn't expose a
   per-revision env override flag like Cloud Run, the cleanest path is
   to redeploy:
   ```bash
   adk deploy agent_engine    # picks up new GUB_SERVICE_JWT from .env
   ```
   For an in-place update without redeploy, use the Vertex AI console
   under Agent Engine → your agent → Edit env vars.
3. If JWT is stored locally: update `.env`. Don't commit.

**Verification.** Send a test query through Gemini Enterprise that
exercises a GUB-backed tool. Confirm the response is non-error and
includes data the user is authorized to see.

**Cleanup.** GUB-issued JWTs expire on their own — no explicit revocation
step today. **Note: this is a gap.** Until refresh-token revocation lands,
a leaked service JWT remains valid until `exp`. Mitigation: keep TTL
short (default 15 min for access tokens) and rotate immediately on any
suspected exposure.

#### OAuth relay redirect URI whitelist

The OAuth relay (`deployment/oauth-relay/main.py`, Cloud Function)
maintains a whitelist of redirect URIs to defend against open-redirect
attacks during the OAuth flow.

**Preconditions.** None — config-only change.

**Steps.**
1. Edit `deployment/oauth-relay/main.py` (or wherever the whitelist
   lives in the deployed function — verify against the actual deployed
   source if unsure).
2. Redeploy the Cloud Function:
   ```bash
   gcloud functions deploy oauth-relay --source=deployment/oauth-relay --runtime=python311 ...
   ```
   (verify the actual deploy command matches your existing infra).

**Verification.** Initiate a Gemini Enterprise → agent flow that should
succeed (whitelisted redirect) and one that should fail (an arbitrary
redirect URI). The latter must be rejected.

**Cleanup.** Old function revisions are kept by Cloud Functions for
rollback. After a few days of stable behavior, prune old revisions if
quota is a concern.

#### GCP service account / Application Default Credentials

This agent uses ADC, not a static SA key file. Rotation = the standard
GCP SA-key rotation if you've configured one for ADC; locally, run
`gcloud auth application-default login` again to refresh the user
credential. There's no per-service-key rotation procedure here because
the service has none.

### What this service does NOT have

- No JWT signing keys (it consumes JWTs verified by GUB; doesn't sign).
- No DB credentials (it talks to GUB over HTTP; never touches the DB).
- No third-party API keys beyond ADC for GCP-native APIs.

### Cut a user off

Access to this agent is controlled at the **Gemini Enterprise app**
level, not by this code. To revoke a specific user's ability to invoke
the agent:

1. Remove them from the Gemini Enterprise app's user/group binding (in
   the GCP console under Agentspace → your app → Access).
2. If the user also has access to gub-admin, follow the [Cut a user
   off](https://github.com/bpriddy/gcp-universal-backend#cut-a-user-off-revoke-admin-access)
   procedure in gcp-universal-backend.
3. Their existing GUB-issued JWTs remain valid until `exp` (see above
   gap note). For immediate cutoff, escalate to the IT process.

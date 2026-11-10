"""
query_service — the query agent service (bpriddy/ub-query-agent#92).

A small Cloud Run service in front of the Agent Engine. UB's `/ask` calls it,
and nothing else can: it is private, with UB's runtime service account as its
only invoker. The rule it follows: the agent owns user things, UB owns org
things (UB `docs/ASK.md`, D12).

- **Who is asking** comes from the caller's UB token, never from a field
  (D28): the person is the token's subject, the app its per-app audience.
  An actor and a context may be sent; they are recorded, never used for
  permission.
- **Conversations** live in this service's own database on the shared Cloud
  SQL instance (`migrations/`). A conversation is resumed only by the same
  person in the same app, within 15 minutes idle.
- **Per turn** the caller's UB token, and a Google access token minted from
  the person's Gemini Enterprise grant when there is one, go to the engine in
  `temp:` state, which ADK never saves (gcp-universal-backend#162).
- **The stream** back to UB carries only what the `/ask` turn adds
  (`gub_agent/agents/ask.py`): progress lines, then one answer.

Separate from `gub_agent` on purpose: this is deployed to Cloud Run with its
own image (`Dockerfile`), and imports nothing from the agent package, so the
image carries no ADK.
"""

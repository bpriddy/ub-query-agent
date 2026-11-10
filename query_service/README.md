# Query agent service

The private Cloud Run service UB's `/ask` calls (ub-query-agent#92). It owns
the query agent's database and hands each turn to the Agent Engine. Design:
UB `docs/ASK.md`; decisions D12 and D28 in UB `docs/ARCHITECTURE-DECISIONS.md`.

## `POST /ask`

Called only by UB's runtime service account.

| Header | Holds |
|---|---|
| `X-Serverless-Authorization` | UB's Google ID token; Cloud Run checks it (invoker) |
| `Authorization` | `Bearer <the person's UB access token>`; checked here against UB's keys |

```jsonc
{ "question": "Who's staffed on the Nike holiday campaign?",
  "conversation_id": "…",              // optional: resume
  "actor": "ava",                      // optional, recorded only
  "context": { "type": 1, "id": "holocron" } }  // optional, recorded only
```

There is no person field. The person is the token's subject and the app its
per-app audience (D28).

The answer is a Server-Sent Events stream:

```
event: conversation  {"conversation_id": "…", "new": true}
event: progress      {"step": "lookup", "text": "Checking staffing in Runn"}
event: answer        {AskResponse, conversation_id = ours}
```

The answer is always last and always sent; a failed turn is
`kind="error"` with the reason. Errors before the stream are JSON
`{"code", "message"}`:

| Status | Code |
|---|---|
| 400 | `BAD_REQUEST` |
| 401 | `MISSING_TOKEN`, `INVALID_TOKEN` |
| 404 | `CONVERSATION_NOT_FOUND` (not yours, or no such id) |
| 502 | `ENGINE_UNAVAILABLE` |
| 503 | `UB_KEYS_UNAVAILABLE` |

## Conversations

Resumed only by the same person in the same app, within 15 minutes idle.
After that, the turn starts a new conversation and the stream carries the new
id. Actor and context are recorded, never compared.

## Tokens

Per turn, the person's UB token goes to the engine as `temp:gub_jwt`, and a
Google access token minted from their Gemini Enterprise grant (when they have
one) as `temp:workspace_token`. ADK never saves `temp:` state. The only token
stored is the grant's refresh token, encrypted with the service's KMS key.

## Database

`migrations/*.sql`, applied in order at startup. Never edit an applied file;
add a new one.

## Run the tests

```bash
docker run -d --rm --name qa-test-pg -e POSTGRES_PASSWORD=test -e POSTGRES_DB=qa_test \
  -p 127.0.0.1:55432:5432 postgres:15
pip install -e ".[dev,service]"
TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/qa_test pytest tests/service
```

## Deploy (dev)

1. UB terraform `query_agent.tf`: account, database, IAM database user, KMS
   key.
2. Once, as a `cloudsqlsuperuser` member, connected to `ub_query_agent_dev`:

   ```sql
   GRANT ALL ON SCHEMA public TO "sa-ub-query-agent-dev@os-test-491819.iam";
   ```

3. `deployment/deploy-service.sh`
4. UB terraform `query_agent_provisioned = true`: UB's invoker grant.

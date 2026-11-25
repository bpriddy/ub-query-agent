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
event: progress      {"step": "lookup", "text": "Checking staffing in resourcing"}
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

A turn that used the person's Gemini Enterprise connection in a second or
later app carries, once, `notices: [{"code": "WORKSPACE_CONNECTION_REUSED",
"message": "Using your existing Gemini Enterprise connection."}]` (#93).
`WORKSPACE_NOT_CONNECTED` carries the connect page (`<PUBLIC_BASE_URL>/connect`, on UB) as its `link`.

## Conversations

Resumed only by the same person in the same app, within 15 minutes idle.
After that, the turn starts a new conversation and the stream carries the new
id. Actor and context are recorded, never compared.

## Tokens

Per turn, the person's UB token goes to the engine as `temp:gub_jwt`, and a
Google access token minted from their Gemini Enterprise grant (when they have
one) as `temp:workspace_token`. ADK never saves `temp:` state. The only token
stored is the grant's refresh token, encrypted with the service's KMS key.

## Connect page (#93)

Where a person connects Gemini Enterprise once, for every app. Code:
`connect.py`.

| Route | What it does |
|---|---|
| `GET /connect` | The page: connected or not, Connect / Disconnect |
| `GET /connect/start` | To Google consent (`?intent=manage`: sign in only) |
| `GET /connect/callback` | Google's redirect URI |
| `POST /connect/disconnect` | Revoke at Google, delete the grant |
| `GET /connection`, `DELETE /connection` | The same for an app through UB, with the person's UB token |

- **Served through UB.** This service is IAM-only (UB is its only invoker),
  so browsers open the page on UB: UB forwards exactly `GET /connect`,
  `GET /connect/start`, `GET /connect/callback` and `POST /connect/disconnect`
  here with its ID token (UB `src/modules/connect/connect.proxy.ts`, docs/ASK.md
  "Connect page"). It forwards the query string, the two page cookies, Origin
  on the POST, and `X-UB-Connect-Origin` (the origin it served). Never
  Authorization, other cookies or a body. The service stays private.
- **URLs come from `PUBLIC_BASE_URL` only**, never Host or X-Forwarded-*. It
  is UB's public origin (no path), the same value as UB's
  `CONNECT_PUBLIC_URL`. The redirect URI is `<PUBLIC_BASE_URL>/connect/callback`.
  When `X-UB-Connect-Origin` differs from it, the page answers
  `503 CONNECT_ORIGIN_MISMATCH`: the state cookie and Google's redirect would
  land on different hosts. That header is trusted only because UB alone can
  invoke this service.
- **Sign in with UB.** The callback gives the Google access token to UB's
  `/auth/google/access-token-exchange`; UB says who the person is, and the
  grant is stored under that UB user id. The UB session the exchange opens is
  closed at once. So the client must be one UB admits: its id on an active
  `trusted_apps` row in that environment.
- **Consent:** `openid email discoveryengine.assist.readwrite`, offline,
  `prompt=consent`. Only the assist scope is recorded on the grant.
- **Cookies:** the OAuth state (10 min) and the manage sign-in (15 min), each
  a KMS ciphertext with its own `aad`, `Path=/connect`, host-only on UB's
  host. No extra secret.
- **Config:** `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (Secret Manager,
  mounted by name), `PUBLIC_BASE_URL`, optional `GOOGLE_HOSTED_DOMAIN`
  (account chooser hint). Any of the first three unset: the page answers
  `503 CONNECT_NOT_CONFIGURED`, and no Google token is minted.

### The OAuth client (one per environment, made by a human)

The connect page uses **UB's own Google OAuth web client**, one per
environment, not the sunsetting chat bot's. Google has no API for creating
web clients, so a human makes each one in the console.

**Which project.** The client's project is the quota project of every
`streamAssist` call: the call sends no `x-goog-user-project`, and with a
user token Google bills and checks quota against the project that owns the
OAuth client. So the project must have the Discovery Engine API enabled and
be under the anomaly.com organization (an Internal consent screen requires
it). Google's docs do not require it to be the Gemini Enterprise app's own
project. The simplest choice is: the environment's own project, which also
holds (or will hold) its Gemini Enterprise app.

| Env | Project | Redirect URIs | Ready |
|---|---|---|---|
| prod | `os-test-491819` | `https://gcp-universal-backend-dev-843516467880.us-central1.run.app/connect/callback` and `https://ub.anomaly.com/connect/callback` | now (Discovery Engine on) |
| stage | `ub-stage-anomaly` | `https://ub-stage.anomaly.com/connect/callback` (+ its run.app URL once UB is deployed there) | after billing is linked and Discovery Engine is enabled |
| dev | `ub-dev-anomaly` | `https://ub-dev.anomaly.com/connect/callback` (+ its run.app URL once UB is deployed there) | after billing is linked and Discovery Engine is enabled |

Steps, per environment, in order:

1. **UB terraform** (`connect_client_secrets = true`, already set in each
   environment's `ub.tf`): `terraform apply` creates two empty secrets,
   `ub-connect-google-client-id` and `ub-connect-google-client-secret`, and
   this service's runtime account's `secretAccessor` on each.
2. **Console → the project above → Google Auth Platform** (formerly "OAuth
   consent screen"):
   - Branding: app name `UB connect (<env>)`, support email, authorized domain
     `anomaly.com`, developer contact email.
   - Audience: **Internal**.
   - Data Access → Add or remove scopes: `openid`, `.../auth/userinfo.email`
     (`email`), and, under "Manually add scopes",
     `https://www.googleapis.com/auth/discoveryengine.assist.readwrite`.
3. **Clients → Create client → Web application**, name `ub-connect-<env>`.
   No JavaScript origins. Authorized redirect URIs: the row above. Create,
   then copy the client id and secret (download the JSON: the secret is shown
   once).
4. **Secret values**, by the person holding them, never in a file, a ticket
   or a shell history:
   `gcloud secrets versions add ub-connect-google-client-id --project <project> --data-file=-`
   then the same for `ub-connect-google-client-secret`; paste, then Ctrl-D.
5. **Admit the client in UB**: gub-admin → Trusted apps, an active entry for
   the connect page with this client id (UB's access-token exchange admits
   only registered clients; `GOOGLE_ACCESS_TOKEN_ENFORCEMENT=strict`
   rejects any other).
6. **This repo's GitHub Environment `<env>`**: `QUERY_SERVICE_PUBLIC_URL` =
   UB's public origin (prod today: the run.app URL above; `https://ub.anomaly.com`
   once that hostname serves UB), `QUERY_SERVICE_CONNECT_CLIENT=ub`, and
   optionally `QUERY_SERVICE_GOOGLE_HOSTED_DOMAIN=anomaly.com`.
7. **Deploy this service** (Actions → Deploy query service, or
   `deployment/deploy-service.sh <env>`), then **UB** with
   `CONNECT_PUBLIC_URL` set to the same origin.
8. Open `<UB origin>/connect`, connect, and check a `/ask` turn reports
   Workspace.

Moving prod to `ub.anomaly.com` later: change `QUERY_SERVICE_PUBLIC_URL` and
UB's `CONNECT_PUBLIC_URL` together, then deploy both. Both redirect URIs are
already on the client. UB redirects a GET on its other address to
`CONNECT_PUBLIC_URL`, so old links keep working.

**Fallback, until UB's client exists.** `QUERY_SERVICE_CONNECT_CLIENT=bot`
mounts the chat bot's client (`gub-gchat-bot-oauth-client-id-dev`,
`gub-gchat-bot-oauth-client-secret-dev`, prod only). It also needs this
service's runtime account granted `secretAccessor` on both and the UB
callback added to the bot's client, and it ends when the bot is retired.
Unset (today) mounts no client: nothing changes. The explicit
`QUERY_SERVICE_GOOGLE_CLIENT_{ID,SECRET}_SECRET` names still win when set
(`deployment/connect-client.sh`).

### Chat bot grants

The sunsetting chat bot holds its users' grants (`chat_user_sessions`,
encrypted with the bot's KMS key, issued to the bot's client). **Bot users
reconnect here.** A refresh token works only with the client it was issued
to, so with UB's own client the bot's grants cannot be copied. (Under the
`bot` fallback a one-time copy would be possible; it is not built and must
not be run without the owner's decision.)

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

3. Actions → **Deploy query service** with `environment: prod` (not yet run;
   see the workflow header for the deployer IAM it needs). Until then,
   `deployment/deploy-service.sh prod`, which reads the same GitHub
   Environment variables.
4. UB terraform `query_agent_provisioned = true`: UB's invoker grant.
5. Connect page (optional, #93): "The OAuth client" above.

-- The query agent's own tables (ub-query-agent#92). UB never reads them (D12).
--
-- No token is stored in clear anywhere here: the caller's UB token and any
-- Google access token live only for the turn, in the engine's temp: state.

-- Every /ask caller's conversations, not only Ava's. The person and app come
-- from the caller's UB token; actor and context are recorded as the caller
-- states them and never used for permission (D28).
CREATE TABLE conversations (
  id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  person_id         text        NOT NULL,
  app_id            text,
  actor             text,
  context_type      smallint    CHECK (context_type IN (1, 2)),
  context_id        text,
  engine_session_id text        NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now(),
  last_active_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX conversations_person_idx ON conversations (person_id, last_active_at DESC);

-- One Gemini Enterprise grant per person, reused by every app in the
-- collection (#93). The refresh token is KMS-encrypted.
CREATE TABLE workspace_grants (
  person_id                text        PRIMARY KEY,
  refresh_token_ciphertext bytea       NOT NULL,
  scopes                   text[]      NOT NULL,
  client_id                text        NOT NULL,
  connected_at             timestamptz NOT NULL DEFAULT now()
);

-- The apps that have told the person their grant is reused (#93), once each.
CREATE TABLE workspace_grant_notices (
  person_id text        NOT NULL REFERENCES workspace_grants (person_id) ON DELETE CASCADE,
  app_id    text        NOT NULL,
  shown_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (person_id, app_id)
);

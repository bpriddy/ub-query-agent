"""
config.py — the service's settings, from the environment.

Read once at startup (`Settings.from_env`), so a missing setting fails the
revision's start rather than its first request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set")
    return value


def _optional(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


@dataclass(frozen=True)
class Settings:
    # The engine: projects/<p>/locations/<r>/reasoningEngines/<id>.
    engine: str
    # UB's access tokens: RS256, keys at <UB_BASE_URL>/auth/jwks.
    ub_base_url: str
    ub_issuer: str
    ub_audience: str
    # Postgres. DATABASE_URL locally and in tests; on Cloud Run the Cloud SQL
    # connector with IAM database auth (no password anywhere).
    database_url: str | None
    cloud_sql_instance: str | None
    db_name: str | None
    db_iam_user: str | None
    # KMS key that encrypts Gemini Enterprise refresh tokens.
    grant_kms_key: str | None
    # The Google client the grants were issued to. Without both, no Google
    # token is minted and Workspace is reported not connected (#90).
    google_client_id: str | None
    google_client_secret: str | None
    # Where a person's browser reaches the connect page (#93), e.g.
    # https://<host> (+ a path prefix when a proxy serves it under one). The
    # Google redirect URI is <this>/connect/callback and must be registered on
    # the client. Unset: the connect routes answer CONNECT_NOT_CONFIGURED and
    # WORKSPACE_NOT_CONNECTED carries no link.
    public_base_url: str | None = None
    # The Workspace domain, as a hint to Google's account chooser only.
    google_hosted_domain: str | None = None
    conversation_idle_seconds: int = 900

    @property
    def connect_url(self) -> str | None:
        return f"{self.public_base_url}/connect" if self.public_base_url else None

    @property
    def engine_region(self) -> str:
        parts = self.engine.split("/")
        if len(parts) != 6 or parts[0] != "projects" or parts[4] != "reasoningEngines":
            raise ConfigError("ENGINE must be projects/<p>/locations/<r>/reasoningEngines/<id>")
        return parts[3]

    @classmethod
    def from_env(cls) -> Settings:
        ub_base_url = _required("UB_BASE_URL").rstrip("/")
        settings = cls(
            engine=_required("ENGINE"),
            ub_base_url=ub_base_url,
            ub_issuer=_optional("UB_ISSUER") or ub_base_url,
            ub_audience=_optional("UB_AUDIENCE") or ub_base_url,
            database_url=_optional("DATABASE_URL"),
            cloud_sql_instance=_optional("CLOUD_SQL_INSTANCE"),
            db_name=_optional("DB_NAME"),
            db_iam_user=_optional("DB_IAM_USER"),
            grant_kms_key=_optional("GRANT_KMS_KEY"),
            google_client_id=_optional("GOOGLE_CLIENT_ID"),
            google_client_secret=_optional("GOOGLE_CLIENT_SECRET"),
            public_base_url=(_optional("PUBLIC_BASE_URL") or "").rstrip("/") or None,
            google_hosted_domain=_optional("GOOGLE_HOSTED_DOMAIN"),
            conversation_idle_seconds=int(os.environ.get("CONVERSATION_IDLE_SECONDS", "900")),
        )
        settings.engine_region  # noqa: B018 — validates the engine name
        if not settings.database_url and not (
            settings.cloud_sql_instance and settings.db_name and settings.db_iam_user
        ):
            raise ConfigError("set DATABASE_URL, or CLOUD_SQL_INSTANCE + DB_NAME + DB_IAM_USER")
        if settings.public_base_url and not settings.public_base_url.startswith("https://"):
            raise ConfigError("PUBLIC_BASE_URL must be https://")
        return settings

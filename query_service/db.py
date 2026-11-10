"""
db.py — the connection pool and the migrations.

On Cloud Run the pool connects through the Cloud SQL connector with IAM
database auth as the service's own account, so there is no database password
anywhere. Locally and in tests, DATABASE_URL.

Migrations are the numbered `.sql` files in `migrations/`, applied in order at
startup, each in its own transaction, under an advisory lock so two starting
instances do not race. A file is never edited once applied: a change is a new
file.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg

from .config import Settings

MIGRATIONS = Path(__file__).parent / "migrations"
# Any fixed number; only this service takes it.
_MIGRATION_LOCK = 920_092


async def create_pool(settings: Settings) -> asyncpg.Pool:
    if settings.database_url:
        return await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)

    from google.cloud.sql.connector import Connector, create_async_connector

    connector: Connector = await create_async_connector()

    async def connect(*_args, **_kwargs) -> asyncpg.Connection:
        return await connector.connect_async(
            settings.cloud_sql_instance,
            "asyncpg",
            user=settings.db_iam_user,
            db=settings.db_name,
            enable_iam_auth=True,
        )

    return await asyncpg.create_pool(connect=connect, min_size=1, max_size=5)


async def migrate(pool: asyncpg.Pool) -> list[str]:
    """Apply the migrations not yet applied; returns their names."""
    applied: list[str] = []
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK)
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name text PRIMARY KEY,"
                " applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {r["name"] for r in await conn.fetch("SELECT name FROM schema_migrations")}
            for path in sorted(MIGRATIONS.glob("*.sql")):
                if path.name in done:
                    continue
                async with conn.transaction():
                    await conn.execute(path.read_text())
                    await conn.execute(
                        "INSERT INTO schema_migrations (name) VALUES ($1)", path.name
                    )
                applied.append(path.name)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK)
    return applied

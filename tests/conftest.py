"""
Suite-wide fixtures: every test runs with a fresh shared client.

The MockGub server fixture lives in tests/integration/conftest.py — unit
tests never open a socket to a real endpoint.
"""

from __future__ import annotations

import os

# gub_agent.config requires GUB_BASE_URL (no default, D37). Tests never reach a
# real backend: they monkeypatch the URL or use MockGub. Set before any import.
os.environ.setdefault("GUB_BASE_URL", "http://gub.test")

import pytest  # noqa: E402

from gub_agent import models  # noqa: E402
from gub_agent.tools._client import aclose_client  # noqa: E402


@pytest.fixture(autouse=True)
async def _close_shared_client():
    yield
    await aclose_client()


@pytest.fixture(autouse=True)
def _unspent_hedge_budgets():
    """Every test starts with every invocation's hedge budget unspent. The
    budget is module state keyed on the invocation id (models._HEDGES), and the
    tests reuse ids such as "inv-1", so one test's fired hedges would otherwise
    cap the next one's."""
    budgets = getattr(models, "_HEDGES", None)
    if budgets is not None:
        budgets.clear()
    yield
    if budgets is not None:
        budgets.clear()

"""
PUBLIC_BASE_URL (ub-query-agent#93): UB's public origin, because UB forwards
/connect* at its root to this private service. An https origin, no path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from query_service.config import ConfigError, Settings  # noqa: E402

BASE = {
    "ENGINE": "projects/1/locations/us-central1/reasoningEngines/2",
    "UB_BASE_URL": "https://ub.test",
    "DATABASE_URL": "postgresql://x",
}


@pytest.mark.parametrize(
    "value",
    [
        "https://ub.anomaly.com",
        "https://ub.anomaly.com/",
        "https://gcp-universal-backend-dev-843516467880.us-central1.run.app",
    ],
)
def test_an_https_origin_is_accepted(monkeypatch, value):
    for k, v in {**BASE, "PUBLIC_BASE_URL": value}.items():
        monkeypatch.setenv(k, v)
    s = Settings.from_env()
    assert s.public_base_url == value.rstrip("/")
    assert s.connect_url == f"{value.rstrip('/')}/connect"
    assert s.callback_url == f"{value.rstrip('/')}/connect/callback"


@pytest.mark.parametrize(
    "value", ["http://ub.anomaly.com", "https://ub.anomaly.com/qa", "https://ub.anomaly.com?x=1"]
)
def test_anything_else_is_refused(monkeypatch, value):
    for k, v in {**BASE, "PUBLIC_BASE_URL": value}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_unset_means_no_connect_page(monkeypatch):
    for k, v in BASE.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    s = Settings.from_env()
    assert s.connect_url is None and s.callback_url is None

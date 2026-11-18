"""Never let tests write to a developer's configured MongoDB by accident."""

import pytest


@pytest.fixture(autouse=True)
def isolate_mongodb_configuration(monkeypatch):
    # Integration tests explicitly opt into a dedicated local test server.
    monkeypatch.delenv("MONGODB_URI", raising=False)

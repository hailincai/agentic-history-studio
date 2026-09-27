import socket

import pytest


@pytest.fixture(autouse=True)
def offline_network_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every default test fails closed on real network access, even with API credentials set."""
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Real network access is forbidden in the offline test suite")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

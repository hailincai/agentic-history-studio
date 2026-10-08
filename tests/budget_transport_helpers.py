"""Legacy transport-only SDK tests explicitly mock the budget boundary.

Real project accounting and SDK coverage are tested in test_project_budget.py.
This fixture is imported only by the three transport unit-test modules.
"""
import pytest


class TransportOnlyBudget:
    def responses(self, request, call, **pricing):
        return call()

    def tts(self, *, call, **request):
        return call()

    def disabled(self, operation):
        pass


@pytest.fixture(autouse=True)
def transport_only_budget(monkeypatch):
    monkeypatch.setattr("history_studio.budget.require_budget", lambda client: TransportOnlyBudget())

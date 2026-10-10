"""Explicit custom provider configuration is durable recovery authority."""
import pytest
import json

from history_studio.media.recovery import provider_identity
from history_studio.media.recovery import AssetRecovery, configuration_digest
from history_studio.workflow import ProjectState as S
from test_media_workflow import prepared, workflow, read
from test_media_recovery import journal, counts


class ConfiguredProvider:
    def __init__(self, seed=1, model="fake", voice="mandarin", style="plain"):
        self.seed, self.model, self.voice, self.style = seed, model, voice, style

    def recovery_identity(self):
        return {"version": 1, "complete": True, "implementation": "fake-v1",
                "settings": dict(seed=self.seed, model=self.model, voice=self.voice, style=self.style)}


def test_seed_collision_is_rejected():
    assert provider_identity(ConfiguredProvider(1)) != provider_identity(ConfiguredProvider(2))


@pytest.mark.parametrize("field,value", [("model", "other"), ("voice", "other"), ("style", "other")])
def test_generation_settings_change_identity(field, value):
    original = ConfiguredProvider()
    changed = ConfiguredProvider(**{field: value})
    assert provider_identity(original) != provider_identity(changed)
    assert provider_identity(original) == provider_identity(ConfiguredProvider())


@pytest.mark.parametrize("bad", [None, {}, {"version": 1, "complete": False, "implementation": "v1", "settings": {}},
    {"version": 1, "complete": True, "implementation": "v1", "settings": {"seed": object()}},
    {"version": 1, "complete": True, "implementation": "v1", "settings": {"seed": float("nan")}},
    {"version": 1, "complete": True, "implementation": "v1", "settings": {"api_key": "private"}},
    {"version": 1, "complete": True, "settings": {}}])
def test_invalid_identity_fails_before_any_dispatch(prepared, monkeypatch, bad):
    project, store, providers = prepared
    authority = read(store).artifacts
    monkeypatch.setattr(providers.tts, "recovery_identity", None if bad is None else lambda: bad)
    outcome = workflow(providers).run(project, store)
    assert outcome.state.current_state == S.FAILED
    assert counts(providers) == (0, 0, 0)
    assert outcome.state.artifacts == authority
    assert not (store.project_dir / ".runtime/media_recovery.json").exists()


def test_visible_setting_missing_or_mismatched():
    provider = ConfiguredProvider()
    declaration = provider.recovery_identity()
    declaration["settings"]["seed"] = 2
    provider.recovery_identity = lambda: declaration
    with pytest.raises(ValueError, match="mismatches"):
        provider_identity(provider)


def test_ambiguous_identity_is_rejected():
    provider = ConfiguredProvider()
    original = provider.recovery_identity
    def unstable():
        provider.seed += 1
        return original()
    provider.recovery_identity = unstable
    with pytest.raises(ValueError, match="Invalid complete"):
        provider_identity(provider)


def test_identity_never_persists_configuration_values_or_credentials(prepared, monkeypatch):
    project, store, providers = prepared
    providers.tts.api_key = "secret-client-key"
    declaration = providers.tts.recovery_identity()
    declaration["settings"]["opaque_setting"] = "private-value"
    monkeypatch.setattr(providers.tts, "recovery_identity", lambda: declaration)
    assert workflow(providers).run(project, store).state.current_state == S.ASSEMBLING
    raw = (store.project_dir / ".runtime/media_recovery.json").read_text(encoding="utf-8")
    assert "secret-client-key" not in raw and "private-value" not in raw
    assert journal(store).scope.providers["tts"]["custom_identity_version"] == 1


@pytest.mark.parametrize("change", ["same", "seed", "legacy"])
def test_completed_recovery_scope_reuse_or_safe_rejection(prepared, monkeypatch, change):
    project, store, providers = prepared
    perform = AssetRecovery.perform
    def interrupt(self, **kwargs):
        if kwargs["key"] == "narration:SEG-A":
            raise OSError("local interruption before dispatch")
        return perform(self, **kwargs)
    monkeypatch.setattr(AssetRecovery, "perform", interrupt)
    assert workflow(providers).run(project, store).state.current_state == S.FAILED
    before = journal(store).entries["narration:SEG-Z"].model_dump()
    if change == "seed":
        declaration = providers.tts.recovery_identity()
        declaration["settings"]["seed"] = 2
        monkeypatch.setattr(providers.tts, "recovery_identity", lambda: declaration)
    elif change == "legacy":
        path = store.project_dir / ".runtime/media_recovery.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["scope"]["providers"]["tts"] = {"class": "legacy.Custom"}
        record["scope_sha256"] = configuration_digest(record["scope"])
        path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(AssetRecovery, "perform", perform)
    outcome = workflow(providers).run(project, store)
    assert journal(store).entries["narration:SEG-Z"].model_dump() == before
    if change == "same":
        assert outcome.state.current_state == S.ASSEMBLING
        assert counts(providers) == (2, 2, 2)
    else:
        assert outcome.state.current_state == S.FAILED and outcome.state.artifacts.media is None
        assert counts(providers) == (1, 0, 0)

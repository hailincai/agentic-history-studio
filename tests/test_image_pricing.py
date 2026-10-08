"""Synthetic prices and fake SDK only; estimates never grant paid authority."""
import base64
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from history_studio.budget import UnsupportedPrice, attach_budget
from history_studio.media.openai_provider import ImagePricing, MediaConfiguration, OpenAIImageProvider, image_request
from history_studio.models import StoryboardPackage
from test_narration_media import storyboard_package
from test_project_budget import setup, reserve
from test_visual_media import png_bytes


def pricing(**changes):
    # Arbitrary synthetic test prices; no statement about current provider rates.
    return ImagePricing(**(dict(provenance="synthetic test fixture, not official pricing", as_of="2026-10-08",
        input_usd_per_million_tokens="2", output_usd_per_million_tokens="3",
        configured_output_token_allowance=1000) | changes))


def request(**changes):
    return image_request(model="gpt-image-1", quality="low", size="1536x1024", prompt="李白") | changes


def config(**changes):
    return MediaConfiguration(**(dict(tts_model="tts-1", tts_voice="alloy", tts_usd_per_million_characters=1,
        image_model="gpt-image-1", image_size="1536x1024", image_pricing=pricing()) | changes))


def test_versioned_estimate_identity_and_input_output_accounting():
    result = pricing().estimate(request())
    assert result["input_token_allowance"] == 4102
    assert Decimal(result["configured_reservation_usd"]) == Decimal("0.011204")
    assert result["provider_guaranteed_maximum"] is False
    assert result["pricing"]["version"] == 1
    assert result["pricing"]["quality"] == "low"
    assert result["pricing"]["n"] == 1
    assert result["pricing"]["provenance"].startswith("synthetic")
    assert pricing().estimate(request(prompt="a"))["input_token_allowance"] == 4097


@pytest.mark.parametrize("field,value", [("model", "other"), ("size", "1024x1024"), ("quality", "auto"),
    ("n", 2), ("n", True), ("output_format", "jpeg"), ("background", "transparent"),
    ("moderation", "low"), ("stream", True)])
def test_mismatched_request_identity_rejected(field, value):
    with pytest.raises(UnsupportedPrice, match="identity mismatch"):
        pricing().estimate(request(**{field: value}))


def test_extra_request_parameters_cannot_evade_pricing_identity():
    with pytest.raises(UnsupportedPrice, match="unsupported pricing dimensions"):
        pricing().estimate(request(output_compression=20))


@pytest.mark.parametrize("changes", [dict(version=2), dict(quality="auto"), dict(size="1024x1024"),
    dict(n=2), dict(provenance=""), dict(input_usd_per_million_tokens=0),
    dict(output_usd_per_million_tokens="NaN"), dict(configured_output_token_allowance=0),
    dict(provider_guaranteed_maximum=True)])
def test_invalid_price_contract_cannot_claim_a_guarantee(changes):
    with pytest.raises(ValidationError):
        pricing(**changes)


def test_prompt_limit_counts_utf8_without_rewriting():
    assert image_request(model="gpt-image-1", size="1536x1024", quality="low", prompt="a" * 32000)["prompt"] == "a" * 32000
    for prompt in ("a" * 32001, "白" * 10667, " "):
        with pytest.raises(ValueError, match="32000 UTF-8 bytes"):
            image_request(model="gpt-image-1", size="1536x1024", quality="low", prompt=prompt)
    with pytest.raises(ValueError, match="explicitly low"):
        image_request(model="gpt-image-1", size="1536x1024", quality="auto", prompt="scene")


def test_old_config_remains_readable_and_new_config_roundtrips():
    old = MediaConfiguration(tts_model="tts-1", tts_voice="alloy", image_model="gpt-image-1")
    assert old.image_quality == "low" and old.image_pricing is None
    assert MediaConfiguration.model_validate_json(config().model_dump_json()) == config()
    with pytest.raises(UnsupportedPrice, match="pricing is missing"):
        old.validate_guarded_prices(storyboard_package())


def test_fake_sdk_sends_explicit_low_request_and_preserves_png(monkeypatch):
    from budget_transport_helpers import TransportOnlyBudget
    monkeypatch.setattr("history_studio.budget.require_budget", lambda client: TransportOnlyBudget())
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(png_bytes()).decode())], output_format="png")
    fake = SimpleNamespace(images=SimpleNamespace(generate=generate))
    result = OpenAIImageProvider(fake, model="gpt-image-1", size="1536x1024", quality="low", pricing=pricing()).generate(prompt="李白")
    assert calls == [request()]
    assert result.image_bytes == png_bytes()


@pytest.mark.parametrize("exhausted", [False, True])
def test_strict_guard_never_calls_sdk_or_writes_estimated_charge(tmp_path, exhausted):
    root, guard = setup(tmp_path)
    if exhausted:
        reserve(guard, "existing", "1")
    before = guard.path.read_bytes(), (root / ".runtime/state.json").read_bytes()
    fake = SimpleNamespace(max_retries=0, images=SimpleNamespace(generate=lambda **kwargs: pytest.fail("No paid image call")))
    attach_budget(fake, root, "media")
    with pytest.raises(UnsupportedPrice):
        OpenAIImageProvider(fake, model="gpt-image-1", size="1536x1024", pricing=pricing()).generate(prompt="李白")
    assert before == (guard.path.read_bytes(), (root / ".runtime/state.json").read_bytes())
    # Restart neither adopts the estimate nor releases earlier reservations.
    from history_studio.budget import ProjectBudget
    assert ProjectBudget(root, stage="restart").snapshot()["outstanding_usd"] == (1 if exhausted else 0)


def test_batch_preflight_checks_all_prompts_and_preserves_board():
    board = storyboard_package()
    before = board.model_dump_json()
    with pytest.raises(UnsupportedPrice, match="not a proven total"):
        config().validate_guarded_prices(board)
    assert board.model_dump_json() == before
    data = board.model_dump(mode="json")
    data["sections"][0]["shots"][-1]["generation_prompt"] = "白" * 10667
    with pytest.raises(ValueError, match="32000 UTF-8 bytes"):
        config().validate_guarded_prices(StoryboardPackage.model_validate(data))
    with pytest.raises(UnsupportedPrice, match="identity mismatch: size"):
        config(image_size="1024x1024").validate_guarded_prices(board)


@pytest.mark.parametrize("method", ["TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"])
def test_mixed_video_batch_stays_blocked(method):
    data = storyboard_package().model_dump(mode="json")
    data["sections"][0]["shots"][-1]["generation_method"] = method
    with pytest.raises(UnsupportedPrice, match="Guarded video is disabled"):
        config(video_model="sora-2").validate_guarded_prices(StoryboardPackage.model_validate(data))

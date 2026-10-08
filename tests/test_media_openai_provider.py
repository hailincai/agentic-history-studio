"""Actual installed SDK calls over MockTransport; offline guard remains enabled."""
import base64
import json
from contextlib import contextmanager

import httpx
import pytest
from budget_transport_helpers import transport_only_budget
from openai import OpenAI, InternalServerError
from pydantic import ValidationError

from history_studio.media import measure_wav_duration
from history_studio.media.openai_provider import MediaConfiguration, OpenAITTSProvider, OpenAIImageProvider, OpenAIVideoProvider
from test_narration_media import wav_bytes, storyboard_package
from test_visual_media import png_bytes, mp4_bytes


@contextmanager
def sdk(handler):
    requests = []
    def capture(request):
        request.read()
        requests.append(request)
        return handler(request)
    with OpenAI(api_key="test-key", base_url="https://api.openai.com/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(capture))) as client:
        yield client, requests


def job(status="completed", job_id="video_123"):
    return dict(id=job_id, model="sora-2", object="video", created_at=1,
                seconds="4", size="1280x720", progress=100, status=status)


def test_tts_exact_text_wav_request_actual_audio_and_metadata():
    text = "Exact narration — 李白\nwith  spaces."
    with sdk(lambda request: httpx.Response(200, content=wav_bytes(), headers={"Content-Type": "audio/wav"})) as (client, requests):
        result = OpenAITTSProvider(client, model="tts-1", voice="alloy").synthesize(text=text)
        assert json.loads(requests[0].content) == dict(input=text, model="tts-1", voice="alloy", response_format="wav")
        assert requests[0].url.path == "/v1/audio/speech" and len(requests) == 1
        assert result.audio_bytes == wav_bytes() and measure_wav_duration(result.audio_bytes) == 0.25
        assert result.provider == "openai" and result.model == "tts-1" and result.audio_format == "wav"


@pytest.mark.parametrize("text", ["", " ", "a" * 4097])
def test_invalid_tts_input_not_rewritten_or_split(text):
    with sdk(lambda request: pytest.fail("No request for invalid input")) as (client, requests):
        with pytest.raises(ValueError):
            OpenAITTSProvider(client, model="tts-1", voice="alloy").synthesize(text=text)
        assert requests == []


def test_empty_tts_response_rejected():
    with sdk(lambda request: httpx.Response(200, content=b"")) as (client, _):
        with pytest.raises(ValueError, match="non-empty"):
            OpenAITTSProvider(client, model="tts-1", voice="alloy").synthesize(text="Narration")


def test_image_exact_prompt_png_base64_transport_and_metadata():
    prompt = "Approved prompt — 李白\nwith  spaces."
    response = dict(created=1, output_format="png", data=[dict(b64_json=base64.b64encode(png_bytes()).decode("ascii"))])
    with sdk(lambda request: httpx.Response(200, json=response)) as (client, requests):
        result = OpenAIImageProvider(client, model="gpt-image-1").generate(prompt=prompt)
        assert json.loads(requests[0].content) == dict(prompt=prompt, model="gpt-image-1", n=1, output_format="png", size="1024x1024")
        assert result.image_bytes == png_bytes() and result.image_format == "png"
        assert result.provider == "openai" and result.model == "gpt-image-1"


@pytest.mark.parametrize("response", [
    dict(created=1, data=[]),
    dict(created=1, data=[dict(url="https://example.invalid/image.png")]),
    dict(created=1, data=[dict(b64_json="bad%%%base64")]),
    dict(created=1, data=[dict(b64_json=base64.b64encode(b"not PNG").decode())]),
    dict(created=1, output_format="jpeg", data=[dict(b64_json=base64.b64encode(png_bytes()).decode())]),
])
def test_invalid_image_transport_no_url_fallback(response):
    with sdk(lambda request: httpx.Response(200, json=response)) as (client, requests):
        with pytest.raises(ValueError):
            OpenAIImageProvider(client, model="gpt-image-1").generate(prompt="Approved prompt")
        assert len(requests) == 1


@pytest.mark.parametrize("use_image", [False, True])
def test_video_real_sdk_text_image_request_poll_download_identity_and_metadata(monkeypatch, use_image):
    monkeypatch.setattr("history_studio.media.openai_provider.time.sleep", lambda seconds: None)
    prompt = "Exact approved video prompt — 李白"
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json=job("queued"))
        if request.url.path.endswith("/content"):
            assert request.url.params["variant"] == "video"
            return httpx.Response(200, content=mp4_bytes(), headers={"Content-Type": "video/mp4"})
        return httpx.Response(200, json=job())
    with sdk(respond) as (client, requests):
        provider = OpenAIVideoProvider(client, model="sora-2", poll_attempts=2)
        result = (provider.generate_from_image(image=png_bytes(), prompt=prompt) if use_image else provider.generate_from_text(prompt=prompt))
        assert [request.url.path for request in requests] == ["/v1/videos", "/v1/videos/video_123", "/v1/videos/video_123/content"]
        body = requests[0].content
        if use_image:
            assert prompt.encode("utf-8") in body
            assert png_bytes() in body and b'filename="reference.png"' in body and b"image/png" in body
        else:
            # Installed SDK may serialize file-free multipart requests as JSON.
            if requests[0].headers["content-type"].startswith("application/json"):
                assert json.loads(body) == dict(prompt=prompt, model="sora-2", size="1280x720", seconds="4")
            else:
                assert prompt.encode("utf-8") in body
            assert b"input_reference" not in body
        assert result.video_bytes == mp4_bytes() and result.video_format == "mp4"
        assert result.provider == "openai" and result.model == "sora-2"


@pytest.mark.parametrize("failure", ["failed", "unknown", "timeout", "wrong_id"])
def test_video_failure_timeout_or_identity_mismatch_never_downloads(monkeypatch, failure):
    monkeypatch.setattr("history_studio.media.openai_provider.time.sleep", lambda seconds: None)
    def respond(request):
        assert not request.url.path.endswith("/content")
        if request.method == "POST":
            return httpx.Response(200, json=job("queued"))
        status = {"failed": "failed", "unknown": "unexpected", "timeout": "queued", "wrong_id": "completed"}[failure]
        return httpx.Response(200, json=job(status, "foreign_job" if failure == "wrong_id" else "video_123"))
    with sdk(respond) as (client, requests):
        with pytest.raises(TimeoutError if failure == "timeout" else ValueError):
            OpenAIVideoProvider(client, model="sora-2", poll_attempts=2).generate_from_text(prompt="Approved prompt")
        assert len(requests) == (3 if failure == "timeout" else 2)


@pytest.mark.parametrize("image", [None, b"", b"not PNG"])
def test_image_video_bad_reference_cannot_fall_back_to_text(image):
    with sdk(lambda request: pytest.fail("No creation for bad image")) as (client, requests):
        with pytest.raises(ValueError):
            OpenAIVideoProvider(client, model="sora-2").generate_from_image(image=image, prompt="Approved prompt")
        assert requests == []


def test_sdk_errors_propagate_without_automatic_paid_retries():
    with sdk(lambda request: httpx.Response(500, json=dict(error=dict(message="failed", type="server_error")))) as (client, requests):
        with pytest.raises(InternalServerError):
            OpenAITTSProvider(client, model="tts-1", voice="alloy").synthesize(text="Narration")
        assert len(requests) == 1


def test_config_requires_explicit_models_no_secret_fields_and_method_requirements():
    with pytest.raises(ValidationError):
        MediaConfiguration.model_validate({})
    with pytest.raises(ValidationError):
        MediaConfiguration(tts_model="tts-1", tts_voice="alloy", api_key="not-allowed")
    settings = MediaConfiguration(tts_model="tts-1", tts_voice="alloy")
    with pytest.raises(ValueError, match="image_model"):
        settings.validate_for(storyboard_package())
    assert not {"api_key", "cost", "billed_cost"} & set(MediaConfiguration.model_fields)
    with pytest.raises(ValidationError):
        OpenAIVideoProvider(None, model="sora-2", poll_attempts=0)

"""Adapters audited against installed openai 2.54.0; no client creation or secrets here."""
import base64
import binascii
import time
from datetime import date
from decimal import Decimal
from typing import Literal

from openai import OpenAI
from pydantic import Field

from history_studio.models import GenerationMethod, StoryboardPackage
from history_studio.models.base import Contract, Text

from .image import ImageGenerationResult
from .tts import TTSResult
from .video import VideoGenerationResult
from history_studio import budget


IMAGE_PROMPT_MAX_UTF8_BYTES = 32000


class ImagePricing(Contract):
    """Operator-supplied estimate, not a provider-guaranteed charge ceiling."""

    version: Literal[1] = 1
    provenance: Text
    as_of: date
    model: Literal["gpt-image-1"] = "gpt-image-1"
    quality: Literal["low"] = "low"
    size: Literal["1536x1024"] = "1536x1024"
    n: Literal[1] = 1
    output_format: Literal["png"] = "png"
    background: Literal["opaque"] = "opaque"
    moderation: Literal["auto"] = "auto"
    stream: Literal[False] = False
    input_usd_per_million_tokens: Decimal = Field(gt=0, allow_inf_nan=False)
    output_usd_per_million_tokens: Decimal = Field(gt=0, allow_inf_nan=False)
    configured_output_token_allowance: int = Field(gt=0, strict=True)

    def estimate(self, request: dict) -> dict:
        for field in ("model", "quality", "size", "n", "output_format", "background", "moderation", "stream"):
            if request.get(field) != getattr(self, field) or type(request.get(field)) is not type(getattr(self, field)):
                raise budget.UnsupportedPrice(f"Image pricing identity mismatch: {field}")
        if set(request) != {"prompt", "model", "quality", "size", "n", "output_format", "background", "moderation", "stream"}:
            raise budget.UnsupportedPrice("Image request contains unsupported pricing dimensions")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > IMAGE_PROMPT_MAX_UTF8_BYTES:
            raise budget.UnsupportedPrice("Image prompt must be non-empty and at most 32000 UTF-8 bytes")
        # Mirrors P9-C text framing allowance. Output allowance is configured,
        # not enforced by Images.generate; this estimate NEVER authorizes a call.
        input_allowance = len(prompt.encode("utf-8")) + 4096
        amount = (input_allowance * self.input_usd_per_million_tokens
                  + self.configured_output_token_allowance * self.output_usd_per_million_tokens) / Decimal(1000000)
        return dict(pricing=self.model_dump(mode="json"), input_token_allowance=input_allowance,
                    configured_reservation_usd=str(amount), provider_guaranteed_maximum=False,
                    accounting="configured input/output estimate; not an invoice or proven total bound")


class MediaConfiguration(Contract):
    """Explicit media models; environment-only credentials, no fictitious billed costs."""

    tts_model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15"]
    tts_voice: Text
    image_model: Literal["gpt-image-1"] | None = None
    image_size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    image_quality: Literal["low"] = "low"
    image_pricing: ImagePricing | None = None
    video_model: Literal["sora-2", "sora-2-pro"] | None = None
    video_size: Literal["720x1280", "1280x720", "1024x1792", "1792x1024"] = "1280x720"
    video_seconds: Literal["4", "8", "12"] = "4"
    video_poll_attempts: int = Field(default=60, strict=True, gt=0, le=300)
    video_poll_interval_seconds: float = Field(default=2, gt=0, le=30, allow_inf_nan=False)
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    tts_usd_per_million_characters: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    def validate_for(self, storyboard: StoryboardPackage) -> None:
        methods = {shot.generation_method for section in storyboard.sections for shot in section.shots}
        if methods & {GenerationMethod.STATIC_IMAGE, GenerationMethod.IMAGE_TO_VIDEO} and self.image_model is None:
            raise ValueError("Media configuration requires image_model for approved image methods")
        if methods & {GenerationMethod.TEXT_TO_VIDEO, GenerationMethod.IMAGE_TO_VIDEO} and self.video_model is None:
            raise ValueError("Media configuration requires video_model for approved video methods")

    def validate_guarded_prices(self, storyboard: StoryboardPackage) -> None:
        """Reject known unsupported batch requirements before even the first TTS."""
        methods = {shot.generation_method for section in storyboard.sections for shot in section.shots}
        if methods & {GenerationMethod.TEXT_TO_VIDEO, GenerationMethod.IMAGE_TO_VIDEO}:
            raise budget.UnsupportedPrice("Guarded video is disabled: no durable asynchronous charge rule")
        if GenerationMethod.STATIC_IMAGE in methods:
            if self.image_pricing is None:
                raise budget.UnsupportedPrice("Guarded image is disabled: explicit image_pricing is missing")
            for section in storyboard.sections:
                for shot in section.shots:
                    if shot.generation_method == GenerationMethod.STATIC_IMAGE:
                        self.image_pricing.estimate(image_request(model=self.image_model, size=self.image_size,
                            quality=self.image_quality, prompt=shot.generation_prompt))
            raise budget.UnsupportedPrice("Guarded image is disabled: configured estimate is not a proven total request upper bound")
        if self.tts_model not in ("tts-1", "tts-1-hd") or self.tts_usd_per_million_characters is None:
            raise budget.UnsupportedPrice("Guarded TTS requires an explicit supported character price")


def _binary(response) -> bytes:
    try:
        data = response.read()
    finally:
        response.close()
    if not isinstance(data, bytes) or not data:
        raise ValueError("OpenAI media response must contain non-empty bytes")
    return data


class OpenAITTSProvider:
    def __init__(self, client: OpenAI, *, model: str, voice: str,
                 usd_per_million_characters: float | None = None) -> None:
        self.client, self.model, self.voice = client, model, voice
        self.usd_per_million_characters = usd_per_million_characters

    def synthesize(self, *, text: str) -> TTSResult:
        if not text.strip() or len(text) > 4096:
            raise ValueError("OpenAI speech requires non-empty text of at most 4096 characters; narration is not split or rewritten")
        data = budget.require_budget(self.client).tts(model=self.model, voice=self.voice, text=text,
            usd_per_million_characters=self.usd_per_million_characters,
            call=lambda: _binary(self.client.audio.speech.create(
                input=text, model=self.model, voice=self.voice, response_format="wav")))
        return TTSResult(audio_bytes=data, provider="openai", model=self.model)


def image_request(*, model: str, size: str, quality: str, prompt: str) -> dict:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > IMAGE_PROMPT_MAX_UTF8_BYTES:
        raise ValueError("Image prompt must be non-empty and at most 32000 UTF-8 bytes")
    if quality != "low":
        raise ValueError("Image quality must be explicitly low; auto is unsupported")
    if size not in ("1024x1024", "1024x1536", "1536x1024"):
        raise ValueError("Unsupported image size")
    return dict(prompt=prompt, model=model, n=1, output_format="png", size=size,
                quality=quality, background="opaque", moderation="auto", stream=False)


class OpenAIImageProvider:
    def __init__(self, client: OpenAI, *, model: str, size: str = "1024x1024",
                 quality: str = "low", pricing: ImagePricing | None = None) -> None:
        if model != "gpt-image-1":
            raise ValueError("Image adapter is audited for gpt-image-1 PNG output only")
        self.client, self.model, self.size = client, model, size
        self.quality, self.pricing = quality, pricing

    def generate(self, *, prompt: str) -> ImageGenerationResult:
        request = image_request(model=self.model, size=self.size, quality=self.quality, prompt=prompt)
        if self.pricing is not None:
            self.pricing.estimate(request)
        budget.require_budget(self.client).disabled("images.generate")
        response = self.client.images.generate(**request)
        if not response.data or len(response.data) != 1 or not response.data[0].b64_json:
            raise ValueError("OpenAI image response must contain exactly one base64 image; URL fallback is unsupported")
        if response.output_format not in (None, "png"):
            raise ValueError("OpenAI image output must be PNG")
        try:
            data = base64.b64decode(response.data[0].b64_json, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image transport") from exc
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("OpenAI image output must contain PNG bytes")
        return ImageGenerationResult(image_bytes=data, provider="openai", model=self.model)


class OpenAIVideoProvider:
    """Explicit text/image creation, bounded retrieval polling, then MP4 download.

    Fixed generation settings do not allocate timeline duration. Backend image
    compatibility/account access is not proven by an SDK surface or offline tests.
    Timed-out remote jobs may finish later; no job discovery/adoption is performed.
    """

    def __init__(self, client: OpenAI, *, model: str, size: str = "1280x720", seconds: str = "4",
                 poll_attempts: int = 60, poll_interval_seconds: float = 2) -> None:
        # Validate bounded polling even for direct adapter construction.
        settings = MediaConfiguration(tts_model="tts-1", tts_voice="alloy", video_model=model,
            video_size=size, video_seconds=seconds, video_poll_attempts=poll_attempts,
            video_poll_interval_seconds=poll_interval_seconds)
        self.client, self.model = client, settings.video_model
        self.size, self.seconds = settings.video_size, settings.video_seconds
        self.poll_attempts, self.poll_interval_seconds = settings.video_poll_attempts, settings.video_poll_interval_seconds

    def _generate(self, *, prompt: str, image: bytes | None = None) -> VideoGenerationResult:
        if not prompt.strip():
            raise ValueError("Video generation requires a non-empty approved prompt")
        budget.require_budget(self.client).disabled("videos.create/retrieve/download")
        kwargs = dict(prompt=prompt, model=self.model, size=self.size, seconds=self.seconds)
        if image is not None:
            if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Image-to-video requires persisted PNG bytes")
            kwargs["input_reference"] = ("reference.png", image, "image/png")
        job = self.client.videos.create(**kwargs)
        job_id = job.id
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("OpenAI video creation must return a job ID")
        for attempt in range(self.poll_attempts + 1):
            if job.id != job_id:
                raise ValueError("Video retrieval must preserve the exact created job identity")
            if job.status == "completed":
                data = _binary(self.client.videos.download_content(job_id, variant="video"))
                return VideoGenerationResult(video_bytes=data, provider="openai", model=job.model)
            if job.status == "failed":
                raise ValueError("OpenAI video generation failed")
            if job.status not in ("queued", "in_progress"):
                raise ValueError("Unsupported OpenAI video job status")
            if attempt == self.poll_attempts:
                raise TimeoutError("OpenAI video polling limit reached; no final media returned")
            time.sleep(self.poll_interval_seconds)
            job = self.client.videos.retrieve(job_id)
        raise AssertionError("Unreachable video polling state")

    def generate_from_text(self, *, prompt: str) -> VideoGenerationResult:
        return self._generate(prompt=prompt)

    def generate_from_image(self, *, image: bytes, prompt: str) -> VideoGenerationResult:
        if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Image-to-video requires persisted PNG bytes")
        return self._generate(prompt=prompt, image=image)

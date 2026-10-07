"""Adapters audited against installed openai 2.54.0; no client creation or secrets here."""
import base64
import binascii
import time
from typing import Literal

from openai import OpenAI
from pydantic import Field

from history_studio.models import GenerationMethod, StoryboardPackage
from history_studio.models.base import Contract, Text

from .image import ImageGenerationResult
from .tts import TTSResult
from .video import VideoGenerationResult


class MediaConfiguration(Contract):
    """Explicit media models; environment-only credentials, no fictitious billed costs."""

    tts_model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15"]
    tts_voice: Text
    image_model: Literal["gpt-image-1"] | None = None
    image_size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    video_model: Literal["sora-2", "sora-2-pro"] | None = None
    video_size: Literal["720x1280", "1280x720", "1024x1792", "1792x1024"] = "1280x720"
    video_seconds: Literal["4", "8", "12"] = "4"
    video_poll_attempts: int = Field(default=60, strict=True, gt=0, le=300)
    video_poll_interval_seconds: float = Field(default=2, gt=0, le=30, allow_inf_nan=False)
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)

    def validate_for(self, storyboard: StoryboardPackage) -> None:
        methods = {shot.generation_method for section in storyboard.sections for shot in section.shots}
        if methods & {GenerationMethod.STATIC_IMAGE, GenerationMethod.IMAGE_TO_VIDEO} and self.image_model is None:
            raise ValueError("Media configuration requires image_model for approved image methods")
        if methods & {GenerationMethod.TEXT_TO_VIDEO, GenerationMethod.IMAGE_TO_VIDEO} and self.video_model is None:
            raise ValueError("Media configuration requires video_model for approved video methods")


def _binary(response) -> bytes:
    try:
        data = response.read()
    finally:
        response.close()
    if not isinstance(data, bytes) or not data:
        raise ValueError("OpenAI media response must contain non-empty bytes")
    return data


class OpenAITTSProvider:
    def __init__(self, client: OpenAI, *, model: str, voice: str) -> None:
        self.client, self.model, self.voice = client, model, voice

    def synthesize(self, *, text: str) -> TTSResult:
        if not text.strip() or len(text) > 4096:
            raise ValueError("OpenAI speech requires non-empty text of at most 4096 characters; narration is not split or rewritten")
        response = self.client.audio.speech.create(input=text, model=self.model, voice=self.voice, response_format="wav")
        return TTSResult(audio_bytes=_binary(response), provider="openai", model=self.model)


class OpenAIImageProvider:
    def __init__(self, client: OpenAI, *, model: str, size: str = "1024x1024") -> None:
        if model != "gpt-image-1":
            raise ValueError("Image adapter is audited for gpt-image-1 PNG output only")
        self.client, self.model, self.size = client, model, size

    def generate(self, *, prompt: str) -> ImageGenerationResult:
        if not prompt.strip():
            raise ValueError("Image generation requires a non-empty approved prompt")
        response = self.client.images.generate(prompt=prompt, model=self.model, n=1, output_format="png", size=self.size)
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

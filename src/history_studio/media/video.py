"""Video transport boundary with explicit approved text/image execution methods."""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class VideoGenerationResult:
    video_bytes: bytes
    provider: str
    model: str
    video_format: Literal["mp4"] = "mp4"


class VideoProvider(Protocol):
    def generate_from_text(self, *, prompt: str) -> VideoGenerationResult: ...

    def generate_from_image(self, *, image: bytes, prompt: str) -> VideoGenerationResult: ...

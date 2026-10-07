"""Image transport boundary; transient bytes never become JSON contract fields."""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ImageGenerationResult:
    image_bytes: bytes
    provider: str
    model: str
    image_format: Literal["png"] = "png"


class ImageProvider(Protocol):
    """Execute the approved prompt without semantic reasoning or rewriting."""

    def generate(self, *, prompt: str) -> ImageGenerationResult: ...

"""TTS transport boundary; transient bytes are never durable JSON contracts."""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes
    provider: str
    model: str
    audio_format: Literal["wav"] = "wav"


class TTSProvider(Protocol):
    """Produce audio from exact narration text; Runtime owns duration measurement."""

    def synthesize(self, *, text: str) -> TTSResult: ...

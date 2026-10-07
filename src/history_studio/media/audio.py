"""Deterministic v1 PCM WAV duration inspection without external decoders."""
import io
import math
import struct
import wave


class AudioDurationError(ValueError):
    """Audio is empty, malformed, truncated, unsupported or has no duration."""


def measure_wav_duration(data: bytes) -> float:
    """Measure actual frames / sample rate, also checking complete PCM frame data."""
    if not isinstance(data, bytes) or len(data) < 12:
        raise AudioDurationError("Audio must contain non-empty valid PCM WAV bytes")
    if (data[:4] != b"RIFF" or data[8:12] != b"WAVE"
            or struct.unpack_from("<I", data, 4)[0] + 8 != len(data)):
        raise AudioDurationError("Malformed or truncated WAV container")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            frame_size = audio.getnchannels() * audio.getsampwidth()
            if audio.getcomptype() != "NONE" or rate <= 0 or frames <= 0:
                raise AudioDurationError("WAV must contain positive-duration PCM audio")
            # Reading one extra frame detects partial trailing frames as well as
            # truncation. Header-only duration claims do not count as actual audio.
            if len(audio.readframes(frames + 1)) != frames * frame_size:
                raise AudioDurationError("WAV frame data is truncated or incomplete")
            duration = frames / rate
    except (wave.Error, EOFError, struct.error) as exc:
        raise AudioDurationError("Malformed or unsupported PCM WAV audio") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AudioDurationError("WAV duration must be finite and positive")
    return duration
